import argparse
import gc
import json
import os
import torch
import bitsandbytes as bnb
from torch.optim import AdamW
from tqdm import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from utils import NerUtils, set_seed
from data_process import NERSFTDataset
import swanlab


class Trainer:
    def __init__(self, config):
        set_seed(config.get("seed", 42))
        self.config = config
        self.grad_accum_steps = config["gradient_accumulation_steps"]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = None
        self.model = None
        self.train_dataset = None
        self.train_loader = None
        self.dev_dataset = None
        self.dev_loader = None
        self.eval_dataset = None
        self.eval_loader = None
        self.optimizer = None
        self.scheduler = None
        self.best_f1 = 0.0
        self.counter = 0

    def load_data(self):
        dataset_kwargs = dict(
            tokenizer=self.tokenizer,
            max_seq_length=self.config["max_len"],
            system_prompt=self.config["system_prompt"],
        )
        self.train_dataset = NERSFTDataset(file=self.config["train_data"], is_train=True, **dataset_kwargs)
        self.dev_dataset = NERSFTDataset(file=self.config["dev_data"], is_train=False, **dataset_kwargs)
        self.eval_dataset = NERSFTDataset(file=self.config["eval_data"], is_train=False, **dataset_kwargs)
        self.train_loader = self.train_dataset.get_dataloader(batch_size=self.config["batch_size"])
        self.dev_loader = self.dev_dataset.get_dataloader(batch_size=self.config["dev_batch_size"])
        self.eval_loader = self.eval_dataset.get_dataloader(batch_size=self.config["eval_batch_size"])

    def init_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config["model_path"], trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.tokenizer.padding_side = "left"
        self.model = NerUtils.load_model(self.config)
        self.model.to(self.device)
        if self.config.get("gradient_checkpointing", False):
            self.model.gradient_checkpointing_enable()
            if hasattr(self.model, "config"):
                self.model.config.use_cache = False
        params = [p for p in self.model.parameters() if p.requires_grad]
        train_mode = self.config["train_mode"]
        train_mode = train_mode.lower()
        if train_mode == "qlora":
            self.optimizer = bnb.optim.PagedAdamW8bit(
                params,
                lr=self.config["learning_rate"],
                weight_decay=self.config["weight_decay"],
            )
        else:
            self.optimizer = AdamW(
                params,
                lr=self.config["learning_rate"],
                weight_decay=self.config["weight_decay"],
            )

    def save_model(self):
        path = os.path.join(self.config["output_dir"], f"best-f1_{self.best_f1:.4f}")
        os.makedirs(path, exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    def save_checkpoint(self, epoch, global_step):
        os.makedirs(self.config["output_dir"], exist_ok=True)
        adapter_path = os.path.join(self.config["output_dir"], "adapter_checkpoint")
        self.model.save_pretrained(adapter_path)
        state = {
            "epoch": epoch,
            "best_f1": self.best_f1,
            "global_step": global_step,
            "counter": self.counter,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
        }
        ckpt_path = os.path.join(self.config["output_dir"], "checkpoint.pt")
        torch.save(state, ckpt_path)

    def load_checkpoint(self):
        ckpt_path = os.path.join(self.config["output_dir"], "checkpoint.pt")
        adapter_path = os.path.join(self.config["output_dir"], "adapter_checkpoint")
        if not os.path.exists(ckpt_path) or not os.path.exists(adapter_path):
            return 0, 0
        self.model.load_adapter(adapter_path, adapter_name="default")
        state = torch.load(ckpt_path, map_location="cpu")
        self.optimizer.load_state_dict(state["optimizer_state_dict"])
        self.scheduler.load_state_dict(state["scheduler_state_dict"])
        self.best_f1 = state["best_f1"]
        self.counter = state["counter"]
        global_step = state.get("global_step", 0)
        if global_step == 0 and state["epoch"] > 0:
            steps_per_epoch = max(1, (len(self.train_loader) + self.grad_accum_steps - 1) // self.grad_accum_steps)
            global_step = state["epoch"] * steps_per_epoch

        print(f"Resumed from epoch {state['epoch'] + 1}, step {global_step}, best F1={self.best_f1:.4f}")
        return state["epoch"] + 1, global_step

    def train(self):
        swanlab.init(project="SFT-NER", resume=False, experiment_name="SFT-NER")
        len_loader = len(self.train_loader)
        accum_step = self.grad_accum_steps
        steps_per_epoch = (len_loader + accum_step - 1) // accum_step
        steps_per_epoch = max(1, steps_per_epoch)
        total_steps = steps_per_epoch * self.config["epochs"]
        warmup_steps = self.config.get("warmup_steps")
        if warmup_steps is None:
            warmup_steps = int(total_steps * self.config.get("warmup_ratio", 0.1))
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer, warmup_steps, total_steps)
        start_epoch, global_step = self.load_checkpoint()
        for epoch in range(start_epoch, self.config["epochs"]):
            self.model.train()
            total_loss = 0.0
            count = 0
            progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}")
            self.optimizer.zero_grad(set_to_none=True)
            for step, batch in enumerate(progress_bar):
                model_inputs = {
                    "input_ids": batch["input_ids"].to(self.device),
                    "attention_mask": batch["attention_mask"].to(self.device),
                    "labels": batch["labels"].to(self.device)
                }
                with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
                    loss = self.model(**model_inputs).loss
                loss = loss / self.grad_accum_steps
                total_loss += loss.item() * self.grad_accum_steps
                count += 1
                loss.backward()

                if (step + 1) % self.grad_accum_steps == 0 or (step + 1) == len(self.train_loader):
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config["max_grad_norm"])
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    global_step += 1
            metrics = self.evaluate(self.dev_loader, f"Epoch {epoch + 1} [Eval]")
            avg_loss = total_loss / max(count, 1)
            swanlab.log({"train/loss": avg_loss, "eval/f1": metrics["f1"]}, step=global_step)
            print(f"Epoch {epoch + 1} |Step {global_step}|  Loss: {avg_loss:.4f} | F1: {metrics['f1']:.4f}")
            if metrics["f1"] > self.best_f1:
                self.best_f1 = metrics["f1"]
                self.counter = 0
                self.save_model()
            else:
                self.counter += 1
                if self.counter >= self.config["patience"]:
                    print(f"Early stopping at epoch {epoch + 1}, best F1={self.best_f1:.4f}")
                    break
            self.save_checkpoint(epoch, global_step)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        test_metrics = self.evaluate(self.eval_loader, "[Test]")
        swanlab.log({"test/f1": test_metrics["f1"]}, step=global_step)
        print(f"Final F1: {test_metrics['f1']:.4f}")
        swanlab.finish()

    def evaluate(self, loader, desc="[Eval]"):
        self.model.eval()
        all_preds, all_golds = [], []
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        original_use_cache = getattr(self.model.config, "use_cache", False)
        self.model.config.use_cache = True
        with torch.no_grad():
            for batch in tqdm(loader, desc=desc):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)

                with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
                    generated = self.model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=self.config["max_new_tokens"],
                        do_sample=False,
                        pad_token_id=self.tokenizer.pad_token_id,
                    )

                for i in range(input_ids.shape[0]):
                    real_prompt_len = batch["prompt_len"][i].item()
                    total_seq_len = input_ids.shape[1]
                    pad_num = total_seq_len - real_prompt_len
                    text = self.tokenizer.decode(generated[i, pad_num + real_prompt_len:], skip_special_tokens=True)
                    predicted_entities = NerUtils.get_entities(text)
                    gold_entities = batch["entities"][i]
                    all_preds.append(predicted_entities)
                    all_golds.append(gold_entities)

                del input_ids, attention_mask, generated
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        self.model.config.use_cache = original_use_cache
        return NerUtils.compute_metrics(all_preds, all_golds)

    def run(self):
        self.init_model()
        self.load_data()
        self.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", required=True)
    args = parser.parse_args()
    with open(args.config_path, encoding="utf-8") as f:
        config = json.load(f)
    Trainer(config).run()
