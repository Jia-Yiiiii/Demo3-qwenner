import json
import random
from collections import Counter

import numpy as np
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import (
    LoraConfig,
    PeftModel,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training,
)


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



class NerUtils:
    @staticmethod
    def load_model(config):
        mode = config.get("train_mode", "lora").lower()
        load_args = {
            "torch_dtype": torch.bfloat16,
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
        }

        if mode == "qlora":
            load_args["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )

        model = AutoModelForCausalLM.from_pretrained(
            config["model_path"], **load_args
        )

        if mode == "qlora":
            model = prepare_model_for_kbit_training(
                model,
                use_gradient_checkpointing=config.get("gradient_checkpointing", False),
                gradient_checkpointing_kwargs={"use_reentrant": False},
            )
        elif config.get("gradient_checkpointing", False):
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )

        model = get_peft_model(
            model,
            LoraConfig(
                target_modules=config["lora_target_modules"],
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=config["lora_r"],
                lora_alpha=config["lora_alpha"],
                lora_dropout=config["lora_dropout"],
            ),
        )
        model.enable_input_require_grads()
        model.config.use_cache = False
        model.print_trainable_parameters()
        return model

    @staticmethod
    def load_trained_model(config):
        load_args = {
            "torch_dtype": torch.bfloat16,
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
        }
        if config.get("train_mode", "lora").lower() == "qlora":
            load_args["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
        model = AutoModelForCausalLM.from_pretrained(
            config["model_path"], **load_args
        )
        return PeftModel.from_pretrained(
            model, config.get("adapter_path", config["output_dir"])
        )

    @staticmethod
    def get_entities(text):
        if not isinstance(text, str) or not text.strip():
            return []
        text = text.strip()
        if text.startswith("entities"):
            text = "{" + text + "}"
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            return []

        entities = []
        for entity in data.get("entities", []):
            name = entity.get("name", "")
            entity_type = entity.get("type", "")
            if name and entity_type:
                entities.append((name, entity_type))
        return entities

    @staticmethod
    def compute_metrics(pred_list, gold_list):
        tp = 0
        fp = 0
        fn = 0
        for p, g in zip(pred_list, gold_list):
            temp_tp = 0
            gold_copy = list(g)
            for ent_pred in p:
                for ent_gold in gold_copy:
                    if ent_pred == ent_gold:
                        temp_tp += 1
                        gold_copy.remove(ent_gold)
                        break
            temp_fp = len(p) - temp_tp
            temp_fn = len(g) - temp_tp
            tp += temp_tp
            fp += temp_fp
            fn += temp_fn

        if tp + fp != 0:
            precision = tp / (tp + fp)
        else:
            precision = 0.0
        if tp + fn != 0:
            recall = tp / (tp + fn)
        else:
            recall = 0.0
        if precision + recall != 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0
        return {"precision": precision, "recall": recall, "f1": f1}
