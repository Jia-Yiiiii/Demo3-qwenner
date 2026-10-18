# Demo3-QwenNER

基于 Qwen2.5-7B 和 LoRA/QLoRA 的生物医学基因命名实体识别项目。

---

## 项目结构

```
Demo3-qwenner/
├── configs/
│   ├── lora-all-line.json
│   └── qlora-all-line.json
├── data/
│   ├── train.json
│   ├── dev.json
│   └── test.json
├── model/
│   └── Qwen2.5-7B/
├── data_process.py
├── modeldownload.py
├── template.py
├── train.py
├── utils.py
└── README.md
```

---

## 环境安装

```bash
pip install -r requirements.txt
```

> **注意**：由于 `torch==2.9.1+cu128` 是 CUDA 12.8 专用版本，建议显式指定 PyTorch 官方索引源：
>
> ```bash
> pip install torch==2.9.1+cu128 --index-url https://download.pytorch.org/whl/cu128
> pip install -r requirements.txt --no-deps
> ```

---

## 下载模型

项目使用 ModelScope 下载 Qwen2.5-7B：

```bash
python modeldownload.py
```

模型默认保存到：

```
./model/Qwen2.5-7B
```

如果已经准备好本地模型，也可以直接修改配置文件中的 `model_path`。

---

## 训练 LoRA

运行 LoRA 配置：

```bash
python train.py --config_path configs/lora-all-line.json
```

LoRA 配置中的主要参数：

| 参数 | 值 |
|------|-----|
| train_mode | lora |
| batch_size | 4 |
| gradient_accumulation_steps | 1 |
| epochs | 9 |
| learning_rate | 0.00002 |
| max_len | 512 |
| lora_r | 16 |
| lora_alpha | 32 |
| lora_dropout | 0.05 |

---

## 训练 QLoRA

运行 QLoRA 配置：

```bash
python train.py --config_path configs/qlora-all-line.json
```

QLoRA 配置中的主要参数：

| 参数 | 值 |
|------|-----|
| train_mode | qlora |
| batch_size | 4 |
| gradient_accumulation_steps | 1 |
| epochs | 9 |
| learning_rate | 0.00002 |
| max_len | 512 |
| lora_r | 16 |
| lora_alpha | 32 |
| lora_dropout | 0.05 |

---

## 输出文件

训练结果默认保存在配置文件指定的 `output_dir` 中。其中：

- **best-f1_***：验证集 F1 最优时保存的 adapter
- **adapter_checkpoint**：用于断点恢复的 adapter
- **checkpoint.pt**：保存 epoch、optimizer、scheduler 和训练状态

---

## 训练流程

训练过程包括以下步骤：

1. 加载 tokenizer
2. 加载 Qwen2.5-7B
3. 添加 LoRA adapter
4. 加载训练集、验证集和测试集
5. 进行监督微调
6. 每轮训练后在验证集上生成实体结果
7. 计算 Precision、Recall 和 F1（最终只打印 F1 值）
8. 保存验证集 F1 最优的模型
9. 训练结束后加载最佳模型
10. 在测试集上进行最终评估

---

## 实验结果

### LoRA

- **best-f1: 0.8453**
- **test-f1: 0.8324**

![LoRA 训练结果](https://github.com/user-attachments/assets/74b89d95-fb10-4985-b0da-2137f1aee93a)

![LoRA 训练结果](https://github.com/user-attachments/assets/04084e42-6784-4663-adc1-161f9bd2368a)

![LoRA 训练结果](https://github.com/user-attachments/assets/b3dbb777-d891-4303-90e5-afe033c3a5a4)

### LoRA 显存说明

![LoRA 显存](https://github.com/user-attachments/assets/b04ae8f6-b5f5-49b7-85bd-9f4943667594)

### QLoRA

- **best-f1: 0.8472**
- **test-f1: 0.8362**

![QLoRA 训练结果](https://github.com/user-attachments/assets/779c2ed6-7811-4259-ad83-2cf0f737ff81)

![QLoRA 训练结果](https://github.com/user-attachments/assets/4bb6a340-befb-47c0-9fe8-0f85e94b2fee)

![QLoRA 训练结果](https://github.com/user-attachments/assets/75a96e0d-cc59-45c2-925d-6ebc4854a3cd)

### QLoRA 显存说明

![QLoRA 显存](https://github.com/user-attachments/assets/205bfda0-79f3-4b19-a2a1-c8a4083a4de0)
