import os
from datasets import load_dataset
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments

def tokenize_function(examples):
    tokenized = tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)
    tokenized["labels"] = tokenized["input_ids"].copy()  # 关键：labels = input_ids
    return tokenized

def main():
    global tokenizer  # 方便tokenize_function使用

    # 加载小模型和对应tokenizer
    model_name = "gpt2"
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)

    # GPT2没有默认pad_token，手动设置，否则报错
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id

    # 加载简单文本数据集
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train[:1%]")
    tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

    # 训练参数，启用 fp16 和 DeepSpeed 配置
    training_args = TrainingArguments(
        output_dir="./results",
        overwrite_output_dir=True,
        num_train_epochs=1,
        per_device_train_batch_size=8,
        save_steps=10,
        save_total_limit=2,
        logging_steps=10,
        deepspeed="./ds_config.json",
        fp16=True,
        logging_dir="./logs",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets,
        tokenizer=tokenizer,
    )

    trainer.train()
    trainer.save_model(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)

if __name__ == "__main__":
    main()

