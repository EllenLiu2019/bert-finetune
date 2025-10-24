import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from datasets import load_dataset
import evaluate
import wandb

from transformers import (
    BertTokenizer,
    DataCollatorWithPadding,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments,
)

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ModelArguments:
    """
    模型相关参数
    """
    model_name_or_path: str = field(
        default="../models/bert-base-uncased",
        metadata={"help": "预训练模型路径"}
    )
    tokenizer_name: Optional[str] = field(
        default="../tokenizer/bert-base-uncased",
        metadata={"help": "分词器路径"}
    )
    num_labels: int = field(
        default=2,
        metadata={"help": "分类标签数"}
    )
    hub_model_id: Optional[str] = field(
        default="bert-finetuned-claim-detection",
        metadata={"help": "模型保存路径"}
    )
    output_dir: str = field(
        default="../results",
        metadata={"help": "是否将模型推送到Hugging Face Hub"}
    )

@dataclass
class DataTrainingArguments:
    """
    数据训练相关参数
    """
    dataset_name: str = field(
        default="Nithiwat/claim-detection",
        metadata={"help": "数据集名称"}
    )
    train_size: int = field(
        default=8000,
        metadata={"help": "训练集大小"}
    )
    max_seq_length: int = field(
        default=128,
        metadata={"help": "最大序列长度"}
    )


def compute_metrics(eval_pred):
    accuracy_metric = evaluate.load("accuracy")
    f1_metric = evaluate.load("f1")

    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    accuracy = accuracy_metric.compute(predictions=predictions, references=labels)
    f1 = f1_metric.compute(predictions=predictions, references=labels, average="weighted")

    return {
        "accuracy": accuracy["accuracy"],
        "f1": f1["f1"],
    }

def main():
    def tokenize_function(example):
        result = tokenizer(example["text"], truncation=True)
        result["labels"] = example["checkworthiness"]
        return result

    wandb.init(project="bert-fine-tuning",name="evaluation")

    model_args = ModelArguments()
    data_args = DataTrainingArguments()

    logger.info("Loading dataset...")
    dataset = load_dataset(data_args.dataset_name)

    tokenizer = BertTokenizer.from_pretrained(model_args.tokenizer_name)

    logger.info("Tokenizing dataset...")
    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    logger.info("Loading model...")
    model = BertForSequenceClassification.from_pretrained(model_args.model_name_or_path, num_labels=2)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(11000))
    small_eval_dataset = tokenized_datasets["test"]

    training_args = TrainingArguments(
        output_dir=model_args.output_dir,
        num_train_epochs=2,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        gradient_accumulation_steps=1,
        eval_strategy="steps",
        eval_steps=50,
        logging_steps=50,
        save_strategy="steps",
        save_steps=100,
        max_grad_norm=5.0,
        warmup_steps=400,
        learning_rate=2e-5,
        weight_decay=1e-2,
        lr_scheduler_type="cosine",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        push_to_hub=True,
        hub_model_id=model_args.hub_model_id,
        fp16=True,
        report_to="wandb",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=small_train_dataset,
        eval_dataset=small_eval_dataset,
        data_collator=data_collator,
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
    )

    logger.info("Starting training...")
    trainer.train()

    trainer.push_to_hub()

if __name__ == "__main__":
    main()