#!/usr/bin/env python3
import argparse, time, os, torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from datasets import load_dataset
import mlflow, mlflow.transformers

def train_llm(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment("llm-fine-tuning")

    with mlflow.start_run():
        # Phase 7: Tag trial number if running under AutoML
        trial_number = os.getenv("OPTUNA_TRIAL_NUMBER")
        if trial_number:
            mlflow.set_tag("trial_number", trial_number)
            mlflow.set_tag("automl", "true")

        mlflow.log_param("model", args.model_name)
        mlflow.log_param("epochs", args.epochs)
        mlflow.log_param("lr", args.lr)
        mlflow.log_param("batch_size", args.batch_size)
        mlflow.log_param("device", str(device))

        start_time = time.time()

        print(f"Loading model: {args.model_name}")
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        model = AutoModelForCausalLM.from_pretrained(args.model_name).to(device)

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        print("Loading dataset...")
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1")

        def tokenize_fn(examples):
            return tokenizer(examples["text"], truncation=True, max_length=512)

        tokenized = dataset.map(tokenize_fn, batched=True, remove_columns=["text"])

        training_args = TrainingArguments(
            output_dir="./results",
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            learning_rate=args.lr,
            logging_steps=10,
            save_strategy="epoch",
            report_to=[]
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized["train"],
            data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
        )

        print("Starting training...")
        trainer.train()

        duration = time.time() - start_time
        mlflow.log_metric("duration_seconds", duration)

        gpu_cost = ((duration / 3600) * float(os.getenv("GPU_HOURLY_RATE", "0.25"))) if torch.cuda.is_available() else 0
        cpu_cost = (duration / 3600) * 4 * float(os.getenv("CPU_HOURLY_RATE", "0.05"))
        total_cost = cpu_cost + gpu_cost

        mlflow.log_metric("gpu_cost_usd", gpu_cost)
        mlflow.log_metric("cpu_cost_usd", cpu_cost)
        mlflow.log_metric("total_cost_usd", total_cost)

        print(f"\nComplete! Duration: {duration:.1f}s ({duration/60:.1f}m), Cost: ${total_cost:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="distilgpt2")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--batch-size", type=int, default=4)
    train_llm(parser.parse_args())
