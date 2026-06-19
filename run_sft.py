from datasets import load_dataset
import os
import mlflow
import yaml
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig


os.environ["MLFLOW_TRACKING_URI"] = "http://localhost:5000"
mlflow.set_tracking_uri("http://localhost:5000")


def load_data(dataset_name: str, sample_num: int = 0):
    ds = load_dataset(dataset_name)
    train_sft = ds["train_sft"].select_columns(["messages"])
    eval_sft = ds["test_sft"].select_columns(["messages"])

    if sample_num:
        train_sft = train_sft.shuffle(seed=42).select(range(min(sample_num, len(train_sft))))
    
    return train_sft, eval_sft

def get_trainer(train_config: dict, train_ds, end_ds):
    model_name = train_config["model_name"]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    if train_config["training"]["assistant_only_loss"]:
        with open("granite_chat_template_assistant_only_loss.txt", "r") as file:
            tokenizer.chat_template = file.read()
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",  # required for packing
    )
    train_args = SFTConfig(**train_config["training"])

    if train_config.get("lora", {}).get("use_lora", False):
        lora_config = train_config["lora"]
        peft_config = LoraConfig(
            r=lora_config["lora_rank"],
            lora_alpha=lora_config["lora_alpha"],
            target_modules=lora_config["target_modules"],
            task_type="CAUSAL_LM",
        )
        trainer = SFTTrainer(
            model=model,
            processing_class=tokenizer,
            args=train_args,
            train_dataset=train_ds,
            eval_dataset=end_ds,
            peft_config=peft_config,
        )
    else:
        trainer = SFTTrainer(
            model=model,
            processing_class=tokenizer,
            args=train_args,
            train_dataset=train_ds,
            eval_dataset=end_ds
        )
    
    return trainer
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--config_name", type=str, default="sft_config.yaml")
    
    args = parser.parse_args()
    with open(args.config_name, 'r') as file:
        train_config = yaml.safe_load(file)
    
    # Name your experiment
    mlflow.set_experiment(train_config["experiment_name"])
    train_ds, eval_ds = load_data(train_config["dataset_name"], train_config["sample_num"])
    print("--- Dataset Loaded ---")
    
        
    trainer = get_trainer(train_config, train_ds, eval_ds)
    print(trainer.callback_handler.callbacks)
    print("--- Trainer Obtained---")
    
    with mlflow.start_run(run_name=train_config["run_name"]):
        mlflow.log_params({
            "model":   train_config["model_name"],
            "dataset": train_config["dataset_name"],
        })
        
        print("--- Starting Training ---")
        trainer.train()
        trainer.save_state()
        mlflow.flush_async_logging()

        for i, entry in enumerate(trainer.state.log_history):
            metrics = {k: v for k, v in entry.items() if k.startswith("eval_") and isinstance(v, (int, float))}
            if metrics:
                step = entry.get("step", i)
                mlflow.log_metrics(metrics, step=step)
        
        # Save model & log it
        save_path = os.path.join(train_config["training"]["output_dir"], "final_model")
        
        if train_config.get("lora", {}).get("use_lora", False):
            trainer.model = trainer.model.merge_and_unload()
        
        trainer.save_model(save_path)

        mlflow.log_artifacts(save_path, artifact_path="model")
