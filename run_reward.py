from datasets import load_dataset
import os
import mlflow
import yaml
import argparse
import torch
from peft import LoraConfig
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from trl import RewardConfig, RewardTrainer

os.environ["MLFLOW_TRACKING_URI"] = "http://localhost:5000"
mlflow.set_tracking_uri("http://localhost:5000")


def load_reward_data(dataset_name: str, sample_num: int = 0):
    ds = load_dataset(dataset_name)
    train_reward = ds["train_prefs"].select_columns(["chosen", "rejected"])
    eval_reward = ds["test_prefs"].select_columns(["chosen", "rejected"])

    if sample_num:
        train_reward = train_reward.shuffle(seed=42).select(range(min(sample_num, len(train_reward))))

    return train_reward, eval_reward


def train_reward_model(config: dict, train_ds, eval_ds):
    output_dir = config["reward_model_output_dir"]

    if os.path.exists(output_dir):
        return AutoModelForSequenceClassification.from_pretrained(output_dir)

    reward_cfg = dict(config["reward_training"])
    if "model_init_kwargs" in reward_cfg and "dtype" in reward_cfg["model_init_kwargs"]:
        reward_cfg["model_init_kwargs"]["dtype"] = getattr(torch, reward_cfg["model_init_kwargs"]["dtype"])

    peft_config = LoraConfig(
        **config["reward_lora"],
        modules_to_save=["score"],  # required to train the reward head when base model is causal LM
    )

    tokenizer = AutoTokenizer.from_pretrained(config["reward_model"])
    trainer = RewardTrainer(
        model=config["reward_model"],  # TODO: Use smaller Qwen (max 1B) for reward model training
        args=RewardConfig(**reward_cfg),
        processing_class=tokenizer,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        peft_config=peft_config,
    )
    trainer.train()

    model = trainer.model.merge_and_unload()
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_name", type=str, default="reward_config.yaml")
    args = parser.parse_args()

    with open(args.config_name, "r") as f:
        config = yaml.safe_load(f)

    mlflow.set_experiment(config["experiment_name"])

    train_ds, eval_ds = load_reward_data(config["dataset_name"], config.get("sample_num", 0))
    print("--- Dataset Loaded ---")

    with mlflow.start_run(run_name=config["run_name"]):
        mlflow.log_params({
            "model":   config["reward_model"],
            "dataset": config["dataset_name"],
        })
        mlflow.log_artifact(args.config_name)

        train_reward_model(config, train_ds, eval_ds)
        print("--- Reward Model Trained ---")
