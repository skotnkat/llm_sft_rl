from datasets import load_dataset
import os
import mlflow
import yaml
import argparse
import torch
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer
from trl import GRPOTrainer, GRPOConfig

os.environ["MLFLOW_TRACKING_URI"] = "http://localhost:5000"
mlflow.set_tracking_uri("http://localhost:5000")


def load_grpo_data(dataset_name: str, sample_num: int = 0):
    ds = load_dataset(dataset_name)
    train_grpo = ds["train_prefs"].select_columns(["prompt"])
    eval_grpo = ds["test_prefs"].select_columns(["prompt"])

    if sample_num:
        train_grpo = train_grpo.shuffle(seed=42).select(range(min(sample_num, len(train_grpo))))
    
    return train_grpo, eval_grpo



def get_reward_fns(reward_model, reward_tokenizer, max_completion_length: int = 512):
    # TODO: wrap reward_model as callable + combine with get_repetition_penalty_reward,
    # get_cosine_scaled_reward, get_soft_overlong_punishment from trl.trainer.grpo_trainer
    return [reward_model]


def get_trainer(config, grpo_train_ds, grpo_eval_ds, reward_fns, reward_tokenizer):
    model = AutoModelForCausalLM.from_pretrained(
        config["model_name"],
        device_map="auto",
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )

    tokenizer_cfg = config["tokenizer"]
    tokenizer = AutoTokenizer.from_pretrained(
        config["model_name"],
        padding_side=tokenizer_cfg["train_padding_side"],
    )

    peft_config = None
    if config["lora"].get("use_lora"):
        lora_cfg = {k: v for k, v in config["lora"].items() if k != "use_lora"}
        peft_config = LoraConfig(**lora_cfg)

    grpo_trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        args=GRPOConfig(**config["training"]),
        train_dataset=grpo_train_ds,
        eval_dataset=grpo_eval_ds,
        reward_funcs=reward_fns,
        reward_processing_classes=reward_tokenizer,
        peft_config=peft_config,
    )
    
    return grpo_trainer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_name", type=str, default="config.yaml")
    args = parser.parse_args()

    with open(args.config_name, "r") as f:
        config = yaml.safe_load(f)

    mlflow.set_experiment(config["experiment_name"])

    train_grpo_ds, eval_grpo_ds = load_grpo_data(config["dataset_name"], config.get("sample_num", 0))
    print("--- Dataset Loaded ---")

    reward_model_path = config["reward_model_path"]
    reward_model = AutoModelForSequenceClassification.from_pretrained(
        reward_model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    reward_tokenizer = AutoTokenizer.from_pretrained(reward_model_path)
    print("--- Reward Model Loaded ---")

    reward_fns = get_reward_fns(reward_model, reward_tokenizer, config["training"]["max_completion_length"])

    trainer = get_trainer(config, train_grpo_ds, eval_grpo_ds, reward_fns, reward_tokenizer)
    print("--- Trainer Obtained ---")

    with mlflow.start_run(run_name=config["run_name"]):
        mlflow.log_params({
            "model":               config["model_name"],
            "dataset":             config["dataset_name"],
            "reward_model_path":   reward_model_path,
        })
        mlflow.log_artifact(args.config_name)

        print("--- Starting Training ---")
        trainer.train()
        trainer.save_state()
        mlflow.flush_async_logging()

        for i, entry in enumerate(trainer.state.log_history):
            metrics = {k: v for k, v in entry.items() if k.startswith("eval_") and isinstance(v, (int, float))}
            if metrics:
                mlflow.log_metrics(metrics, step=entry.get("step", i))

        save_path = os.path.join(config["training"]["output_dir"], "final_model")

        if config["lora"].get("use_lora"):
            trainer.model = trainer.model.merge_and_unload()

        trainer.save_model(save_path)
        mlflow.log_artifacts(save_path, artifact_path="model")
