import argparse
import json

import lm_eval

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", type=str, help="Path to the SFT checkpoint")
    parser.add_argument("tasks", type=str, help="Comma-separated lm-eval task names, e.g. ifeval,mmlu,gsm8k")
    args = parser.parse_args()
    
    results = lm_eval.simple_evaluate(
        model="vllm",
        model_args=f"pretrained={args.model_path},dtype=bfloat16",
        tasks=args.tasks.split(","),
        num_fewshot=0,
        batch_size="auto",
        apply_chat_template=True,
    )
    
    print(json.dumps(results["results"], indent=2, default=str))
    
    with open("eval_results.json", "w") as f:
        json.dump(results["results"], f, indent=2, default=str)