import argparse

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", type=str)
    parser.add_argument("prompt", type=str, help="Prompt to send to the model")
    parser.add_argument("--max_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=int, default=0.7)
    args = parser.parse_args()
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    chat_prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": args.prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )
    
    llm = LLM(model=args.model_path, dtype="bfloat16")
    output = llm.generate([chat_prompt], SamplingParams(max_tokens=args.max_tokens, temperature=args.temperature))
    
    print(f'Prompt: {args.prompt}')
    print(f'Answer: {output[0].outputs[0].text}')