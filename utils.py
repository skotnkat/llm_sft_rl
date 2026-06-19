def check_assistant_only_loss(trainer, tokenizer, max_tokens: int = 1024):
    """Decode one training batch and show which tokens get a real loss label."""
    batch = next(iter(trainer.get_train_dataloader()))
    input_ids = batch["input_ids"][0].tolist()
    labels = batch["labels"][0].tolist()

    parts = []
    for tok_id, label in list(zip(input_ids, labels))[:max_tokens]:
        text = tokenizer.decode([tok_id])
        parts.append(f"[{text}]" if label != -100 else text)
    print("".join(parts))