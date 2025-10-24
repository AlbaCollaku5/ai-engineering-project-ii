"""Placeholder and instructions for PEFT/LoRA fine-tuning integration."""
def instructions():
    return {
        "notes": "This file is a placeholder. Use HuggingFace Transformers + PEFT (loralib) to fine-tune smaller models.",
        "example_steps": [
            "Install transformers, accelerate, peft",
            "Prepare dataset in HF Dataset format",
            "Use Trainer with peft.LoraConfig",
        ]
    }
