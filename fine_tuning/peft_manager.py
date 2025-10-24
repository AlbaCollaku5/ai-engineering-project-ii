"""
Fine-Tuning & PEFT Implementation with LoRA
"""
import os
import torch
import json
from typing import Dict, Any, List, Optional, Tuple
from datasets import Dataset, load_dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, 
    TrainingArguments, Trainer, DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
import numpy as np
from monitoring.advanced_monitoring import advanced_metrics


class FineTuningManager:
    """Manager for fine-tuning models with PEFT/LoRA"""
    
    def __init__(self, 
                 base_model_name: str = "microsoft/DialoGPT-small",
                 output_dir: str = "./fine_tuned_models",
                 use_quantization: bool = True):
        self.base_model_name = base_model_name
        self.output_dir = output_dir
        self.use_quantization = use_quantization
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize components
        self.tokenizer = None
        self.model = None
        self.peft_model = None
        self.training_args = None
        
        # Load base model and tokenizer
        self._load_base_model()
    
    def _load_base_model(self):
        """Load base model and tokenizer"""
        try:
            print(f"Loading base model: {self.base_model_name}")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Configure quantization if enabled
            quantization_config = None
            if self.use_quantization:
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                )
            
            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                self.base_model_name,
                quantization_config=quantization_config,
                device_map="auto",
                torch_dtype=torch.float16 if self.use_quantization else torch.float32
            )
            
            print("SUCCESS: Base model loaded successfully")
            
        except Exception as e:
            print(f"ERROR: Failed to load base model: {e}")
            # Fallback to a smaller model
            self._load_fallback_model()
    
    def _load_fallback_model(self):
        """Load a smaller fallback model"""
        try:
            print("Loading fallback model: distilgpt2")
            self.base_model_name = "distilgpt2"
            
            self.tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.model = AutoModelForCausalLM.from_pretrained(
                "distilgpt2",
                torch_dtype=torch.float32
            )
            
            print("SUCCESS: Fallback model loaded")
            
        except Exception as e:
            print(f"ERROR: Failed to load fallback model: {e}")
            raise
    
    def setup_lora(self, 
                   r: int = 16,
                   lora_alpha: int = 32,
                   lora_dropout: float = 0.1,
                   target_modules: Optional[List[str]] = None) -> None:
        """Setup LoRA configuration"""
        try:
            if target_modules is None:
                # Default target modules for different model types
                if "gpt" in self.base_model_name.lower():
                    target_modules = ["c_attn", "c_proj"]
                elif "llama" in self.base_model_name.lower():
                    target_modules = ["q_proj", "v_proj"]
                else:
                    target_modules = ["q_proj", "v_proj"]
            
            # LoRA configuration
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=target_modules,
                bias="none"
            )
            
            # Apply LoRA to model
            self.peft_model = get_peft_model(self.model, lora_config)
            
            # Print trainable parameters
            trainable_params = sum(p.numel() for p in self.peft_model.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in self.peft_model.parameters())
            
            print(f"SUCCESS: LoRA setup complete")
            print(f"Trainable parameters: {trainable_params:,}")
            print(f"Total parameters: {total_params:,}")
            print(f"Trainable %: {100 * trainable_params / total_params:.2f}%")
            
        except Exception as e:
            print(f"ERROR: Failed to setup LoRA: {e}")
            raise
    
    def prepare_dataset(self, 
                       data: List[Dict[str, str]],
                       max_length: int = 512,
                       train_split: float = 0.8) -> Tuple[Dataset, Dataset]:
        """Prepare dataset for training"""
        try:
            # Tokenize data
            def tokenize_function(examples):
                # Combine input and output
                texts = [f"{inp} {out}" for inp, out in zip(examples["input"], examples["output"])]
                
                # Tokenize
                tokenized = self.tokenizer(
                    texts,
                    truncation=True,
                    padding=True,
                    max_length=max_length,
                    return_tensors="pt"
                )
                
                # Set labels (same as input_ids for causal LM)
                tokenized["labels"] = tokenized["input_ids"].clone()
                
                return tokenized
            
            # Create dataset
            dataset = Dataset.from_list(data)
            tokenized_dataset = dataset.map(tokenize_function, batched=True)
            
            # Split dataset
            train_size = int(len(tokenized_dataset) * train_split)
            train_dataset = tokenized_dataset.select(range(train_size))
            eval_dataset = tokenized_dataset.select(range(train_size, len(tokenized_dataset)))
            
            print(f"SUCCESS: Dataset prepared - Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")
            
            return train_dataset, eval_dataset
            
        except Exception as e:
            print(f"ERROR: Failed to prepare dataset: {e}")
            raise
    
    def train(self,
              train_dataset: Dataset,
              eval_dataset: Optional[Dataset] = None,
              num_epochs: int = 3,
              batch_size: int = 4,
              learning_rate: float = 2e-4,
              warmup_steps: int = 100,
              logging_steps: int = 10,
              save_steps: int = 500) -> Dict[str, Any]:
        """Train the model with LoRA"""
        try:
            if self.peft_model is None:
                raise ValueError("LoRA not setup. Call setup_lora() first.")
            
            # Training arguments
            self.training_args = TrainingArguments(
                output_dir=self.output_dir,
                num_train_epochs=num_epochs,
                per_device_train_batch_size=batch_size,
                per_device_eval_batch_size=batch_size,
                warmup_steps=warmup_steps,
                learning_rate=learning_rate,
                logging_steps=logging_steps,
                save_steps=save_steps,
                evaluation_strategy="steps" if eval_dataset else "no",
                eval_steps=save_steps if eval_dataset else None,
                save_total_limit=2,
                load_best_model_at_end=True if eval_dataset else False,
                metric_for_best_model="eval_loss" if eval_dataset else None,
                greater_is_better=False,
                report_to=None,  # Disable wandb/tensorboard
                remove_unused_columns=False,
            )
            
            # Data collator
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer,
                mlm=False,  # Causal LM, not masked LM
            )
            
            # Trainer
            trainer = Trainer(
                model=self.peft_model,
                args=self.training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                data_collator=data_collator,
                tokenizer=self.tokenizer,
            )
            
            # Start training
            print("Starting training...")
            training_result = trainer.train()
            
            # Save model
            trainer.save_model()
            self.tokenizer.save_pretrained(self.output_dir)
            
            # Record training metrics
            training_metrics = {
                "train_loss": training_result.training_loss,
                "train_runtime": training_result.metrics["train_runtime"],
                "train_samples_per_second": training_result.metrics["train_samples_per_second"],
                "epoch": num_epochs,
                "learning_rate": learning_rate,
                "batch_size": batch_size
            }
            
            # Record to monitoring system
            advanced_metrics.record_model_performance(
                model_name=f"fine_tuned_{self.base_model_name.replace('/', '_')}",
                accuracy=1.0 - training_result.training_loss,  # Approximate accuracy
                latency_ms=0,  # Training latency
                confidence=0.8,  # Default confidence
                input_tokens=0,
                output_tokens=0,
                cost_usd=0.0
            )
            
            print("SUCCESS: Training completed")
            return training_metrics
            
        except Exception as e:
            print(f"ERROR: Training failed: {e}")
            raise
    
    def generate_response(self, 
                         prompt: str, 
                         max_length: int = 100,
                         temperature: float = 0.7,
                         do_sample: bool = True) -> str:
        """Generate response using fine-tuned model"""
        try:
            if self.peft_model is None:
                raise ValueError("Model not trained. Train the model first.")
            
            # Tokenize input
            inputs = self.tokenizer(prompt, return_tensors="pt")
            
            # Generate
            with torch.no_grad():
                outputs = self.peft_model.generate(
                    **inputs,
                    max_length=max_length,
                    temperature=temperature,
                    do_sample=do_sample,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Remove input prompt from response
            if prompt in response:
                response = response.replace(prompt, "").strip()
            
            return response
            
        except Exception as e:
            print(f"ERROR: Generation failed: {e}")
            return f"Generation error: {str(e)}"
    
    def load_fine_tuned_model(self, model_path: str) -> None:
        """Load a previously fine-tuned model"""
        try:
            print(f"Loading fine-tuned model from: {model_path}")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            
            # Load base model
            self.model = AutoModelForCausalLM.from_pretrained(
                self.base_model_name,
                torch_dtype=torch.float32
            )
            
            # Load PEFT model
            self.peft_model = PeftModel.from_pretrained(self.model, model_path)
            
            print("SUCCESS: Fine-tuned model loaded")
            
        except Exception as e:
            print(f"ERROR: Failed to load fine-tuned model: {e}")
            raise
    
    def create_sample_dataset(self) -> List[Dict[str, str]]:
        """Create a sample dataset for demonstration"""
        sample_data = [
            {"input": "What is artificial intelligence?", "output": "Artificial intelligence (AI) is a branch of computer science that aims to create machines capable of intelligent behavior."},
            {"input": "How does machine learning work?", "output": "Machine learning is a subset of AI that enables computers to learn and improve from experience without being explicitly programmed."},
            {"input": "What is deep learning?", "output": "Deep learning is a subset of machine learning that uses neural networks with multiple layers to model and understand complex patterns."},
            {"input": "Explain natural language processing.", "output": "Natural language processing (NLP) is a field of AI that focuses on enabling computers to understand, interpret, and generate human language."},
            {"input": "What are neural networks?", "output": "Neural networks are computing systems inspired by biological neural networks that can learn to perform tasks by analyzing training data."},
            {"input": "How does computer vision work?", "output": "Computer vision is a field of AI that enables computers to interpret and understand visual information from the world."},
            {"input": "What is reinforcement learning?", "output": "Reinforcement learning is a type of machine learning where agents learn to make decisions by taking actions in an environment to maximize rewards."},
            {"input": "Explain supervised learning.", "output": "Supervised learning is a type of machine learning where models are trained on labeled data to make predictions on new, unseen data."},
            {"input": "What is unsupervised learning?", "output": "Unsupervised learning is a type of machine learning that finds patterns in data without labeled examples."},
            {"input": "How do transformers work?", "output": "Transformers are neural network architectures that use attention mechanisms to process sequential data efficiently."}
        ]
        
        return sample_data


# Global instance
fine_tuning_manager = FineTuningManager()
