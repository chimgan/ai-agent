import os
import json
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from datasets import load_dataset

# Ensure that PyTorch uses the CPU by setting the environment variable.
# Remove or adjust this line if you have a GPU and want to use it.
os.environ["CUDA_VISIBLE_DEVICES"] = ""
device = torch.device("cpu")

# Load the tokenizer for the 'distilgpt2' model.
tokenizer = AutoTokenizer.from_pretrained('distilgpt2')

# Set the tokenizer's pad_token to be the same as the eos_token.
# This is necessary because GPT-2 models do not have a pad_token by default.
tokenizer.pad_token = tokenizer.eos_token

# Load your conversational dataset from a JSON Lines file.
# The dataset should be in the format: {'prompt': ..., 'response': ...}
dataset = load_dataset('json', data_files={'train': 'conversations.jsonl'})


def tokenize_function(example):
    # Combine the 'prompt' and 'response' fields into a single text string.
    text = example['prompt'] + example['response']
    # Tokenize the combined text with truncation to a maximum length.
    return tokenizer(text, truncation=True, max_length=512)


# Apply the tokenization function to the dataset.
# 'batched=False' indicates that the function works on single examples.
tokenized_dataset = dataset.map(tokenize_function, batched=False)

# Load the pre-trained 'distilgpt2' model for causal language modeling.
model = AutoModelForCausalLM.from_pretrained('distilgpt2')

# Update the model's pad_token_id to match the tokenizer's pad_token_id.
model.config.pad_token_id = tokenizer.pad_token_id

# Initialize the data collator for language modeling.
# This handles dynamic padding of sequences within a batch.
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,  # Set to False because we're using causal language modeling, not masked LM.
)

# Set up training arguments
training_args = TrainingArguments(
    output_dir='data_output/model_output',  # Directory to save model checkpoints and outputs.
    num_train_epochs=3,                     # Total number of training epochs.
    per_device_train_batch_size=1,          # Batch size per device during training.
    save_steps=500,                         # Save a checkpoint every 500 steps.
    save_total_limit=2,                     # Limit the total number of checkpoints to save.
    logging_steps=100,                      # Log training information every 100 steps.
    eval_strategy='no',                     # No evaluation during training.
)

# Initialize the Trainer with the model, training arguments, dataset, and data collator.
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset['train'],
    data_collator=data_collator,
)

# Start the training process.
trainer.train()

# Save the fine-tuned model and tokenizer to the specified directory.
model.save_pretrained('./fine_tuned_model')
tokenizer.save_pretrained('./fine_tuned_model')
