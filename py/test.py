from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments, pipeline
from torch.utils.data import Dataset
import torch

# ======= Step 1: Load Resume =======
with open("resume.txt", "r", encoding="utf-8") as file:
    resume_text = file.read()

# ======= Step 2: Tokenize the Resume =======
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token  # GPT-2 requires an EOS token for padding

train_encodings = tokenizer(
    [resume_text],  # List of training text samples
    truncation=True,
    padding=False,
    max_length=512,
    return_tensors="pt"
)

# ======= Step 3: Create a Dataset Class with Labels =======
class TextDataset(Dataset):
    def __init__(self, encodings):
        self.input_ids = encodings["input_ids"]
        self.attention_mask = encodings["attention_mask"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels": self.input_ids[idx]  # Set labels equal to input_ids for auto-regressive models
        }

train_dataset = TextDataset(train_encodings)

# ======= Step 4: Fine-Tune the GPT-2 Model =======
model = GPT2LMHeadModel.from_pretrained("gpt2")

training_args = TrainingArguments(
    output_dir="./gpt2-finetuned-resume",  # Directory to save the model
    num_train_epochs=10,
    per_device_train_batch_size=1,
    save_steps=500,  # Save model every 500 steps
    save_total_limit=2,  # Keep only the last 2 saved models
    logging_dir="./logs",
    logging_steps=10,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset
)

print("Training started...")
trainer.train()

# Save the model explicitly after training
model.save_pretrained("./gpt2-finetuned-resume")
tokenizer.save_pretrained("./gpt2-finetuned-resume")
print("Model saved!")

# ======= Step 5: Use the Fine-tuned Model =======
generator = pipeline("text-generation", model="./gpt2-finetuned-resume", tokenizer="./gpt2-finetuned-resume")

prompt = "Based on my resume, how many skill I have? please give me the name of those skills."
result = generator(prompt, max_length=100, num_return_sequences=1)
print("Generated response:", result)
