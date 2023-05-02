import torch
from transformers import BertTokenizerFast, BertForSequenceClassification
from transformers import TrainingArguments, Trainer
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import json

# Custom Dataset class
class DepressionTextDataset(Dataset):
    def __init__(self, data_file, tokenizer, max_length=512):
        self.data = json.load(open(data_file))
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = item["text"]
        label = item["label"]

        encoding = self.tokenizer(text, return_tensors='pt', max_length=self.max_length, padding="max_length", truncation=True)
        encoding = {key: value.squeeze(0) for key, value in encoding.items()}  # Remove batch dimension added by return_tensors='pt'
        encoding["labels"] = torch.tensor(label, dtype=torch.long)

        return encoding

# Initialize tokenizer and model
model_name = "bert-base-uncased"
tokenizer = BertTokenizerFast.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)

# Load the preprocessed data
with open("dataset/daic_woz_preprocessed.json", "r") as f:
    data = json.load(f)

# Split the data into train, validation, and test sets
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
train_data, val_data = train_test_split(train_data, test_size=0.25, random_state=42)

# Save the data splits as JSON files
with open("dataset/train_data_true_data.json", "w") as f:
    json.dump(train_data, f)
with open("dataset/val_data_true_data.json", "w") as f:
    json.dump(val_data, f)
with open("dataset/test_data_true_data.json", "w") as f:
    json.dump(test_data, f)

# Create datasets
train_dataset = DepressionTextDataset("dataset/train_data_true_data.json", tokenizer)
val_dataset = DepressionTextDataset("dataset/val_data_true_data.json", tokenizer)
test_dataset = DepressionTextDataset("dataset/test_data_true_data.json", tokenizer)

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=10,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    evaluation_strategy="epoch",
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

# Fine-tune the model
trainer.train()

# Evaluate the model on the test dataset
results = trainer.predict(test_dataset)
print(results.metrics)

# Save the model
torch.save(model.state_dict(), 'pre-trained_model/text_model_true_model.pt')

