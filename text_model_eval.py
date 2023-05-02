import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast, BertConfig, BertForSequenceClassification
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

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

def evaluate_model(model_path, test_data_file):
    # Load tokenizer
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

    # Load model configuration
    config = BertConfig.from_pretrained("bert-base-uncased", num_labels=2)

    # Load model state dictionary
    model = BertForSequenceClassification(config)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    with open(test_data_file, "r") as f:
        test_data = json.load(f)

    # Create test dataset
    test_dataset = DepressionTextDataset(test_data_file, tokenizer)

    # Initialize dataloader
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=8)

    # Variables for metrics
    total_samples = 0
    correct_predictions = 0
    all_predictions = []
    all_labels = []

    # Evaluate
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            labels = batch["labels"]

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            predictions = torch.argmax(outputs.logits, dim=1)

            correct_predictions += (predictions == labels).sum().item()
            total_samples += len(labels)

            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = correct_predictions / total_samples
    f1 = f1_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions)
    recall = recall_score(all_labels, all_predictions)

    print(f"Accuracy: {accuracy}")
    print(f"F1 Score: {f1}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")

# Usage
model_path = "pre-trained_model/text_model_generative_data.pt"
test_data_file = "dataset/test_data.json"
accuracy = evaluate_model(model_path, test_data_file)

