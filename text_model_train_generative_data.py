import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast, BertForSequenceClassification, Trainer, TrainingArguments

# from moviepy.editor import VideoFileClip

# # Extract audio from video
# def extract_audio_from_video(video_path, audio_path):
#     video = VideoFileClip(video_path)
#     audio = video.audio
#     audio.write_audiofile(audio_path)

# video_path = "vid2.mov"
# audio_path = "audio2.wav"
# extract_audio_from_video(video_path, audio_path)

# Convert txt files to JSON
def convert_txt_to_json(folder):
    data = []
    for file in os.listdir(folder):
        if file.endswith(".txt"):
            with open(os.path.join(folder, file), "r") as f:
                text = f.read()
                label = 0 if "no_depression" in folder else 1
                data.append({"text": text, "label": label})
    return data

depression_folder = "dataset/generated_conversations_depression_PHQ-8"
no_depression_folder = "dataset/generated_conversations_no_depression_PHQ-8"

depression_data = convert_txt_to_json(depression_folder)
no_depression_data = convert_txt_to_json(no_depression_folder)
all_data = depression_data + no_depression_data

# Save the combined data as a JSON file
with open("dataset/all_data.json", "w") as f:
    json.dump(all_data, f)

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

# Split the data into train, validation, and test sets
from sklearn.model_selection import train_test_split
train_data, test_data = train_test_split(all_data, test_size=0.2, random_state=42)
train_data, val_data = train_test_split(train_data, test_size=0.25, random_state=42)

# Save the data splits as JSON files
with open("dataset/train_data_generative_data.json", "w") as f:
    json.dump(train_data, f)
with open("dataset/val_data_generative_data.json", "w") as f:
    json.dump(val_data, f)
with open("dataset/test_data_generative_data.json", "w") as f:
    json.dump(test_data, f)

# Create datasets
train_dataset = DepressionTextDataset("dataset/train_data_generative_data.json", tokenizer)
val_dataset = DepressionTextDataset("dataset/val_data_generative_data.json", tokenizer)
test_dataset = DepressionTextDataset("dataset/test_data_generative_data.json", tokenizer)

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=5,
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
torch.save(model.state_dict(), 'pre-trained_model/text_model_generative_data.pt')

# Save the model and tokenizer
model.save_pretrained("pre-trained_model/text_model_generative_data")
tokenizer.save_pretrained("pre-trained_model/text_model_generative_data")