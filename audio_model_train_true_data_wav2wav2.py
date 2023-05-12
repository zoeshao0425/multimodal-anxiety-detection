import os
import csv
import torch
import torchaudio
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, accuracy_score
from transformers import Wav2Vec2Model, Wav2Vec2Processor
from torch.nn import functional as F 
from tqdm import tqdm
import librosa
from torch.cuda.amp import GradScaler, autocast

# Load labels
def load_labels(csv_files):
    labels = {}
    for file in csv_files:
        df = pd.read_csv(file)
        for idx, row in df.iterrows():
            patient_id, label = row[0], row[1]
            labels[str(int(patient_id))] = label
    return labels

class DepressionDataset(Dataset):
    def __init__(self, filepaths, labels, processor, wav2vec2_model, max_length, window_size, overlap, max_audio_pieces):
        self.filepaths = filepaths
        self.labels = labels
        self.processor = processor
        self.wav2vec2_model = wav2vec2_model
        self.max_length = max_length
        self.window_size = window_size
        self.overlap = overlap
        self.max_audio_pieces = max_audio_pieces

    def __getitem__(self, idx):
        filepath = self.filepaths[idx]
        
        # Extract the key from the filepath
        key = os.path.splitext(os.path.basename(filepath))[0][:-6]
        
        label = self.labels[key]

        # Load audio
        audio, _ = librosa.load(filepath, sr=16000)

        # Split audio into pieces
        fixed_length = 16000  # 1 second
        step_size = 8000  # 0.5 seconds
        audio_pieces = [audio[i:i+fixed_length] for i in range(0, len(audio)-fixed_length, step_size)]

        # Process each audio piece and store the features in a list
        features_list = []
        for piece in audio_pieces:
            # Preprocess audio
            inputs = self.processor(piece, sampling_rate=16000, return_tensors="pt", padding=True)

            # Apply wav2vec2 feature extraction model
            inputs = self.wav2vec2_model(**inputs)

            # Create overlapping windows
            input_values = inputs['last_hidden_state'][0]
            windows = input_values.unfold(0, self.window_size, self.window_size - self.overlap)

            # Extract features for each window
            features = torch.mean(windows, dim=1)

            features_list.append(features)

        # Randomly select a fixed number of audio pieces
        if len(features_list) > self.max_audio_pieces:
            features_list = random.sample(features_list, self.max_audio_pieces)

        return features_list, torch.tensor(label, dtype=torch.float16)

    def __len__(self):
        return len(self.filepaths)

class DepressionClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(DepressionClassifier, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out, _ = self.gru(x)
        out = self.fc(out[:, -1, :])
        return self.sigmoid(out)



def evaluate(model, dataloader, device):
    model.eval()
    predictions, ground_truth = [], []
    with torch.no_grad():
        for inputs_list, targets in tqdm(dataloader, desc="Evaluating"):
            targets = targets.to(device)
            batch_outputs = []

            for inputs in inputs_list:
                inputs = inputs.to(device)
                outputs = model(inputs)
                batch_outputs.append(outputs.squeeze())

            # Take the average output over all pieces of the audio
            avg_output = torch.mean(torch.stack(batch_outputs), dim=0)
            avg_output = (avg_output > 0.5).float()
            predictions.extend(avg_output.tolist())
            ground_truth.extend(targets.tolist())

    acc = accuracy_score(ground_truth, predictions)
    f1 = f1_score(ground_truth, predictions)

    return acc, f1

# Load labels and create a dictionary
csv_files = [
    "train_split_Depression_AVEC2017.csv",
    "dev_split_Depression_AVEC2017.csv",
    "full_test_split.csv"
]
labels = load_labels(csv_files)

from sklearn.model_selection import train_test_split
import random

subset_percentage = 0.1  # Use 10% of the data

def load_audio(file_path):
    waveform, sample_rate = torchaudio.load(file_path)
    return waveform, sample_rate

# Preprocess audio files
audio_folder = "audio_files"
audio_files = [os.path.join(audio_folder, f) for f in os.listdir(audio_folder)]

# Randomly select a subset of audio files
subset_size = int(len(audio_files) * subset_percentage)
subset_audio_files = random.sample(audio_files, subset_size)

# Split audio files into train, val, and test sets
train_files, test_files = train_test_split(subset_audio_files, test_size=0.2, random_state=42)
train_files, val_files = train_test_split(train_files, test_size=0.25, random_state=42)

# Initialize Wav2Vec2 processor and model
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-100h", sampling_rate=16000)
wav2vec2_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-100h")

max_length = 0
# Set max_length to the longest audio file
for waveform, sample_rate in map(load_audio, audio_files):
    processed_audio = processor(waveform, return_tensors="pt", sampling_rate=sample_rate).input_values.squeeze(0)
    max_length = max(max_length, processed_audio.shape[1])

print(f"Max length: {max_length}")

# Set the batch size
hidden_size = 128
num_layers = 2
output_size = 1
num_epochs = 10
batch_size = 8
gradient_accumulation_steps = 8
learning_rate = 0.001
scaler = GradScaler()

# Initialize model, criterion, optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DepressionClassifier(input_size=768, hidden_size=hidden_size, num_layers=num_layers, output_size=output_size).to(device)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# Set the sliding window size and overlap
window_size = 20  # Arbitrary value smaller than the sequence_length
overlap = window_size // 2
max_audio_pieces = 50

# Create datasets
train_dataset = DepressionDataset(train_files, labels, processor, wav2vec2_model, max_length, window_size, overlap, max_audio_pieces)
val_dataset = DepressionDataset(val_files, labels, processor, wav2vec2_model, max_length, window_size, overlap, max_audio_pieces)
test_dataset = DepressionDataset(test_files, labels, processor, wav2vec2_model, max_length, window_size, overlap, max_audio_pieces)

# Create data loaders
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

def train(model, dataloader, criterion, optimizer, device, gradient_accumulation_steps, scaler):
    model.train()
    running_loss = 0.0
    optimizer.zero_grad()
    for step, (inputs, targets) in enumerate(tqdm(dataloader, desc="Training")):
        inputs, targets = inputs.to(device), targets.to(device)

        with autocast():
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), targets) / gradient_accumulation_steps

        scaler.scale(loss).backward()

        if (step + 1) % gradient_accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        running_loss += loss.item() * gradient_accumulation_steps
        torch.cuda.empty_cache()

    return running_loss / len(dataloader)

for epoch in range(num_epochs):
    train_loss = train(model, train_dataloader, criterion, optimizer, device, gradient_accumulation_steps, scaler)
    train_acc, train_f1 = evaluate(model, train_dataloader, device)
    val_acc, val_f1 = evaluate(model, val_dataloader, device)
    print(f"Epoch {epoch+1}/{num_epochs}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Train F1: {train_f1:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}")

max_length

