import os
import shutil
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


def extract_audio_files(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for participant_id in range(300, 500):
        participant_id = str(participant_id)
        participant_folder = os.path.join(input_dir, participant_id + "_P")
        print(participant_folder)
        
        if os.path.isdir(participant_folder):
            audio_file = os.path.join(participant_folder, f"{participant_id}_AUDIO.wav")
            if os.path.exists(audio_file):
                shutil.copy(audio_file, os.path.join(output_dir, f"{participant_id}_AUDIO.wav"))

input_dir = "dataset/DAIC-WOZ"  # Set the path to the parent folder containing participant folders
output_dir = "dataset/DAIC-WOZ/audio_files"  # Set the path to store the extracted audio files

extract_audio_files(input_dir, output_dir)

# Load labels
def load_labels(csv_files):
    labels = {}
    for file in csv_files:
        df = pd.read_csv(file)
        for idx, row in df.iterrows():
            patient_id, label = row[0], row[1]
            labels[str(int(patient_id))] = label
    return labels

def compute_mel_spectrogram(waveform, sample_rate, n_mels=128):
    mel_spectrogram = T.MelSpectrogram(sample_rate, n_fft=2048, hop_length=512, n_mels=n_mels)(waveform)
    return torch.log(mel_spectrogram + 1e-9)

class DepressionDataset(Dataset):
    def __init__(self, file_list, labels, max_length, n_mels=128):
        self.file_list = file_list
        self.labels = labels
        self.max_length = max_length
        self.n_mels = n_mels
        self.resampler = T.Resample(orig_freq=16000, new_freq=8000)  # Resample to 8kHz

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path = self.file_list[idx]
        patient_id = os.path.splitext(os.path.basename(file_path))[0][:-6]
        label = self.labels[patient_id]
        waveform, sample_rate = torchaudio.load(file_path)

        # Resample the audio waveform
        waveform = self.resampler(waveform)

        # Compute Mel Spectrogram
        mel_spectrogram = compute_mel_spectrogram(waveform, sample_rate, self.n_mels)

        # Trim or pad Mel Spectrogram
        if mel_spectrogram.shape[2] > self.max_length:
            mel_spectrogram = mel_spectrogram[:, :, :self.max_length]
        elif mel_spectrogram.shape[2] < self.max_length:
            padding = self.max_length - mel_spectrogram.shape[2]
            mel_spectrogram = torch.nn.functional.pad(mel_spectrogram, (0, padding))

        mel_spectrogram = mel_spectrogram.squeeze(0)

        return mel_spectrogram, torch.tensor(label, dtype=torch.float32)

class DepressionClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(DepressionClassifier, self).__init__()
        self.conv1 = nn.Conv1d(input_size, 64, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.gru = nn.GRU(128, hidden_size, num_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        
        x = x.transpose(1, 2)
        
        out, _ = self.gru(x)
        
        out = self.fc1(out[:, -1, :])
        out = self.relu3(out)
        out = self.fc2(out)
        
        return self.sigmoid(out)

from tqdm import tqdm

# Training function
def train(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), leave=False)
    for i, (inputs, labels) in progress_bar:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs).squeeze(1)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        progress_bar.set_description(f"Loss: {loss.item():.4f}")
    return running_loss / len(dataloader)

def evaluate(model, dataloader, device):
    model.eval()
    predictions = []
    ground_truth = []
    with torch.no_grad():
        for inputs, labels in dataloader:
            print("label: ", labels)
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs).squeeze()
            print("outputs: ", outputs)
            preds = (outputs > 0.5).float()

            # Updated lines
            predictions.extend(preds.cpu().unsqueeze(-1).numpy().flatten())
            ground_truth.extend(labels.cpu().unsqueeze(-1).numpy().flatten())

    return accuracy_score(ground_truth, predictions), f1_score(ground_truth, predictions)

# Parameters
n_mels = 128
input_size = n_mels
hidden_size = 128
num_layers = 2
output_size = 1
num_epochs = 10
batch_size = 8
learning_rate = 0.01

# Load labels and create a dictionary
csv_files = [
    "dataset/DAIC-WOZ/train_split_Depression_AVEC2017.csv",
    "dataset/DAIC-WOZ/dev_split_Depression_AVEC2017.csv",
    "dataset/DAIC-WOZ/ull_test_split.csv"
]
labels = load_labels(csv_files)

# Preprocess audio files
audio_folder = "audio_files"
audio_files = [os.path.join(audio_folder, f) for f in os.listdir(audio_folder)]
train_files, test_files = train_test_split(audio_files, test_size=0.2, random_state=42)
train_files, val_files = train_test_split(train_files, test_size=0.25, random_state=42)

# Determine the maximum audio length (in terms of Mel Spectrogram frames)
def get_max_length(file_list, resampler, n_mels=128):
    max_length = 0
    new_sample_rate = 8000
    for file_path in file_list:
        waveform, _ = torchaudio.load(file_path)
        waveform = resampler(waveform)
        mel_spectrogram = compute_mel_spectrogram(waveform, new_sample_rate, n_mels)
        max_length = max(max_length, mel_spectrogram.shape[2])
    return max_length


# Determine the maximum audio length
resampler = T.Resample(orig_freq=16000, new_freq=8000)
max_length = get_max_length(audio_files, resampler, n_mels)

# Create datasets and dataloaders
train_dataset = DepressionDataset(train_files, labels, max_length)
val_dataset = DepressionDataset(val_files, labels, max_length)
test_dataset = DepressionDataset(test_files, labels, max_length)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Initialize model, criterion, optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DepressionClassifier(input_size, hidden_size, num_layers, output_size).to(device)
criterion = nn.BCELoss()


# Change the optimizer
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)  # Use SGD instead of Adam

# Train and evaluate the model
for epoch in range(num_epochs):
    train_loss = train(model, train_dataloader, criterion, optimizer, device)
    train_acc, train_f1 = evaluate(model, train_dataloader, device)
    val_acc, val_f1 = evaluate(model, val_dataloader, device)

    print(f"Epoch [{epoch+1}/{num_epochs}]")
    print(f"Train Loss: {train_loss:.4f} Train Acc: {train_acc:.4f} Train F1: {train_f1:.4f}")
    print(f"Val Acc: {val_acc:.4f} Val F1: {val_f1:.4f}")

# Test the model
test_acc, test_f1 = evaluate(model, test_dataloader, device)
print(f"Test Acc: {test_acc:.4f} Test F1: {test_f1:.4f}")

# Save the model
torch.save(model.state_dict(), 'audio_model.pt')

train_acc, train_f1 = evaluate(model, train_dataloader, device)

def visualize_mel_spectrogram(mel_spectrogram):
    plt.figure(figsize=(10, 4))
    plt.imshow(mel_spectrogram, aspect='auto', origin='lower')
    plt.colorbar()
    plt.title('Mel Spectrogram')
    plt.show()

# Load an example waveform and compute the Mel spectrogram
example_file = audio_files[1]
waveform, sample_rate = torchaudio.load(example_file)
waveform = resampler(waveform)
mel_spectrogram = compute_mel_spectrogram(waveform, sample_rate, n_mels)
mel_spectrogram = mel_spectrogram.squeeze(0)

visualize_mel_spectrogram(mel_spectrogram.numpy())

