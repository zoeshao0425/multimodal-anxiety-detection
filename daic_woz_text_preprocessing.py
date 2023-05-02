import pandas as pd
# Load labels
def load_labels(csv_files):
    labels = {}
    for file in csv_files:
        df = pd.read_csv(file)
        for idx, row in df.iterrows():
            patient_id, label = row[0], row[1]
            labels[str(int(patient_id))] = label
    return labels

# Load labels and create a dictionary
csv_files = [
    "dataset/DAIC-WOZ/train_split_Depression_AVEC2017.csv",
    "dataset/DAIC-WOZ/dev_split_Depression_AVEC2017.csv",
    "dataset/DAIC-WOZ/full_test_split.csv"
]
labels = load_labels(csv_files)

import csv
import json
import os

input_directory = "dataset/DAIC-WOZ"
output_file = "daic_woz_preprocessed.json"

def preprocess_transcript(file_path, patient_id):
    conversation = []
    with open(file_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile, delimiter='\t')
        for row in reader:
            speaker = "Doctor" if row['speaker'] == "Ellie" else "Patient"
            conversation.append(f"{speaker}: {row['value']}")
    text = "\n\n".join(conversation)
    patient_id = patient_id[:-2]
    label = labels.get(patient_id, -1)  # -1 indicates the label was not found
    print(label)
    return text, label

preprocessed_data = []
for patient_folder in os.listdir(input_directory):
    folder_path = os.path.join(input_directory, patient_folder)
    if os.path.isdir(folder_path):
        transcript_file = os.path.join(folder_path, f"{patient_folder[:-2]}_TRANSCRIPT.csv")
        if os.path.isfile(transcript_file):
            text, label = preprocess_transcript(transcript_file, patient_folder)
            if label != -1:
                preprocessed_data.append({"text": text, "label": int(label)})
            print(preprocessed_data)

with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(preprocessed_data, f, ensure_ascii=False, indent=4)



