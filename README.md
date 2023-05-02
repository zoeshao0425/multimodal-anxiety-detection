# multimodal-anxiety-detection

# Depression Detection from Speech and Text

This repository contains the code for training and evaluating models to detect depression from speech and text data. The data comes from the DAIC-WOZ dataset and is preprocessed before being fed into the models.

## Directory Structure

- `audio_model_train_true_data_Mel.py`: Script for training an audio-based model using Mel spectrogram features.
- `daic_woz_text_preprocessing.py`: Script for preprocessing the DAIC-WOZ text dataset.
- `generated_conversations.py`: Script for generating synthetic conversation data.
- `text_model_eval.py`: Script for evaluating the text-based model.
- `text_model_train_generative_data.py`: Script for training the text-based model on generated data.
- `text_model_training_true_data.py`: Script for training the text-based model on true data.

## Data Preprocessing

The DAIC-WOZ dataset contains audio and text data collected during Wizard of Oz interviews. The data is preprocessed using the following steps:

1. Text data is tokenized and cleaned using the `daic_woz_text_preprocessing.py` script.
2. Audio data is preprocessed using Mel spectrogram features (in `audio_model_train_true_data_Mel.py`) or Wav2Vec2 features (currently under development).

## Model Training

There are two types of models in this repository: text-based and audio-based models.

### Text-based Models

Text-based models are trained using the following scripts:

- `text_model_training_true_data.py`: Trains a text-based model on true data from the DAIC-WOZ dataset.
- `text_model_train_generative_data.py`: Trains a text-based model on generated data, which is created using the `generated_conversations.py` script.

### Audio-based Models

Audio-based models are trained using the following scripts:

- `audio_model_train_true_data_Mel.py`: Trains an audio-based model using Mel spectrogram features.

## Model Evaluation

The `text_model_eval.py` script is used to evaluate the performance of the text-based models. The evaluation script for the audio-based models is currently under development.
