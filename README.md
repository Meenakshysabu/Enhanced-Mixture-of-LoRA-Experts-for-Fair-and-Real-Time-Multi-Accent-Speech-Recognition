
# Enhanced Mixture of LoRA Experts for Fair and Real-Time Multi-Accent Speech Recognition (MAS-LoRA++)

## 📌 Project Overview

MAS-LoRA++ is an accent-aware Automatic Speech Recognition (ASR) system designed to improve speech transcription across different English accents while maintaining computational efficiency.

The project builds on Whisper Small and uses Low-Rank Adaptation (LoRA) for parameter-efficient model adaptation. It incorporates accent-aware learning, phoneme-aware processing, and contrastive learning to address pronunciation variations and improve recognition across diverse English accents.

## 🎯 Objectives

- Improve speech recognition across multiple English accents.
- Reduce accent-related transcription errors.
- Apply LoRA for parameter-efficient fine-tuning.
- Incorporate phoneme-aware processing to handle pronunciation variations.
- Improve inference efficiency and reduce latency.
- Evaluate performance using standard speech recognition metrics.

## ✨ Key Features

- **Whisper-Based ASR:** Uses Whisper Small for speech transcription.
- **LoRA-Based Adaptation:** Enables parameter-efficient model fine-tuning.
- **Accent-Aware Learning:** Addresses differences in English pronunciation.
- **Phoneme-Aware Processing:** Incorporates phonetic information into the recognition process.
- **Contrastive Learning:** Supports learning useful speech representations.
- **Audio Preprocessing:** Prepares audio for model input.
- **Performance Evaluation:** Uses WER, CER, and other relevant metrics.
- **Interactive Interface:** Supports speech recognition evaluation through Gradio, if included in the implementation.

## 🛠️ Technology Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Speech Recognition | Whisper Small |
| Deep Learning | PyTorch |
| Model Adaptation | LoRA |
| Model Framework | Hugging Face Transformers |
| Audio Processing | Librosa, Torchaudio, SoundFile |
| Data Processing | NumPy, Pandas |
| Evaluation | JiWER |
| Interface | Gradio |
| Dataset | Mozilla Common Voice |

## 🧠 Methodology

1. **Dataset Preparation:** Prepare audio recordings and corresponding transcripts from Mozilla Common Voice.
2. **Audio Preprocessing:** Convert audio into the required format, such as 16 kHz mono WAV, where applicable.
3. **Baseline Evaluation:** Evaluate Whisper Small to establish baseline performance.
4. **LoRA Adaptation:** Fine-tune the model using parameter-efficient adaptation.
5. **Accent-Aware Learning:** Adapt the system to pronunciation differences across accents.
6. **Phoneme-Aware Processing:** Incorporate phonetic information to improve accent robustness.
7. **Model Evaluation:** Compare transcription quality and inference efficiency.
8. **Demonstration:** Use the available interface to evaluate speech samples and display transcriptions.

## 📂 Project Structure

```text
MAS-LoRA-Plus-Plus/
│
├── README.md
├── requirements.txt
├── install_mfa.py
│
├── preprocessing/
│   ├── new_tsv.py
│   ├── phoneme.py
│   ├── preprocess.py
│   ├── processed_augmented.py
│   ├── processed_cleaned.py
│   ├── textgrid_to_pt.py
│   └── txt_creation.py
│
├── models/
│   ├── __init__.py
│   ├── dataset.py
│   ├── dataset_whisper.py
│   ├── maslora_model.py
│   └── maslora_whisper.py
│
├── training/
│   └── train_maslora_whisper.py
│
└── evaluation/
    ├── accent_identifier.py
    └── evaluate_wer_comparison.py
```


## 📊 Dataset

This project uses the Mozilla Common Voice speech dataset for training and/or evaluation.

The dataset provides speech recordings and corresponding text transcriptions from speakers with different linguistic backgrounds.

### Data Preprocessing

Depending on the implemented pipeline, preprocessing may include:

- Audio format and sample-rate conversion.
- Transcript preparation and cleaning.
- Voice activity detection (VAD).
- Audio filtering and augmentation.
- Preparation of accent labels and phonetic information.
- Training, validation, and test dataset splitting.

**Note:** The dataset is not included in this repository. Download it from the official Mozilla Common Voice platform and follow its licensing requirements.


## 📈 Evaluation Metrics

The system can be evaluated using the following metrics, where implemented:

- **Word Error Rate (WER):** Measures word-level transcription errors.
- **Character Error Rate (CER):** Measures character-level transcription errors.
- **Accent-Specific WER:** Measures recognition performance for individual accent groups.
- **Inference Latency:** Measures the time taken to transcribe audio.
- **Real-Time Factor (RTF):** Compares processing time with audio duration.

Lower WER and CER generally indicate better transcription accuracy. Lower latency and RTF indicate faster processing.




