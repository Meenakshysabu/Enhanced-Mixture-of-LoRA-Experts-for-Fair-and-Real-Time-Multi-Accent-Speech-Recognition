
# MAS-LoRA++

## Enhanced Mixture of LoRA Experts for Fair and Real-Time Multi-Accent Speech Recognition

MAS-LoRA++ is an AI-based speech recognition project designed to improve transcription accuracy across different English accents using Whisper Small, Low-Rank Adaptation (LoRA), accent identification, phoneme-aware learning, and contrastive learning.

## Features

- Whisper Small-based speech recognition
- Accent identification
- LoRA-based model adaptation
- Phoneme-aware training
- Contrastive learning
- Word Error Rate (WER) and Character Error Rate (CER) evaluation
- Baseline Whisper and MAS-LoRA++ comparison
- Gradio-based evaluation interface (as described in the project presentation)

## Technologies Used

- Python
- PyTorch
- Hugging Face Transformers
- Whisper Small
- LoRA
- Mozilla Common Voice
- Librosa
- Torchaudio
- JiWER
- Gradio

## Project Structure

```text
MAS-LoRA-Plus-Plus/
├── README.md
├── requirements.txt
├── .gitignore
├── configs/
├── models/
├── preprocessing/
├── training/
├── evaluation/
├── demo/
├── docs/
└── results/
```

## Dataset

This project uses Mozilla Common Voice speech data for multi-accent speech recognition. Download and prepare the dataset separately according to its applicable terms.


## Usage

Prepare the dataset and update the paths in the project configuration before training or evaluation.

Training and evaluation commands depend on the scripts and configuration included in the repository.

## Evaluation Metrics

- **WER:** Word Error Rate
- **CER:** Character Error Rate
- **AS-WER:** Accent-Specific Word Error Rate
- **Inference Time:** Time required to generate a transcription
- **RTF:** Real-Time Factor




*Academic research project focused on accent-aware automatic speech recognition.*
