import torchaudio
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os

class AccentDataset(Dataset):
    def __init__(self, folder):
        self.files = [f for f in os.listdir(folder) if f.endswith(".wav")]
        self.folder = folder
        self.labels = list(set([f.split("_")[0] for f in self.files]))
        self.label_map = {label: i for i, label in enumerate(self.labels)}

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file = self.files[idx]
        waveform, sr = torchaudio.load(os.path.join(self.folder, file))
        label = self.label_map[file.split("_")[0]]
        return waveform[:, :16000], label  # 1 sec clip

class AccentCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 16, 5),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, 5),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        self.fc = nn.Sequential(
            nn.Linear(32 * 61, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

dataset = AccentDataset("C:/Main/data/common_voice/processed")
loader = DataLoader(dataset, batch_size=8, shuffle=True)
model = AccentCNN(num_classes=len(dataset.labels))
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()

for epoch in range(5):
    for x, y in loader:
        x = x[:, 0, :]  # mono
        out = model(x)
        loss = loss_fn(out, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1} Loss: {loss.item():.4f}")
