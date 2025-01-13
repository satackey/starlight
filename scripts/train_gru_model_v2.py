import argparse
import pandas as pd
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score, top_k_accuracy_score

# データセットの定義
class AccessDataset(Dataset):
    def __init__(self, commands, accessed_files, file_to_id):
        self.commands = commands
        self.accessed_files = accessed_files
        self.file_to_id = file_to_id

    def __len__(self):
        return len(self.commands)

    def __getitem__(self, idx):
        command = self.commands[idx]
        files = self.accessed_files[idx]
        file_ids = [self.file_to_id[file['path']] for file in files]
        return command, file_ids

# モデルAの定義
class ModelA(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, output_size):
        super(ModelA, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        embedded = self.embedding(x)
        _, hidden = self.rnn(embedded)
        out = self.fc(hidden.squeeze(0))
        return out

# モデルBの定義
class ModelB(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, output_size):
        super(ModelB, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.gru = nn.GRU(embed_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, history):
        embedded = self.embedding(x)
        combined = torch.cat((embedded, history), dim=1)
        out, hidden = self.gru(combined.unsqueeze(1))
        out = self.fc(hidden.squeeze(0))
        return out

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('csv_file', type=str, help='Path to the CSV file')
    args = parser.parse_args()

    # データの読み込み
    data = pd.read_csv(args.csv_file)
    data['accessed_files'] = data['accessed_files'].apply(json.loads)
    
    # ファイルパスの辞書作成
    all_files = set()
    for files in data['accessed_files']:
        for file in files:
            all_files.add(file['path'])
    file_to_id = {file: idx for idx, file in enumerate(sorted(all_files))}
    id_to_file = {idx: file for file, idx in file_to_id.items()}

    commands = data['command'].tolist()
    accessed_files = data['accessed_files'].tolist()

    # データセットとデータローダーの準備
    dataset = AccessDataset(commands, accessed_files, file_to_id)
    train_size, val_size = train_test_split(len(dataset), test_size=0.2)
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # モデルの初期化
    model_a = ModelA(vocab_size=10000, embed_size=128, hidden_size=256, output_size=len(file_to_id))
    model_b = ModelB(vocab_size=10000, embed_size=128, hidden_size=256, output_size=len(file_to_id))
    criterion = nn.CrossEntropyLoss()
    optimizer_a = torch.optim.Adam(model_a.parameters(), lr=0.001)
    optimizer_b = torch.optim.Adam(model_b.parameters(), lr=0.001)

    # モデルAのトレーニング
    for epoch in range(10):
        model_a.train()
        for commands_batch, files_batch in train_loader:
            optimizer_a.zero_grad()
            outputs = model_a(commands_batch)
            loss = criterion(outputs, files_batch[:,0])
            loss.backward()
            optimizer_a.step()

    # モデルBのトレーニング
    for epoch in range(10):
        model_b.train()
        for commands_batch, files_batch in train_loader:
            optimizer_b.zero_grad()
            history = files_batch[:,:-1]
            targets = files_batch[:,1]
            outputs = model_b(commands_batch, history)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer_b.step()

if __name__ == "__main__":
    main()
