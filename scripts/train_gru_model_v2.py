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

def train_model_a(model, loader, criterion, optimizer):
    model.train()
    for commands_batch, files_batch in loader:
        optimizer.zero_grad()
        outputs = model(commands_batch)
        loss = criterion(outputs, files_batch[:,0])
        loss.backward()
        optimizer.step()

def train_model_b(model, loader, criterion, optimizer):
    model.train()
    for commands_batch, files_batch in loader:
        optimizer.zero_grad()
        history = files_batch[:,:-1]
        targets = files_batch[:,1]
        outputs = model(commands_batch, history)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

def evaluate_model_a(model, loader):
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for commands_batch, files_batch in loader:
            outputs = model(commands_batch)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(files_batch[:,0].cpu().numpy())
    accuracy = accuracy_score(all_targets, all_preds)
    return accuracy

def evaluate_model_b(model, loader, top_k=5):
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for commands_batch, files_batch in loader:
            history = files_batch[:,:-1]
            targets = files_batch[:,1]
            outputs = model(commands_batch, history)
            _, preds = torch.topk(outputs, top_k, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    top_k_acc = top_k_accuracy_score(all_targets, np.array(all_preds), k=top_k)
    return top_k_acc

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

    # データセットの準備
    dataset = AccessDataset(commands, accessed_files, file_to_id)
    
    # KFoldの設定
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
        print(f'Fold {fold+1}')
        train_subset = torch.utils.data.Subset(dataset, train_idx)
        val_subset = torch.utils.data.Subset(dataset, val_idx)
        train_loader = DataLoader(train_subset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=32, shuffle=False)

        # モデルの初期化
        model_a = ModelA(vocab_size=10000, embed_size=128, hidden_size=256, output_size=len(file_to_id))
        model_b = ModelB(vocab_size=10000, embed_size=128, hidden_size=256, output_size=len(file_to_id))
        criterion = nn.CrossEntropyLoss()
        optimizer_a = torch.optim.Adam(model_a.parameters(), lr=0.001)
        optimizer_b = torch.optim.Adam(model_b.parameters(), lr=0.001)

        # モデルAのトレーニング
        for epoch in range(10):
            train_model_a(model_a, train_loader, criterion, optimizer_a)

        # モデルBのトレーニング
        for epoch in range(10):
            train_model_b(model_b, train_loader, criterion, optimizer_b)

        # モデルAの評価
        accuracy_a = evaluate_model_a(model_a, val_loader)
        print(f'Fold {fold+1} ModelA Accuracy: {accuracy_a}')

        # モデルBの評価
        top_k_acc_b = evaluate_model_b(model_b, val_loader, top_k=5)
        print(f'Fold {fold+1} ModelB Top-5 Accuracy: {top_k_acc_b}')

if __name__ == "__main__":
    main()
