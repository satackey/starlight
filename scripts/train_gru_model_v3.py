import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import re
import torch
import torch.nn as nn
import pandas as pd
import sys
import os
import json

class ModelA(nn.Module):
    """Model A: コマンド → 最初のファイルを予測"""
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_files):
        super(ModelA, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_files)

    def forward(self, cmd_input):
        embedded = self.embedding(cmd_input)
        _, hidden = self.rnn(embedded)
        out = self.fc(hidden[-1])
        return out


class ModelB(nn.Module):
    """Model B: (コマンド, これまでのファイルID列) → 次のファイルを予測"""
    def __init__(self, vocab_size, file_size, embed_dim, hidden_dim):
        super(ModelB, self).__init__()
        self.cmd_embedding = nn.Embedding(vocab_size, embed_dim)
        self.file_embedding = nn.Embedding(file_size, embed_dim)
        self.rnn = nn.GRU(embed_dim*2, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, file_size)

    def forward(self, cmd_input, file_history):
        cmd_emb = self.cmd_embedding(cmd_input)[:, -1, :].unsqueeze(1)
        file_emb = self.file_embedding(file_history)
        combined = torch.cat([cmd_emb.repeat(1, file_emb.size(1), 1), file_emb], dim=2)
        _, hidden = self.rnn(combined)
        out = self.fc(hidden[-1])
        return out

def load_data(csv_path):
    """CSVを読み込み、コマンドとファイルシーケンスのIDを返す"""
    df = pd.read_csv(csv_path)
    commands = df['command'].tolist()
    accessed_files = df['accessed_files'].tolist()
    # ...コマンドのトークナイズやファイルIDへの変換など...
    return commands, accessed_files


def train_modelA(modelA, train_loader, criterion, optimizer, epoch):
    """Model Aの学習ループ"""
    modelA.train()
    total_loss = 0
    for batch in train_loader:
        cmd_input, target_file = batch
        optimizer.zero_grad()
        output = modelA(cmd_input)
        loss = criterion(output, target_file)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)
    print(f"ModelA Epoch {epoch} Loss: {avg_loss:.4f}")

def train_modelB(modelB, train_loader, criterion, optimizer, epoch):
    """Model Bの学習ループ"""
    modelB.train()
    total_loss = 0
    for batch in train_loader:
        cmd_input, file_history, next_file = batch
        optimizer.zero_grad()
        output = modelB(cmd_input, file_history)
        loss = criterion(output, next_file)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)
    print(f"ModelB Epoch {epoch} Loss: {avg_loss:.4f}")

def evaluate_modelA(modelA, eval_loader):
    """Model Aの評価: 最初のファイル予測精度(Accuracy)など"""
    modelA.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in eval_loader:
            cmd_input, target_file = batch
            output = modelA(cmd_input)
            preds = output.argmax(dim=1)
            correct += (preds == target_file).sum().item()
            total += target_file.size(0)
    return correct / total if total > 0 else 0

def evaluate_modelB(modelB, eval_loader, k=5):
    """Model Bの評価: 次のファイルTop-k Accuracyなど"""
    modelB.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in eval_loader:
            cmd_input, file_history, next_file = batch
            output = modelB(cmd_input, file_history)
            _, topk_indices = output.topk(k, dim=1)
            for i in range(next_file.size(0)):
                if next_file[i].item() in topk_indices[i]:
                    correct += 1
            total += next_file.size(0)
    return correct / total if total > 0 else 0

def combined_inference(modelA, modelB, cmd_input, max_steps=10):
    """モデルA&Bを組み合わせて最初のファイル→続くファイルを順次推測"""
    # 最初のファイルを予測
    first_file_pred = modelA(cmd_input).argmax(dim=1, keepdim=True)
    predicted_files = [first_file_pred.item()]
    # 続くファイルを段階的に推測
    for _ in range(max_steps-1):
        file_hist_tensor = torch.tensor(predicted_files).unsqueeze(0)
        next_file_pred = modelB(cmd_input, file_hist_tensor).argmax(dim=1)
        predicted_files.append(next_file_pred.item())
    return predicted_files

def evaluate_modelA(modelA, eval_loader):
    """Model Aの評価: 最初のファイル予測精度(Accuracy)など"""
    modelA.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in eval_loader:
            cmd_input, target_file = batch
            output = modelA(cmd_input)
            preds = output.argmax(dim=1)
            correct += (preds == target_file).sum().item()
            total += target_file.size(0)
    return correct / total if total > 0 else 0

def evaluate_modelB(modelB, eval_loader, k=5):
    """Model Bの評価: 次のファイルTop-k Accuracyなど"""
    modelB.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in eval_loader:
            cmd_input, file_history, next_file = batch
            output = modelB(cmd_input, file_history)
            _, topk_indices = output.topk(k, dim=1)
            for i in range(next_file.size(0)):
                if next_file[i].item() in topk_indices[i]:
                    correct += 1
            total += next_file.size(0)
    return correct / total if total > 0 else 0

def tokenize_command(command):
    """コマンド文字列をトークナイズしてトークンのリストを返す"""
    # 簡単なスペース区切りのトークナイズ
    tokens = re.findall(r'\b\w+\b', command.lower())
    return tokens

def build_vocab(commands, min_freq=1):
    """コマンドから語彙を構築"""
    from collections import Counter
    counter = Counter()
    for cmd in commands:
        tokens = tokenize_command(cmd)
        counter.update(tokens)
    vocab = {token: idx+1 for idx, (token, freq) in enumerate(counter.items()) if freq >= min_freq}
    vocab['<PAD>'] = 0
    return vocab

def build_file_map(accessed_files):
    """ファイルパスをIDにマッピング"""
    file_set = set()
    for files in accessed_files:
        for file in eval(files):
            file_set.add(file['path'])
    file_map = {file: idx for idx, file in enumerate(sorted(file_set))}
    return file_map

def encode_command(command, vocab):
    """コマンドをIDのシーケンスにエンコード"""
    tokens = tokenize_command(command)
    return [vocab.get(token, 0) for token in tokens]

def encode_files(accessed_files, file_map):
    """ファイルパスをIDのリストにエンコード"""
    file_ids = []
    for files in accessed_files:
        for file in eval(files):
            file_id = file_map.get(file['path'], 0)
            file_ids.append(file_id)
    return file_ids

class CommandDatasetA(Dataset):
    """Model A 用のデータセット: コマンド -> 最初のファイル"""
    def __init__(self, commands, first_files, vocab):
        self.commands = commands
        self.first_files = first_files
        self.vocab = vocab
        # 文字列を数値にマッピングする辞書を作成
        self.file_to_idx = {file: idx for idx, file in enumerate(sorted(set(self.first_files)))}
        # `self.first_files` を数値にエンコード
        self.first_files = [self.file_to_idx[file] for file in self.first_files]

    def __len__(self):
        return len(self.commands)

    def __getitem__(self, idx):
        cmd_encoded = encode_command(self.commands[idx], self.vocab)
        # パディング
        cmd_padded = cmd_encoded + [0]*(10 - len(cmd_encoded)) if len(cmd_encoded) < 10 else cmd_encoded[:10]
        return torch.tensor(cmd_padded, dtype=torch.long), torch.tensor(self.first_files[idx], dtype=torch.long)

class CommandDatasetB(Dataset):
    """Model B 用のデータセット: コマンド + ファイル履歴 -> 次のファイル"""
    def __init__(self, commands, file_sequences, vocab, file_map):
        self.commands = commands
        self.file_sequences = file_sequences
        self.vocab = vocab
        self.file_map = file_map

    def __len__(self):
        return len(self.commands)

    def __getitem__(self, idx):
        cmd_encoded = encode_command(self.commands[idx], self.vocab)
        cmd_padded = cmd_encoded + [0]*(10 - len(cmd_encoded)) if len(cmd_encoded) < 10 else cmd_encoded[:10]
        files = [self.file_map.get(file['path'], 0) for file in eval(self.file_sequences[idx])]
        # 最初のファイルを除く履歴と次のファイル
        if len(files) < 2:
            file_history = files[:-1] + [0]*(10 - len(files))  # パディング
            next_file = files[-1] if files else 0
        else:
            file_history = files[:-1] + [0]*(10 - len(files)) if len(files) < 10 else files[:9]
            next_file = files[-1]
        file_history_padded = file_history[:10]
        return torch.tensor(cmd_padded, dtype=torch.long), torch.tensor(file_history_padded, dtype=torch.long), torch.tensor(next_file, dtype=torch.long)

def train_model_by_type(df: pd.DataFrame, model_dir: str):
    """base_image_typeごとにモデルを学習"""
    results = {}
    
    image_types = df['base_image_type'].unique()
    
    # タイプごとにデータを分割
    for image_type in image_types:
        print(f"\n=== Training model for {image_type} type ===")
        type_df = df[df['base_image_type'] == image_type]
        
        if len(type_df) < 10:
            print(f"Skipping {image_type}: insufficient data")
            continue
            
        # モデルのパスにタイプを含める
        type_model_path = os.path.join(model_dir, f'model_{image_type}.pt')
        
        # データの準備
        commands = type_df['command'].tolist()
        accessed_files = type_df['accessed_files'].tolist()
        
        # 語彙とファイルマップの構築
        vocab = build_vocab(commands)
        file_map = build_file_map(accessed_files)
        
        # データセットの作成と学習の実行
        train_val_split = train_test_split(commands, accessed_files, test_size=0.2)
        commands_train, commands_val, files_train, files_val = train_val_split
        
        # 以下、既存のモデル学習処理を実行
        datasetA_train = CommandDatasetA(commands_train, files_train, vocab)
        datasetA_val = CommandDatasetA(commands_val, files_val, vocab)
        
        datasetB_train = CommandDatasetB(commands_train, files_train, vocab, file_map)
        datasetB_val = CommandDatasetB(commands_val, files_val, vocab, file_map)
        
        train_loaderA = DataLoader(datasetA_train, batch_size=32, shuffle=True)
        eval_loaderA = DataLoader(datasetA_val, batch_size=32, shuffle=False)
        
        train_loaderB = DataLoader(datasetB_train, batch_size=32, shuffle=True)
        eval_loaderB = DataLoader(datasetB_val, batch_size=32, shuffle=False)
        
        # ハイパーパラメータ設定
        vocab_size = len(vocab)
        file_size = len(file_map)
        embed_dim = 64
        hidden_dim = 128
        num_files = file_size
        epochs = 10  # エポック数を設定
        
        # モデルの初期化
        modelA = ModelA(vocab_size=vocab_size, embed_dim=embed_dim, hidden_dim=hidden_dim, num_files=num_files)
        modelB = ModelB(vocab_size=vocab_size, file_size=file_size, embed_dim=embed_dim, hidden_dim=hidden_dim)
        
        criterion = nn.CrossEntropyLoss()
        optimizerA = torch.optim.Adam(modelA.parameters(), lr=0.001)
        optimizerB = torch.optim.Adam(modelB.parameters(), lr=0.001)
        
        # トレーニングループ
        for epoch in range(1, epochs + 1):
            train_modelA(modelA, train_loaderA, criterion, optimizerA, epoch)
            train_modelB(modelB, train_loaderB, criterion, optimizerB, epoch)
            
            # 評価
            accuracyA = evaluate_modelA(modelA, eval_loaderA)
            topkB = evaluate_modelB(modelB, eval_loaderB, k=5)
            print(f"Epoch {epoch} - ModelA Accuracy: {accuracyA:.3f}, ModelB Top-5 Accuracy: {topkB:.3f}")
        
        # モデルの保存
        torch.save({
            'modelA_state': modelA.state_dict(),
            'modelB_state': modelB.state_dict(),
            'vocab': vocab,
            'file_map': file_map
        }, type_model_path)
        
        results[image_type] = {
            'accuracy_A': accuracyA,
            'accuracy_B_top5': topkB,
            'data_size': len(type_df)
        }
    
    return results

def main():
    if len(sys.argv) != 3:
        print("Usage: train_gru_model_v3.py <input_csv> <model_dir>")
        sys.exit(1)
    
    input_csv = sys.argv[1]
    model_dir = sys.argv[2]
    os.makedirs(model_dir, exist_ok=True)
    
    # CSVファイルの読み込み
    print(f"Loading data from: {input_csv}")
    df = pd.read_csv(input_csv)
    
    if 'base_image_type' not in df.columns:
        print("Error: CSV must contain 'base_image_type' column")
        sys.exit(1)
    
    # タイプごとのモデル学習
    results = train_model_by_type(df, model_dir)
    
    # 結果の保存
    results_path = os.path.join(model_dir, 'training_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n=== Training Results ===")
    for image_type, metrics in results.items():
        print(f"\n{image_type}:")
        print(f"- Data size: {metrics['data_size']}")
        print(f"- Model A accuracy: {metrics['accuracy_A']:.4f}")
        print(f"- Model B top-5 accuracy: {metrics['accuracy_B_top5']:.4f}")

if __name__ == "__main__":
    main()