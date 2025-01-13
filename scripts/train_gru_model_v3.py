import torch
import torch.nn as nn
import pandas as pd
import sys

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
        cmd_emb = self.cmd_embedding(cmd_input).unsqueeze(1)
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

def train_modelA(modelA, train_loader, criterion, optimizer):
    """Model Aの学習ループ"""
    modelA.train()
    for batch in train_loader:
        cmd_input, target_file = batch
        optimizer.zero_grad()
        output = modelA(cmd_input)
        loss = criterion(output, target_file)
        loss.backward()
        optimizer.step()

def train_modelB(modelB, train_loader, criterion, optimizer):
    """Model Bの学習ループ"""
    modelB.train()
    for batch in train_loader:
        cmd_input, file_history, next_file = batch
        optimizer.zero_grad()
        output = modelB(cmd_input, file_history)
        loss = criterion(output, next_file)
        loss.backward()
        optimizer.step()

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

def main():
    csv_path = sys.argv[1]
    commands, accessed_files = load_data(csv_path)
    # ...コマンドID化、ファイルID化など前処理...

    # 例: vocab_size=5000, file_size=3000 など適当に設定
    modelA = ModelA(vocab_size=5000, embed_dim=64, hidden_dim=128, num_files=3000)
    modelB = ModelB(vocab_size=5000, file_size=3000, embed_dim=64, hidden_dim=128)

    # 例: DataLoaderのdummy生成(実際はDataset作成)
    train_loaderA = []
    train_loaderB = []
    eval_loaderA = []
    eval_loaderB = []
    criterion = nn.CrossEntropyLoss()
    optimizerA = torch.optim.Adam(modelA.parameters())
    optimizerB = torch.optim.Adam(modelB.parameters())

    # Model Aを学習
    train_modelA(modelA, train_loaderA, criterion, optimizerA)
    # Model Bを学習
    train_modelB(modelB, train_loaderB, criterion, optimizerB)

    # 評価
    accuracyA = evaluate_modelA(modelA, eval_loaderA)
    topkB = evaluate_modelB(modelB, eval_loaderB, k=5)
    print(f"ModelA Accuracy: {accuracyA:.3f}")
    print(f"ModelB Top-5 Accuracy: {topkB:.3f}")

    # 推論サンプル
    cmd_input = torch.randint(0, 5000, (1, 10))  # ダミーのコマンド入力
    result = combined_inference(modelA, modelB, cmd_input, max_steps=5)
    print("推定結果:", result)

if __name__ == "__main__":
    main()
