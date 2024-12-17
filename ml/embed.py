import pandas as pd
from gensim.models import FastText

# ファイルパスをトークン化する関数
def tokenize_path(path):
    return path.strip('/').replace('.', '/').split('/')

# 埋め込みを計算する関数
def get_path_embedding(path, model):
    tokens = tokenize_path(path)
    embeddings = [model.wv[token] for token in tokens if token in model.wv]
    if embeddings:
        return sum(embeddings) / len(embeddings)  # 平均ベクトルを計算
    else:
        return None

# メイン処理
def main():
    # ファイルパスCSVを読み込む
    input_file = "file_paths.csv"
    df = pd.read_csv(input_file)
    
    # トークン化したパスのリストを作成
    tokenized_paths = [tokenize_path(path) for path in df["file"]]
    
    # FastTextモデルの学習
    model = FastText(vector_size=100, window=3, min_count=1, workers=4)
    model.build_vocab(sentences=tokenized_paths)
    model.train(sentences=tokenized_paths, total_examples=len(tokenized_paths), epochs=10)
    
    # 各ファイルパスの埋め込みを計算
    embeddings = []
    for path in df["file"]:
        embedding = get_path_embedding(path, model)
        embeddings.append(embedding.tolist() if embedding is not None else None)
    
    # 埋め込みをDataFrameに追加
    df["embedding"] = embeddings
    
    # DataFrameをCSVファイルに保存
    output_file = "file_embeddings.csv"
    df.to_csv(output_file, index=False)
    print(f"CSVファイルが保存されました: {output_file}")

if __name__ == "__main__":
    main()
