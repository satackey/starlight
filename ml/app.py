# import psycopg2
import pandas as pd
# from gensim.models import FastText

# データベース接続情報
DB_CONFIG = {
    "host": "host.docker.internal",
    "dbname": "postgres",
    "user": "postgres",
    "password": "postgres",
    "port": 5432
}

# ファイルパスをトークン化する関数
def tokenize_path(path):
    return path.strip('/').replace('.', '/').split('/')

# PostgreSQLデータを取得する関数
def fetch_file_paths():
    query = 'SELECT id, file FROM file WHERE "order" IS NOT NULL'
    with psycopg2.connect(**DB_CONFIG) as conn:
        with conn.cursor() as cur:
            cur.execute(query)
            rows = cur.fetchall()
    return rows

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
    # PostgreSQLからデータを取得
    data = fetch_file_paths()
    
    # # トークン化したパスのリストを作成
    # tokenized_paths = [tokenize_path(row[1]) for row in data]
    
    # # FastTextモデルの学習
    # model = FastText(vector_size=100, window=3, min_count=1, workers=4)
    # model.build_vocab(sentences=tokenized_paths)
    # model.train(sentences=tokenized_paths, total_examples=len(tokenized_paths), epochs=10)
    
    # # 各ファイルパスの埋め込みを計算
    # results = []
    # for row in data:
    #     id, path = row
    #     embedding = get_path_embedding(path, model)
    #     if embedding is not None:
    #         results.append({
    #             "id": id,
    #             "path": path,
    #             "embedding": embedding.tolist()  # リストを文字列に変換
    #         })
    
    # pandasでDataFrameを作成
    df = pd.DataFrame(results)
    
    # DataFrameをCSVファイルに保存
    output_file = "file_embeddings.csv"
    df.to_csv(output_file, index=False)
    print(f"CSVファイルが保存されました: {output_file}")

if __name__ == "__main__":
    main()
