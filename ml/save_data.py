import psycopg2
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
# PostgreSQLデータを取得する関数
def fetch_file_paths():
    query = 'SELECT id, file FROM file WHERE "order" IS NOT NULL'
    with psycopg2.connect(**DB_CONFIG) as conn:
        with conn.cursor() as cur:
            cur.execute(query)
            rows = cur.fetchall()
    return rows

# メイン処理
def main():
    # データを取得
    data = fetch_file_paths()
    
    # DataFrameを作成
    df = pd.DataFrame(data, columns=["id", "file"])
    
    # CSVに保存
    output_file = "file_paths.csv"
    df.to_csv(output_file, index=False)
    print(f"CSVファイルが保存されました: {output_file}")

if __name__ == "__main__":
    main()