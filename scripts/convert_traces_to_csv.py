#!/usr/bin/env python3
"""
ファイルアクセストレース変換スクリプト
--------------------------------------

collect_traces.py で収集されたデータを機械学習用のCSVデータに変換します。

入力:
- ファイルアクセストレース収集結果のCSV
- PostgreSQLデータベース内のトレース情報

出力:
- 学習用CSVファイル
  - コマンド情報
  - アクセスされたファイル情報（パス、サイズ、順序）
"""

import os
import sys
import json
import psycopg2
import pandas as pd
from typing import List, Dict, Tuple

def connect_to_db(connection_string: str) -> psycopg2.extensions.connection:
    """PostgreSQLデータベースに接続"""
    try:
        print(f"データベースに接続中...")
        conn = psycopg2.connect(connection_string)
        print("接続成功")
        return conn
    except Exception as e:
        print(f"データベース接続エラー: {str(e)}")
        sys.exit(1)

def get_file_access_traces(conn: psycopg2.extensions.connection, image_name: str) -> List[Dict]:
    """
    指定されたイメージのファイルアクセストレースを取得
    
    Returns:
        List[Dict]: アクセスされたファイルのリスト（パス、サイズ、順序を含む）
    """
    cursor = conn.cursor()
    
    try:
        # イメージIDを取得
        print(f"イメージの検索: {image_name}")
        cursor.execute("""
            SELECT id, image, hash FROM image 
            WHERE image = %s
        """, (image_name,))
        
        image_records = cursor.fetchall()
        if not image_records:
            print(f"イメージが見つかりません: {image_name}")
            # 部分一致で再試行
            cursor.execute("""
                SELECT id, image, hash FROM image 
                WHERE image LIKE %s
            """, (f"%{image_name}%",))
            image_records = cursor.fetchall()
            if not image_records:
                print("部分一致でも見つかりません")
                return []
        
        print(f"見つかったイメージ:")
        for record in image_records:
            print(f"- ID: {record[0]}, イメージ: {record[1]}, ハッシュ: {record[2]}")
        
        image_id = image_records[0][0]  # 最初のマッチを使用
        
        # ファイルアクセストレースを取得
        print(f"ファイルアクセストレースの取得中...")
        cursor.execute("""
            SELECT DISTINCT ON (f.file)
                f.file as path,
                f.size,
                f."order"[1] as first_access,
                ROW_NUMBER() OVER (ORDER BY f."order"[1]) as access_order
            FROM layer l
            JOIN filesystem fs ON l.layer = fs.id
            JOIN file f ON f.fs = fs.id
            WHERE l.image = %s
            AND f."order" IS NOT NULL
            AND array_length(f."order", 1) > 0
            ORDER BY f.file, f."order"[1];
        """, (image_id,))
        
        files = []
        for path, size, first_access, order in cursor.fetchall():
            files.append({
                'path': path,
                'size': size,
                'order': order
            })
        
        print(f"取得したファイル数: {len(files)}")
        if files:
            print("最初の5件:")
            for f in files[:5]:
                print(f"- {f['path']} (size: {f['size']}, order: {f['order']})")
        
        return files
    
    except Exception as e:
        print(f"エラー発生: {str(e)}")
        cursor.execute("ROLLBACK")
        return []
    
    finally:
        cursor.close()

def process_image_name(image_name: str) -> str:
    """イメージ名の前処理"""
    try:
        # cloud.cluster.local:5000/ を除去
        if 'cloud.cluster.local:5000/' in image_name:
            image_name = image_name.split('cloud.cluster.local:5000/')[1]
        
        # :starlight を除去
        if ':starlight' in image_name:
            image_name = image_name.split(':starlight')[0]
        
        # node-lts/ を除去
        if 'node-lts/' in image_name:
            image_name = image_name.split('node-lts/')[1]
        elif 'node-lts-' in image_name:
            image_name = image_name.split('/', 1)[1]
        
        print(f"処理後のイメージ名: {image_name}")
        return image_name
    except Exception as e:
        print(f"イメージ名の処理中にエラー: {str(e)}")
        return image_name

def extract_base_image_name(image_name: str) -> str:
    """イメージ名からベースイメージ名（最初のコンポーネント）を抽出"""
    try:
        # cloud.cluster.local:5000/ を除去
        if 'cloud.cluster.local:5000/' in image_name:
            image_name = image_name.split('cloud.cluster.local:5000/')[1]
        
        # パスの最初のコンポーネントを取得
        components = image_name.split('/')
        if len(components) > 1:
            return components[0]
        else:
            return "unknown"
    except Exception as e:
        print(f"ベースイメージ名の抽出中にエラー: {str(e)}")
        return "unknown"

def get_base_image_type(base_image_name: str) -> str:
    """ベースイメージ名からイメージタイプを判定"""
    base_image_name = base_image_name.lower()
    if base_image_name.startswith('node'):
        return 'node'
    elif base_image_name.startswith('ubuntu'):
        return 'ubuntu'
    return 'other'

def main():
    if len(sys.argv) != 3:
        print("Usage: convert_traces_to_csv.py <input_csv> <output_csv>")
        sys.exit(1)
    
    input_csv = sys.argv[1]
    output_csv = sys.argv[2]
    
    # データベース接続文字列
    db_conn_string = os.getenv('POSTGRES_CONNECTION_STRING')
    if not db_conn_string:
        print("POSTGRES_CONNECTION_STRING環境変数が設定されていません")
        sys.exit(1)
    
    # データベースに接続
    conn = connect_to_db(db_conn_string)
    conn.autocommit = True  # 自動コミットを有効化
    
    try:
        # 入力CSVを読み込み
        print(f"入力CSVの読み込み中: {input_csv}")
        df_input = pd.read_csv(input_csv)
        print(f"読み込み完了: {len(df_input)}行")
        print("\nカラム一覧:")
        for col in df_input.columns:
            print(f"- {col}")
        
        # 各イメージのトレースを取得
        rows = []
        for idx, row in df_input.iterrows():
            print(f"\n行 {idx+1}/{len(df_input)} の処理中...")
            
            # コマンドの取得
            command = row.get('Command')
            if pd.isna(command):
                print("コマンドがありません")
                continue
            
            # イメージ名の取得
            image_name = row.get('Converted Repotag')
            if pd.isna(image_name):
                print("イメージ名がありません")
                continue
            
            print(f"コマンド: {command}")
            print(f"元のイメージ名: {image_name}")
            
            # イメージ名の処理
            processed_image_name = process_image_name(image_name)
            print(f"処理後のイメージ名: {processed_image_name}")
            
            # ベースイメージ名の抽出とタイプの判定
            base_image_name = extract_base_image_name(image_name)
            base_image_type = get_base_image_type(base_image_name)
            print(f"ベースイメージ名: {base_image_name} (タイプ: {base_image_type})")
            
            # トレースを取得
            accessed_files = get_file_access_traces(conn, processed_image_name)
            if not accessed_files:
                print("ファイルアクセストレースなし")
                continue
            
            # 行データを作成
            rows.append({
                'command': command,
                'accessed_files': json.dumps(accessed_files, ensure_ascii=False),
                'base_image_name': base_image_name,
                'base_image_type': base_image_type,  # 新しいカラムを追加
            })
            print(f"データ追加: トレース数 {len(accessed_files)}")
        
        # DataFrameを作成
        df_output = pd.DataFrame(rows)
        
        # CSVとして保存
        print(f"\nデータを保存中: {output_csv}")
        df_output.to_csv(output_csv, index=False)
        
        print(f"\n処理完了:")
        print(f"- 入力コマンド数: {len(df_input)}")
        print(f"- 出力コマンド数: {len(df_output)}")
        print(f"- 出力ファイル: {output_csv}")
    
    except Exception as e:
        print(f"エラー発生: {str(e)}")
        sys.exit(1)
    
    finally:
        conn.close()
        print("データベース接続を閉じました")

if __name__ == "__main__":
    main()
