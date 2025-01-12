#!/usr/bin/env python3
"""
ファイルアクセストレース変換スクリプト
--------------------------------------

collect_traces.py で収集されたデータを機械学習用のCSVデータに変換します。

入力:
- PostgreSQLデータベース内のトレース情報
- 実行コマンドのCSVファイル

出力:
- 学習用CSVファイル（全データ）
  - コマンド情報
  - アクセスされたファイルのシーケンス
  - 次にアクセスされるファイル
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
        conn = psycopg2.connect(connection_string)
        return conn
    except Exception as e:
        print(f"データベース接続エラー: {str(e)}")
        sys.exit(1)

def get_file_access_traces(conn: psycopg2.extensions.connection, image_name: str) -> List[Dict]:
    """
    指定されたイメージのファイルアクセストレースを取得
    
    Args:
        conn: データベース接続
        image_name: イメージ名 (Converted Repotag)
    
    Returns:
        ファイルアクセストレースのリスト
    """
    cursor = conn.cursor()
    
    # イメージIDを取得
    cursor.execute("""
        SELECT id FROM image 
        WHERE image = %s
    """, (image_name,))
    
    image_id = cursor.fetchone()
    if not image_id:
        print(f"イメージが見つかりません: {image_name}")
        return []
    
    # ファイルアクセストレースを取得
    cursor.execute("""
        SELECT 
            f.file,
            f."order",
            f.metadata,
            l."stackIndex",
            f.size,
            f.hash
        FROM file f
        JOIN layer l ON f.fs = l.layer
        WHERE l.image = %s
        AND f."order" IS NOT NULL
        ORDER BY l."stackIndex", 
                 (SELECT AVG(o) FROM UNNEST(f."order") o)
    """, (image_id[0],))
    
    traces = []
    for row in cursor.fetchall():
        file_path, access_order, metadata, stack_index, size, file_hash = row
        if access_order:  # アクセス順序が記録されている場合のみ
            trace = {
                'file_path': file_path,
                'access_order': access_order,
                'metadata': metadata if isinstance(metadata, dict) else json.loads(metadata) if metadata else {},
                'stack_index': stack_index,
                'size': size,
                'hash': file_hash
            }
            traces.append(trace)
    
    cursor.close()
    return traces

def create_sequence_data(traces: List[Dict], command: str, image_name: str, sequence_length: int = 5) -> List[Dict]:
    """
    ファイルアクセストレースからシーケンスデータを生成
    
    Args:
        traces: ファイルアクセストレースのリスト
        command: 実行コマンド
        image_name: イメージ名
        sequence_length: 入力シーケンスの長さ
    
    Returns:
        シーケンスデータのリスト
    """
    sequences = []
    
    # アクセス順でソート
    sorted_traces = sorted(traces, key=lambda x: min(x['access_order']))
    
    # シーケンスデータを生成
    for i in range(len(sorted_traces) - sequence_length):
        sequence = sorted_traces[i:i+sequence_length]
        next_file = sorted_traces[i+sequence_length]
        
        # 入力シーケンスの各ファイルの情報
        input_sequence = []
        for t in sequence:
            file_info = {
                'path': t['file_path'],
                'stack_index': t['stack_index'],
                'size': t['size'],
                'hash': t['hash'],
                'type': t['metadata'].get('type', ''),
                'mode': t['metadata'].get('mode', 0)
            }
            input_sequence.append(file_info)
        
        # 次のファイル（予測対象）の情報
        target_info = {
            'path': next_file['file_path'],
            'stack_index': next_file['stack_index'],
            'size': next_file['size'],
            'hash': next_file['hash'],
            'type': next_file['metadata'].get('type', ''),
            'mode': next_file['metadata'].get('mode', 0)
        }
        
        sequence_data = {
            'image_name': image_name,
            'command': command,
            'input_sequence': input_sequence,
            'target': target_info
        }
        sequences.append(sequence_data)
    
    return sequences

def main():
    if len(sys.argv) != 3:
        print("Usage: convert_traces_to_csv.py <commands_csv> <output_csv>")
        sys.exit(1)
    
    commands_csv = sys.argv[1]
    output_csv = sys.argv[2]
    
    # データベース接続文字列
    db_conn_string = os.getenv('POSTGRES_CONNECTION_STRING', 
                              'postgresql://postgres:postgres@localhost:5432/postgres')
    
    # データベースに接続
    conn = connect_to_db(db_conn_string)
    
    # コマンドCSVを読み込み
    commands_df = pd.read_csv(commands_csv)
    
    all_sequences = []
    image_sequence_counts = {}  # イメージごとのシーケンス数を追跡
    
    # 各イメージのトレースを処理
    for _, row in commands_df.iterrows():
        # repotagからtagを削除
        image_name = str(row['Converted Repotag']).rsplit(':', 1)[0]
        command = row['Command']
        
        print(f"処理中: {image_name}")
        
        # トレースを取得
        traces = get_file_access_traces(conn, image_name)
        if not traces:
            continue
        
        # シーケンスデータを生成
        sequences = create_sequence_data(traces, command, image_name)
        image_sequence_counts[image_name] = len(sequences)
        all_sequences.extend(sequences)
    
    # データフレームに変換
    rows = []
    for seq in all_sequences:
        row = {
            'image_name': seq['image_name'],
            'command': seq['command'],
            'input_files': json.dumps([f['path'] for f in seq['input_sequence']]),
            'input_stack_indices': json.dumps([f['stack_index'] for f in seq['input_sequence']]),
            'input_sizes': json.dumps([f['size'] for f in seq['input_sequence']]),
            'input_hashes': json.dumps([f['hash'] for f in seq['input_sequence']]),
            'input_types': json.dumps([f['type'] for f in seq['input_sequence']]),
            'input_modes': json.dumps([f['mode'] for f in seq['input_sequence']]),
            'target_file': seq['target']['path'],
            'target_stack_index': seq['target']['stack_index'],
            'target_size': seq['target']['size'],
            'target_hash': seq['target']['hash'],
            'target_type': seq['target']['type'],
            'target_mode': seq['target']['mode']
        }
        rows.append(row)
    
    # CSVとして保存
    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)
    
    print(f"\n処理完了:")
    print(f"- 総シーケンス数: {len(rows)}")
    print(f"- イメージ数: {len(image_sequence_counts)}")
    print(f"- イメージごとの平均シーケンス数: {len(rows) / len(image_sequence_counts):.1f}")
    print(f"- 出力ファイル: {output_csv}")
    
    conn.close()

if __name__ == "__main__":
    main()
