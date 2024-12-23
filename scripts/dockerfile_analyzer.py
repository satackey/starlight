"""
GitHub Dockerfile Analyzer
-------------------------

このスクリプトは、GitHubのパブリックリポジトリから特定の条件に合うDockerfileを検索し、
その中で使用されているコマンドを抽出・分析するツールです。

主な機能:
- GitHubのCode Search APIを使用して `node:lts` イメージを使用している Dockerfile を検索
- `node:lts-alpine` などの派生イメージは除外
- Dockerfile内の RUN コマンドを抽出（複数行コマンドに対応）
- 結果をCSVファイルとして出力

出力形式:
- CSV形式（2列）
  - 列1: Dockerfileへのパーマリンク
  - 列2: 抽出されたコマンド

使用方法:
$ python dockerfile_analyzer.py

必要なパッケージ:
- requests

注意事項:
1. GitHub APIの利用制限
   - 認証なし: 60リクエスト/時
   - 認証あり: 5000リクエスト/時
2. 大量のDockerfileを分析する場合は、Personal Access Tokenの使用を推奨
3. API制限に達した場合は適切なエラーメッセージが表示されます

環境変数:
GITHUB_TOKEN - GitHub Personal Access Token（オプション）

Author: [作成者名]
Created: [作成日]
License: [ライセンス]
"""

import requests
import csv
import re
from typing import List, Tuple
from urllib.parse import urljoin

def get_github_headers() -> dict:
    """
    GitHub APIリクエスト用のヘッダーを生成
    環境変数GITHUB_TOKENが設定されている場合は認証ヘッダーを追加
    """
    import os
    
    headers = {
        "Accept": "application/vnd.github.v3+json"
    }
    
    token = os.getenv('GITHUB_TOKEN')
    if token:
        headers["Authorization"] = f"token {token}"
    
    return headers

def search_dockerfiles() -> List[dict]:
    """
    GitHubのAPIを使用してnode:ltsを使用しているDockerfileを検索
    """
    headers = get_github_headers()
    
    query = 'FROM node:lts filename:Dockerfile -filename:*alpine*'
    url = f"https://api.github.com/search/code?q={query}&per_page=100"
    
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        raise Exception(f"GitHub API error: {response.status_code}")
        
    return response.json()['items']

def get_dockerfile_content(url: str, headers: dict) -> str:
    """
    指定されたURLからDockerfileの内容を取得
    """
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        raise Exception(f"Failed to fetch Dockerfile: {response.status_code}")
        
    return response.json()['content']

def parse_dockerfile(content: str) -> Tuple[bool, List[str]]:
    """
    Dockerfileをパースしてnode:ltsの使用を確認し、RUNコマンドを抽出
    
    Returns:
        Tuple[bool, List[str]]: (node:ltsを使用しているか, RUNコマンドのリスト)
    """
    lines = content.split('\n')
    run_commands = []
    uses_node_lts = False
    current_command = ""
    
    for line in lines:
        line = line.strip()
        
        # 空行をスキップ
        if not line:
            continue
            
        # コメント行をスキップ
        if line.startswith('#'):
            continue
            
        # FROM命令を確認
        if line.startswith('FROM'):
            # node:lts-alpine は除外
            if 'node:lts' in line and 'alpine' not in line:
                uses_node_lts = True
                
        # RUN命令を処理
        if line.startswith('RUN'):
            current_command = line[3:].strip()
            # バックスラッシュで終わる場合は継続
            if current_command.endswith('\\'):
                current_command = current_command[:-1].strip()
                continue
            else:
                run_commands.append(current_command)
                current_command = ""
        elif current_command:
            # 継続行の処理
            line = line.strip()
            if line.endswith('\\'):
                current_command += " " + line[:-1].strip()
            else:
                current_command += " " + line
                run_commands.append(current_command)
                current_command = ""
                
    return uses_node_lts, run_commands

def main():
    """
    メイン処理
    """
    # 結果を保存するCSVファイルを開く
    with open('dockerfile_commands.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Dockerfile URL', 'Command'])
        
        # Dockerfileを検索
        dockerfiles = search_dockerfiles()
        
        for dockerfile in dockerfiles:
            try:
                raw_url = dockerfile['url']
                permalink = urljoin(
                    "https://github.com",
                    f"{dockerfile['repository']['full_name']}/blob/{dockerfile['sha']}/{dockerfile['path']}"
                )
                
                # Dockerfileの内容を取得
                content = get_dockerfile_content(raw_url, get_github_headers())
                
                # Dockerfileをパース
                uses_node_lts, commands = parse_dockerfile(content)
                
                # node:ltsを使用している場合のみ結果を出力
                if uses_node_lts:
                    for command in commands:
                        writer.writerow([permalink, command])
                        
            except Exception as e:
                print(f"Error processing {raw_url}: {str(e)}")
                continue

if __name__ == "__main__":
    main()
