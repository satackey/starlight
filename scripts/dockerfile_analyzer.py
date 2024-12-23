"""
GitHub Dockerfile Analyzer
-------------------------

このスクリプトは、GitHubのパブリックリポジトリから特定の条件に合うDockerfileを検索し、
その中で使用されているコマンドを抽出・分析するツールです。

主な機能:
- GitHubのCode Search APIを使用して `node:lts` イメージを使用している Dockerfile を検索
- `node:lts-alpine` などの派生イメージは除外
- Dockerfile内の RUN コマンドを抽出（複数行コマンドに対応）
- 結果を2つのCSVファイルとして出力:
  1. 全てのコマンドを含むファイル
  2. 各Dockerfileの最初のコマンドのみを含むファイル

出力形式:
- CSV形式（2列）
  - 列1: Dockerfileへのパーマリンク
  - 列2: 抽出されたコマンド
- 2種類の出力ファイル:
  1. 全コマンド出力（--all-output）
  2. 最初のコマンドのみ出力（--first-output）

使用方法:
$ python dockerfile_analyzer.py --all-output <全コマンド出力ファイル> --first-output <最初のコマンドのみ出力ファイル>

例:
$ python dockerfile_analyzer.py --all-output dockerfile_commands.csv --first-output dockerfile_commands_first.csv

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
Updated: 2023-12-23 - 2つのCSVファイル出力機能を追加
License: [ライセンス]
"""

import requests
import csv
import re
from typing import List, Tuple
from urllib.parse import urljoin

def get_github_headers(check_rate_limit: bool = False) -> dict:
    """
    GitHub APIリクエスト用のヘッダーを生成し、必要に応じてAPIレート制限情報を表示
    環境変数GITHUB_TOKENが設定されている場合は認証ヘッダーを追加
    
    Args:
        check_rate_limit: レート制限をチェックするかどうか
    """
    import os
    from datetime import datetime, timezone
    import time
    
    headers = {
        "Accept": "application/vnd.github.v3+json"
    }
    
    token = os.getenv('GITHUB_TOKEN')
    if token:
        headers["Authorization"] = f"token {token}"
        print("GitHub Personal Access Tokenが設定されています（5000リクエスト/時）")
    else:
        print("警告: GitHub Personal Access Tokenが設定されていません（60リクエスト/時）")
        print("環境変数GITHUB_TOKENを設定することで制限を緩和できます")
    
    if check_rate_limit:
        # レート制限の確認
        check_url = "https://api.github.com/rate_limit"
        try:
            response = requests.get(check_url, headers=headers)
            if response.status_code == 200:
                data = response.json()
                remaining = data['resources']['search']['remaining']
                reset_timestamp = data['resources']['search']['reset']
                
                # UTCからローカル時間に変換
                reset_time = datetime.fromtimestamp(reset_timestamp, timezone.utc).astimezone()
                
                print(f"APIレート制限情報:")
                print(f"- 残りのリクエスト数: {remaining}")
                print(f"- 制限リセット時刻: {reset_time.strftime('%Y-%m-%d %H:%M:%S')}")
                
                # 制限に近い場合は警告
                if remaining < 10:
                    print(f"警告: APIリクエスト数が残り{remaining}回です")
                    reset_wait = reset_timestamp - time.time()
                    if reset_wait > 0:
                        print(f"制限リセットまであと約{int(reset_wait/60)}分です")
        except Exception as e:
            print(f"APIレート制限情報の取得に失敗しました: {str(e)}")
    
    return headers

def search_dockerfiles() -> List[dict]:
    """
    GitHubのAPIを使用してnode:ltsを使用しているDockerfileを検索
    """
    headers = get_github_headers(check_rate_limit=False)
    
    query = 'FROM node:lts filename:Dockerfile -filename:*alpine*'
    url = f"https://api.github.com/search/code?q={query}&per_page=100"
    
    try:
        response = requests.get(url, headers=headers)
        
        if response.status_code == 403:
            rate_limit = response.headers.get('X-RateLimit-Remaining', '不明')
            reset_time = response.headers.get('X-RateLimit-Reset', '不明')
            raise Exception(f"GitHub APIのレート制限に達しました。\n"
                          f"残りリクエスト数: {rate_limit}\n"
                          f"リセット時刻: {reset_time}")
        elif response.status_code != 200:
            raise Exception(f"GitHub APIエラー: ステータスコード {response.status_code}\n"
                          f"レスポンス: {response.text}")
        
        data = response.json()
        if 'items' not in data:
            raise Exception(f"予期しないAPIレスポンス形式です: {data}")
            
        if len(data['items']) == 0:
            print("警告: 検索条件に一致するDockerfileが見つかりませんでした")
        
        # レスポンスの構造を確認
        if data['items'] and len(data['items']) > 0:
            print("\nデバッグ情報: 最初のアイテムの構造")
            first_item = data['items'][0]
            print("利用可能なフィールド:", first_item.keys())
            print("URL:", first_item.get('url'))
            print("SHA:", first_item.get('sha'))
            print("リポジトリ情報:", first_item.get('repository', {}))
            
        return data['items']
        
    except requests.exceptions.RequestException as e:
        raise Exception(f"GitHub APIへのリクエスト中にエラーが発生しました: {str(e)}")

def get_dockerfile_content(url: str, headers: dict) -> str:
    """
    指定されたURLからDockerfileの内容を取得してデコード
    """
    import base64
    
    try:
        response = requests.get(url, headers=headers)
        
        if response.status_code == 403:
            raise Exception("Dockerfileの取得に失敗しました: アクセス権限がありません")
        elif response.status_code == 404:
            raise Exception("Dockerfileの取得に失敗しました: ファイルが見つかりません")
        elif response.status_code != 200:
            raise Exception(f"Dockerfileの取得に失敗しました: ステータスコード {response.status_code}\n"
                          f"レスポンス: {response.text}")
        
        try:
            data = response.json()
            if 'content' not in data:
                raise Exception(f"予期しないAPIレスポンス形式です: {data}")
                
            # GitHubのAPIはbase64エンコードされたコンテンツを返すのでデコード
            content = data['content']
            try:
                decoded_content = base64.b64decode(content).decode('utf-8')
                return decoded_content
            except Exception as e:
                raise Exception(f"コンテンツのデコードに失敗しました: {str(e)}")
                
        except ValueError as e:
            raise Exception(f"JSONのパースに失敗しました: {str(e)}")
            
    except requests.exceptions.RequestException as e:
        raise Exception(f"Dockerfileの取得中にネットワークエラーが発生しました: {str(e)}")

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
    import argparse

    # コマンドライン引数の設定
    parser = argparse.ArgumentParser(description='Dockerfile解析ツール')
    parser.add_argument('--all-output', required=True, help='全コマンドを出力するCSVファイルのパス')
    parser.add_argument('--first-output', required=True, help='最初のコマンドのみを出力するCSVファイルのパス')
    args = parser.parse_args()

    try:
        # 初回のみレート制限をチェック
        headers = get_github_headers(check_rate_limit=True)
        
        print("GitHubからDockerfileを検索中...")
        dockerfiles = search_dockerfiles()
        total_files = len(dockerfiles)
        print(f"検索結果: {total_files}件のDockerfileが見つかりました")
        
        processed = 0
        success_all = 0
        success_first = 0
        errors = 0
        
        # 結果を保存するCSVファイルを開く
        with open(args.all_output, 'w', newline='', encoding='utf-8') as f_all, \
             open(args.first_output, 'w', newline='', encoding='utf-8') as f_first:
            writer_all = csv.writer(f_all)
            writer_first = csv.writer(f_first)
            
            # ヘッダー行を書き込む
            writer_all.writerow(['Dockerfile URL', 'Command'])
            writer_first.writerow(['Dockerfile URL', 'Command'])
            
            for i, dockerfile in enumerate(dockerfiles, 1):
                try:
                    raw_url = dockerfile['url']
                    repo_name = dockerfile['repository']['full_name']
                    print(f"\n処理中 ({i}/{total_files}): {repo_name}")
                    
                    # URLからrefパラメータを抽出してコミットSHAを取得
                    from urllib.parse import urlparse, parse_qs
                    parsed_url = urlparse(dockerfile['url'])
                    query_params = parse_qs(parsed_url.query)
                    commit_sha = query_params.get('ref', [''])[0]
                    
                    if not commit_sha:
                        raise Exception("コミットSHAの取得に失敗しました")
                    
                    # GitHubの正しいパーマリンク形式を使用
                    permalink = f"https://github.com/{repo_name}/blob/{commit_sha}/{dockerfile['path']}"
                    
                    # Dockerfileの内容を取得（レート制限チェックなし）
                    content = get_dockerfile_content(raw_url, get_github_headers(check_rate_limit=False))
                    
                    # Dockerfileをパース
                    uses_node_lts, commands = parse_dockerfile(content)
                    
                    # node:ltsを使用している場合のみ結果を出力
                    if uses_node_lts and commands:
                        # 全てのコマンドを出力
                        for command in commands:
                            writer_all.writerow([permalink, command])
                            success_all += 1
                        
                        # 最初のコマンドのみを出力
                        writer_first.writerow([permalink, commands[0]])
                        success_first += 1
                    
                    processed += 1
                    
                except Exception as e:
                    print(f"エラー: {str(e)}")
                    errors += 1
                    continue
        
        print("\n処理完了:")
        print(f"- 処理したDockerfile: {processed}/{total_files}")
        print(f"- 抽出した全コマンド数: {success_all}")
        print(f"- 抽出した最初のコマンド数: {success_first}")
        print(f"- エラー数: {errors}")
        
    except Exception as e:
        print(f"致命的なエラーが発生しました: {str(e)}")

if __name__ == "__main__":
    main()
