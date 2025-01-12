"""
GitHub Dockerfile Analyzer
-------------------------

このスクリプトは、GitHubのパブリックリポジトリから特定の条件に合うDockerfileを検索し、
その中で使用されているコマンドを抽出・分析するツールです。

主な機能:
- GitHubのCode Search APIを使用して指定されたベースイメージを使用している Dockerfile を検索
- `alpine` などの派生イメージは除外
- Dockerfile内の RUN コマンドを抽出（複数行コマンドに対応）
- 結果を2つのCSVファイルとして出力:
  1. 全てのコマンドを含むファイル
  2. 各Dockerfileの最初のコマンドのみを含むファイル

出力形式:
- CSV形式（4列）
  - 列1: Base Image
  - 列2: Command
  - 列3: Intermediate Dockerfile Command
  - 列4: Dockerfile URL
- 2種類の出力ファイル:
  1. 全コマンド出力（--all-output）
  2. 最初のコマンドのみ出力（--first-output）

使用方法:
$ python dockerfile_analyzer.py --all-output <全コマンド出力ファイル> --first-output <最初のコマンドのみ出力ファイル> --image <ベースイメージ> [--image <ベースイメージ>...] [--count <イメージごとの処理件数>]

例:
$ python dockerfile_analyzer.py --all-output dockerfile_commands.csv --first-output dockerfile_commands_first.csv --image node:lts --image ubuntu:22.04 --count 10

注: --count オプションで指定した件数は、各イメージごとの処理上限となります。
例えば、--count 10 を指定すると、各イメージに対してそれぞれ最大10件のDockerfileを処理します。

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
import os
import json
import sqlite3
from datetime import datetime, timedelta
from typing import List, Tuple, Optional, Dict, Any
from urllib.parse import urljoin, urlparse

class HttpCache:
    """HTTPリクエストのキャッシュを管理するクラス"""
    
    def __init__(self, cache_dir: str = '.cache'):
        """
        Args:
            cache_dir: キャッシュディレクトリのパス
        """
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.db_path = os.path.join(cache_dir, 'http_cache.db')
        self._init_db()
    
    def _init_db(self):
        """データベースの初期化"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS http_cache (
                    url TEXT PRIMARY KEY,
                    method TEXT,
                    headers TEXT,
                    response TEXT,
                    timestamp DATETIME
                )
            ''')
    
    def get(self, url: str, headers: Optional[Dict] = None) -> Optional[Dict]:
        """
        キャッシュからレスポンスを取得
        
        Args:
            url: リクエストURL
            headers: リクエストヘッダー
        
        Returns:
            キャッシュされたレスポンス（なければNone）
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                'SELECT response, timestamp FROM http_cache WHERE url = ? AND headers = ?',
                (url, json.dumps(headers or {}))
            )
            row = cursor.fetchone()
            
            if row:
                response_data, timestamp = row
                cached_time = datetime.fromisoformat(timestamp)
                # 7日以内のキャッシュのみ有効
                if datetime.now() - cached_time < timedelta(days=7):
                    return json.loads(response_data)
        return None
    
    def set(self, url: str, headers: Optional[Dict], response_data: Dict):
        """
        レスポンスをキャッシュに保存
        
        Args:
            url: リクエストURL
            headers: リクエストヘッダー
            response_data: レスポンスデータ
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                'INSERT OR REPLACE INTO http_cache (url, method, headers, response, timestamp) VALUES (?, ?, ?, ?, ?)',
                (
                    url,
                    'GET',
                    json.dumps(headers or {}),
                    json.dumps(response_data),
                    datetime.now().isoformat()
                )
            )

# グローバルなキャッシュインスタンス
http_cache = HttpCache()

def cached_request(url: str, headers: Optional[Dict] = None) -> Dict:
    """
    キャッシュを考慮したHTTPリクエスト
    
    Args:
        url: リクエストURL
        headers: リクエストヘッダー
    
    Returns:
        レスポンスデータ
    """
    # キャッシュをチェック
    cached_response = http_cache.get(url, headers)
    if cached_response:
        return cached_response
    
    # キャッシュがない場合は実際にリクエスト
    response = requests.get(url, headers=headers)
    
    if response.status_code == 403:
        rate_limit = response.headers.get('X-RateLimit-Remaining', '不明')
        reset_time = response.headers.get('X-RateLimit-Reset', '不明')
        raise Exception(f"GitHub APIのレート制限に達しました。\n"
                      f"残りリクエスト数: {rate_limit}\n"
                      f"リセット時刻: {reset_time}")
    elif response.status_code != 200:
        raise Exception(f"APIエラー: ステータスコード {response.status_code}\n"
                      f"レスポンス: {response.text}")
    
    # レスポンスをキャッシュ
    response_data = response.json()
    http_cache.set(url, headers, response_data)
    
    return response_data

def get_github_headers(check_rate_limit: bool = False) -> dict:
    """
    GitHub APIリクエスト用のヘッダーを生成し、必要に応じてAPIレート制限情報を表示
    環境変数GITHUB_TOKENが設定されている場合は認証ヘッダーを追加
    
    Args:
        check_rate_limit: レート制限をチェックするかどうか
    """
    headers = {
        "Accept": "application/vnd.github.v3+json"
    }
    
    token = os.getenv('GITHUB_TOKEN')
    if token:
        headers["Authorization"] = f"token {token}"
        # print("GitHub Personal Access Tokenが設定されています（5000リクエスト/時）")
    else:
        print("警告: GitHub Personal Access Tokenが設定されていません（60リクエスト/時）")
        print("環境変数GITHUB_TOKENを設定することで制限を緩和できます")
    
    if check_rate_limit:
        # レート制限の確認
        check_url = "https://api.github.com/rate_limit"
        try:
            data = cached_request(check_url, headers)
            remaining = data['resources']['search']['remaining']
            reset_timestamp = data['resources']['search']['reset']
            
            # UTCからローカル時間に変換
            reset_time = datetime.fromtimestamp(reset_timestamp).astimezone()
            
            print(f"APIレート制限情報:")
            print(f"- 残りのリクエスト数: {remaining}")
            print(f"- 制限リセット時刻: {reset_time.strftime('%Y-%m-%d %H:%M:%S')}")
            
            # 制限に近い場合は警告
            if remaining < 10:
                print(f"警告: APIリクエスト数が残り{remaining}回です")
                reset_wait = reset_timestamp - int(datetime.now().timestamp())
                if reset_wait > 0:
                    print(f"制限リセットまであと約{int(reset_wait/60)}分です")
        except Exception as e:
            print(f"APIレート制限情報の取得に失敗しました: {str(e)}")
    
    return headers

def search_dockerfiles(target_image: str, count: Optional[int] = None) -> List[dict]:
    """
    GitHubのAPIを使用して指定されたイメージを使用しているDockerfileを検索
    
    Args:
        target_image: 検索対象のベースイメージ
        count: 取得する最大件数（指定がない場合は1ページ分のみ取得）
    """
    headers = get_github_headers(check_rate_limit=False)
    all_items = []
    page = 1
    
    # 1ページあたり100件
    items_per_page = 100
    
    # 必要なページ数を計算（countが指定されている場合）
    max_pages = 1
    if count:
        max_pages = (count + items_per_page - 1) // items_per_page
    
    base_url = "https://api.github.com/search/code"
    query = f'FROM "{target_image}" filename:Dockerfile -alpine'
    
    while page <= max_pages:
        url = f"{base_url}?q={query}&per_page={items_per_page}&page={page}"
        try:
            data = cached_request(url, headers)
            
            if 'items' not in data:
                raise Exception(f"予期しないAPIレスポンス形式です: {data}")
                
            items = data['items']
            if not items:
                break
                
            all_items.extend(items)
            total_count = data.get('total_count', 0)
            
            # 進捗状況を表示
            print(f"イメージ {target_image} - ページ {page}/{max_pages} を処理中... ({len(all_items)}/{min(count or total_count, total_count)}件)")
            
            # 指定された件数に達した場合は終了
            if count and len(all_items) >= count:
                all_items = all_items[:count]  # 指定件数でカット
                break
                
            page += 1
            
        except Exception as e:
            raise Exception(f"GitHub APIへのリクエスト中にエラーが発生しました: {str(e)}")
    
    if not all_items:
        print(f"警告: {target_image} に一致するDockerfileが見つかりませんでした")
    
    return all_items

def get_dockerfile_content(url: str, headers: dict) -> str:
    """
    指定されたURLからDockerfileの内容を取得してデコード
    
    Args:
        url: DockerfileのURL
        headers: リクエストヘッダー
    """
    import base64
    
    try:
        data = cached_request(url, headers)
        
        if 'content' not in data:
            raise Exception(f"予期しないAPIレスポンス形式です: {data}")
            
        # GitHubのAPIはbase64エンコードされたコンテンツを返すのでデコード
        content = data['content']
        try:
            decoded_content = base64.b64decode(content).decode('utf-8')
            return decoded_content
        except Exception as e:
            raise Exception(f"コンテンツのデコードに失敗しました: {str(e)}")
            
    except Exception as e:
        raise Exception(f"Dockerfileの取得中にエラーが発生しました: {str(e)}")

def parse_dockerfile(content: str, target_image: str) -> Tuple[bool, str, List[str], List[str]]:
    """
    Dockerfileをパースして指定されたベースイメージの使用を確認し、ベースイメージ、中間コマンド、RUNコマンドを抽出
    
    Args:
        content: Dockerfileの内容
        target_image: 検索対象のベースイメージ
    
    Returns:
        Tuple[bool, str, List[str], List[str]]: (対象イメージを使用しているか, ベースイメージ, 中間コマンド, RUNコマンドのリスト)
    """
    lines = content.split('\n')
    run_commands = []
    intermediate_commands = []
    uses_target_image = False
    base_image = ""
    current_command = ""
    found_from = False
    found_run = False
    
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
            found_from = True
            import re
            # remove AS ****
            image_part = re.sub(r'(?i)\s+as\s+.*$', '', line[5:].strip()).strip()
            # remove --platform=linux/amd64 from `--platform=linux/amd64 ubuntu:22.04`
            image_part = re.sub(r'(?i)^\s*--platform=.*\s+', '', image_part).strip()
            # 指定されたイメージと一致するか確認
            if target_image in image_part and 'alpine' not in image_part:
                uses_target_image = True
                base_image = image_part
        
        # 中間コマンドを処理（FROM と最初のRUNの間）
        elif found_from and not found_run and line.startswith(('COPY', 'WORKDIR', 'ADD')):
            intermediate_commands.append(line.strip())
                
        # RUN命令を処理
        elif line.startswith('RUN'):
            found_run = True
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
                
    return uses_target_image, base_image, intermediate_commands, run_commands

def main():
    """
    メイン処理
    """
    import argparse

    # コマンドライン引数の設定
    parser = argparse.ArgumentParser(description='Dockerfile解析ツール')
    parser.add_argument('--all-output', required=True, help='全コマンドを出力するCSVファイルのパス')
    parser.add_argument('--first-output', required=True, help='最初のコマンドのみを出力するCSVファイルのパス')
    parser.add_argument('--count', type=int, help='処理するDockerfileの上限数')
    parser.add_argument('--image', action='append', required=True, help='検索対象のベースイメージ（複数指定可）')
    args = parser.parse_args()

    try:
        # 初回のみレート制限をチェック
        headers = get_github_headers(check_rate_limit=True)
        
        # 結果を保存するCSVファイルを開く
        with open(args.all_output, 'w', newline='', encoding='utf-8') as f_all, \
             open(args.first_output, 'w', newline='', encoding='utf-8') as f_first:
            writer_all = csv.writer(f_all, quoting=csv.QUOTE_MINIMAL, escapechar='\\')
            writer_first = csv.writer(f_first, quoting=csv.QUOTE_MINIMAL, escapechar='\\')
            
            # ヘッダー行を書き込む
            writer_all.writerow(['Base Image', 'Command', 'Intermediate Dockerfile Command', 'Dockerfile URL'])
            writer_first.writerow(['Base Image', 'Command', 'Intermediate Dockerfile Command', 'Dockerfile URL'])
            
            total_processed = 0
            total_success_all = 0
            total_success_first = 0
            total_errors = 0
            
            # 各イメージに対して処理を実行
            for target_image in args.image:
                print(f"\nイメージ {target_image} の検索を開始...")
                
                # 各イメージごとに指定された件数まで処理
                dockerfiles = search_dockerfiles(target_image, count=args.count)
                total_files = len(dockerfiles)
                print(f"検索結果: {total_files}件のDockerfileを処理します")
                
                processed = 0
                success_all = 0
                success_first = 0
                errors = 0
                
                for i, dockerfile in enumerate(dockerfiles, 1):
                    try:
                        raw_url = dockerfile['url']
                        repo_name = dockerfile['repository']['full_name']
                        
                        # URLからrefパラメータを抽出してコミットSHAを取得
                        parsed_url = urlparse(dockerfile['url'])
                        query_params = dict(pair.split('=') for pair in parsed_url.query.split('&') if pair)
                        commit_sha = query_params.get('ref', '')
                        
                        if not commit_sha:
                            raise Exception("コミットSHAの取得に失敗しました")
                        
                        # GitHubの正しいパーマリンク形式を使用
                        permalink = f"https://github.com/{repo_name}/blob/{commit_sha}/{dockerfile['path']}"

                        print(f"処理中 ({i}/{total_files}): {permalink}")
                        
                        # Dockerfileの内容を取得（レート制限チェックなし）
                        content = get_dockerfile_content(raw_url, get_github_headers(check_rate_limit=False))
                        
                        # Dockerfileをパース
                        uses_target_image, base_image, intermediate_commands, commands = parse_dockerfile(content, target_image)
                        
                        # 中間コマンドを文字列に結合（改行を\nで表現）
                        intermediate_str = '\n'.join(intermediate_commands) if intermediate_commands else ''
                        
                        # 指定されたイメージを使用している場合のみ結果を出力
                        if uses_target_image and commands:
                            # 全てのコマンドを出力
                            for command in commands:
                                writer_all.writerow([base_image, command, intermediate_str, permalink])
                                success_all += 1
                            
                            # 最初のコマンドのみを出力
                            writer_first.writerow([base_image, commands[0], intermediate_str, permalink])
                            success_first += 1
                        
                        processed += 1
                        
                    except Exception as e:
                        print(f"エラー: {str(e)}")
                        errors += 1
                        continue
                
                # イメージごとの集計を表示
                print(f"\nイメージ {target_image} の処理完了:")
                print(f"- 処理したDockerfile: {processed}/{total_files}")
                print(f"- 抽出した全コマンド数: {success_all}")
                print(f"- 抽出した最初のコマンド数: {success_first}")
                print(f"- エラー数: {errors}")
                
                # 全体の集計に加算
                total_processed += processed
                total_success_all += success_all
                total_success_first += success_first
                total_errors += errors
            
            # 全体の集計を表示
            print("\n全体の処理完了:")
            print(f"- 処理したDockerfile: {total_processed}")
            print(f"- 抽出した全コマンド数: {total_success_all}")
            print(f"- 抽出した最初のコマンド数: {total_success_first}")
            print(f"- エラー数: {total_errors}")
        
    except Exception as e:
        print(f"致命的なエラーが発生しました: {str(e)}")

if __name__ == "__main__":
    main()
