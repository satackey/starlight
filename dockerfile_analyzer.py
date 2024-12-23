import requests
import csv
import re
from typing import List, Tuple
from urllib.parse import urljoin

def search_dockerfiles() -> List[dict]:
    """
    GitHubのAPIを使用してnode:ltsを使用しているDockerfileを検索
    """
    headers = {
        "Accept": "application/vnd.github.v3+json",
        # 必要に応じてGitHub Personal Access Tokenを追加
    }
    
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
                content = get_dockerfile_content(raw_url, {})
                
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