#!/usr/bin/env python3
"""
Dockerfile Build and Trace Collector

このスクリプトは以下の機能を提供します：
1. CSVファイルからDockerfile URLを読み込み
2. GitHubリポジトリをクローンし、特定のSHAでチェックアウト
3. optimizerを使用してビルドトレースを収集
4. buildctlを使用してDockerfileをビルド

使用方法:
$ python build_and_trace.py dockerfile_commands_10_first.csv

必要なパッケージ:
- gitpython
"""

import csv
import os
import re
import subprocess
import tempfile
from pathlib import Path
from urllib.parse import urlparse
import git
import time
import logging

# ロギングの設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_github_url(url):
    """
    GitHub URLからリポジトリ情報を抽出

    Args:
        url: GitHub URLの文字列
        
    Returns:
        (user, repo, sha, filepath)のタプル
    """
    # URLからパスを抽出
    parsed = urlparse(url)
    path_parts = parsed.path.split('/')
    
    # GitHub URLのフォーマット: /user/repo/blob/sha/path/to/Dockerfile
    user = path_parts[1]
    repo = path_parts[2]
    sha = path_parts[4]
    filepath = '/'.join(path_parts[5:])
    
    return user, repo, sha, filepath

def sanitize_group_name(name):
    """
    グループ名から特殊文字を削除

    Args:
        name: 元の文字列
        
    Returns:
        特殊文字を削除した文字列
    """
    return re.sub(r'[^a-zA-Z0-9-]', '', name)

def clone_repo(user, repo, sha, work_dir):
    """
    リポジトリをクローンし、特定のSHAでチェックアウト

    Args:
        user: GitHubユーザー名
        repo: リポジトリ名
        sha: コミットSHA
        work_dir: 作業ディレクトリ
        
    Returns:
        クローンしたリポジトリのパス
    """
    repo_url = f"https://github.com/{user}/{repo}.git"
    repo_path = os.path.join(work_dir, repo)
    
    logger.info(f"Cloning repository: {repo_url}")
    git.Repo.clone_from(repo_url, repo_path)
    
    # 特定のSHAをチェックアウト
    repo_obj = git.Repo(repo_path)
    repo_obj.git.checkout(sha)
    
    return repo_path

def run_optimizer_command(action, group_name):
    """
    optimizerコマンドを実行

    Args:
        action: "on" または "off"
        group_name: 最適化グループ名
    """
    cmd = ["sudo", "ctr-starlight", "optimizer", action]
    if action == "on":
        cmd.extend(["--group", group_name])
    
    logger.info(f"Running optimizer {action} with group: {group_name}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error(f"Optimizer {action} failed: {result.stderr}")
        raise Exception(f"Optimizer {action} failed")

def stream_output(pipe, log_func, stop_event):
    """
    パイプからの出力を非同期で処理

    Args:
        pipe: subprocess.PIPEオブジェクト
        log_func: ログ出力関数
        stop_event: 停止フラグ
    """
    try:
        for line in iter(pipe.readline, ''):
            if stop_event.is_set():
                break
            log_func(line.strip())
    except Exception as e:
        logger.error(f"Error reading pipe: {str(e)}")

def build_dockerfile(dockerfile_path, timeout=3600):
    """
    buildctlを使用してDockerfileをビルド

    Args:
        dockerfile_path: Dockerfileのパス
        timeout: ビルドのタイムアウト時間（秒）
    """
    import threading
    from queue import Queue
    
    dockerfile_dir = os.path.dirname(dockerfile_path)
    cmd = [
        "sudo", "buildctl", "build",
        "--frontend=dockerfile.v0",
        "--frontend-opt", f"filename={os.path.basename(dockerfile_path)}",
        f"--local=context={dockerfile_dir}",
        f"--local=dockerfile={dockerfile_dir}",
        "--output", "type=image,name=build-output,push=false",
        "--trace", "mode=1",
        "--progress=plain"
    ]
    
    logger.info(f"Building Dockerfile: {dockerfile_path}")
    logger.debug(f"Build command: {' '.join(cmd)}")
    
    # エラーメッセージを保存するキュー
    error_queue = Queue()
    
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
        bufsize=1
    )
    
    # 出力ストリームの読み取りを停止するためのイベント
    stop_event = threading.Event()
    
    # 出力を別スレッドで処理
    stdout_thread = threading.Thread(
        target=stream_output,
        args=(process.stdout, lambda x: logger.info(f"Build output: {x}"), stop_event)
    )
    stderr_thread = threading.Thread(
        target=stream_output,
        args=(process.stderr, lambda x: error_queue.put(x), stop_event)
    )
    
    stdout_thread.daemon = True
    stderr_thread.daemon = True
    stdout_thread.start()
    stderr_thread.start()
    
    try:
        # タイムアウト付きでプロセスの完了を待つ
        process.wait(timeout=timeout)
        
        # エラーメッセージを収集
        error_messages = []
        while not error_queue.empty():
            error_messages.append(error_queue.get_nowait())
        
        if process.returncode != 0:
            error_detail = "\n".join(error_messages)
            raise Exception(f"Build failed with return code {process.returncode}:\n{error_detail}")
            
    except subprocess.TimeoutExpired:
        process.kill()
        raise Exception(f"Build timed out after {timeout} seconds")
    finally:
        # 出力ストリームの読み取りを停止
        stop_event.set()
        stdout_thread.join(timeout=1)
        stderr_thread.join(timeout=1)
        
        # プロセスのクリーンアップ
        try:
            process.stdout.close()
            process.stderr.close()
        except:
            pass

def report_traces():
    """
    トレースを報告
    """
    cmd = ["sudo", "ctr-starlight", "report"]
    logger.info("Reporting traces")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error(f"Report failed: {result.stderr}")
        raise Exception("Report failed")

def process_dockerfile(url):
    """
    単一のDockerfileを処理

    Args:
        url: Dockerfile のGitHub URL
    """
    # GitHub URLをパース
    user, repo, sha, filepath = parse_github_url(url)
    
    # グループ名を生成
    basename = os.path.basename(os.path.dirname(filepath))
    group_name = f"mloptimizer-{sanitize_group_name(user)}-{sanitize_group_name(repo)}-{sha[:7]}-{sanitize_group_name(basename)}"
    
    with tempfile.TemporaryDirectory() as work_dir:
        try:
            # リポジトリをクローン
            repo_path = clone_repo(user, repo, sha, work_dir)
            dockerfile_path = os.path.join(repo_path, filepath)
            
            if not os.path.exists(dockerfile_path):
                logger.error(f"Dockerfile not found: {dockerfile_path}")
                return
            
            # optimizerをオン
            run_optimizer_command("on", group_name)
            
            try:
                # Dockerfileをビルド
                build_dockerfile(dockerfile_path)
            finally:
                # 必ずoptimizerをオフにする
                run_optimizer_command("off", group_name)
                
            # トレースを報告
            report_traces()
            
        except Exception as e:
            logger.error(f"Error processing {url}: {str(e)}")

def main(csv_file):
    """
    メイン処理

    Args:
        csv_file: 入力CSVファイルのパス
    """
    logger.info(f"Processing CSV file: {csv_file}")
    
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            url = row['Dockerfile URL']
            logger.info(f"Processing URL: {url}")
            process_dockerfile(url)
            # レート制限を避けるために少し待機
            time.sleep(2)

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python build_and_trace.py <csv_file>")
        sys.exit(1)
    
    main(sys.argv[1])
