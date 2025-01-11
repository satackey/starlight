import csv
import subprocess
import time
import os
import sys
import signal
import shlex

current_process = None

def _handle_sigint(signum, frame):
    global current_process
    if current_process and current_process.poll() is None:
        try:
            pgid = os.getpgid(current_process.pid)
            os.killpg(pgid, signal.SIGTERM)
        except:
            pass
    print("Interrupted by user (Ctrl+C). Exiting.")
    sys.exit(1)

signal.signal(signal.SIGINT, _handle_sigint)


def run_command(command, shell=False, timeout=900, allow_failure=False):
    """コマンドを実行し、出力を返す"""
    print(f"\n>> Executing: {command}")
    global current_process
    try:
        if shell:
            current_process = subprocess.Popen(
                command,
                shell=True,
                preexec_fn=os.setsid  # 新しいプロセスグループを作る
            )
        else:
            current_process = subprocess.Popen(
                command.split(),
                preexec_fn=os.setsid
            )
        return_code = current_process.wait(timeout=timeout)
        
        if return_code != 0 and not allow_failure:
            raise subprocess.CalledProcessError(return_code, command)
        
        return return_code
    except subprocess.TimeoutExpired:
        print(f"Command timed out after {timeout} seconds: {command}")
        if current_process:
            try:
                pgid = os.getpgid(current_process.pid)
                os.killpg(pgid, signal.SIGKILL)
            except:
                pass
        raise
    except subprocess.CalledProcessError as e:
        print(f"Command failed with exit code {e.returncode}: {command}")
        raise

def extract_repo_info(dockerfile_url: str) -> tuple:
    """GitHubのURLからオーナーとリポジトリ名を抽出する"""
    # URLのパターン: https://github.com/owner/repo/blob/...
    parts = dockerfile_url.split('/')
    if len(parts) >= 5 and parts[2] == 'github.com':
        return parts[3], parts[4]
    return None, None

def process_image(base_image, first_command, proxy_host, dockerfile_url):
    """単一のイメージを処理し、トレースを収集する"""
    image_name = base_image.replace(":", "-").replace("/", "-")
    owner, repo = extract_repo_info(dockerfile_url)
    if owner and repo:
        repo = repo.split('/')[0]  # リポジトリ名のみを取得（blob以降を除去）
        original_tag = base_image.split(':')[1] if ':' in base_image else 'latest'
        starlight_tag = f"{proxy_host}/{image_name}:{original_tag}-starlight-{owner}-{repo}"
    else:
        starlight_tag = f"{proxy_host}/{image_name}:starlight"
    
    print(f"\nProcessing image: {base_image}")
    
    try:
        # もし starlight_tag イメージが存在する場合は、すでに処理済みとみなしてスキップ
        if run_command(f"sudo ctr image ls | grep {starlight_tag}", allow_failure=True, shell=True) != 0:
            # 1. イメージをStarlight形式に変換
            print("Converting image to Starlight format...")
            convert_cmd = f"sudo ctr-starlight convert --insecure-destination --notify --profile myproxy --platform linux/amd64 {base_image} {starlight_tag}"
            run_command(convert_cmd, shell=True)
            
            # 2. オプティマイザーを有効化
            print("Enabling optimizer...")
            run_command("sudo ctr-starlight optimizer on")
            
            # 3. イメージをプル
            print("Pulling image...")
            pull_cmd = f"sudo ctr-starlight pull --profile myproxy {starlight_tag}"
            run_command(pull_cmd, shell=True)
        
        # 4. コンテナを作成して実行
        print("Creating and running container...")
        container_name = f"trace-{image_name}-{starlight_tag.split(':')[-1]}"

        # もし container_name がすでに存在する場合は削除
        while run_command(f"sudo ctr task ps {container_name}", allow_failure=True) == 0:
            try:
                run_command(f"sudo ctr container rm -f {container_name}")
            except:
                pass
            
        # コンテナ作成
        create_cmd = f"sudo ctr c create --snapshotter=starlight {starlight_tag} {container_name} sh -c {shlex.quote(first_command)}"
        run_command(create_cmd, shell=True)
        
        # コンテナ起動
        start_cmd = f"sudo ctr task start {container_name}"
        run_command(start_cmd, shell=True, timeout=30)
        
        # コンテナを停止
        print("Stopping container...")
        try:
            run_command(f"sudo ctr task kill {container_name}")
        except:
            pass
        
        # コンテナを削除
        print("Removing container...")
        try:
            run_command(f"sudo ctr container rm {container_name}")
        except:
            pass

        # オプティマイザーを無効化
        print("\nDisabling optimizer...")
        run_command("sudo ctr-starlight optimizer off")
        
        # トレースをレポート
        print("Reporting traces...")
        run_command("sudo ctr-starlight report --profile myproxy")
        
    except Exception as e:
        print(f"Error processing image {base_image}: {str(e)}")
        return False
    
    return True

def main():
    if len(sys.argv) != 3:
        print("Usage: collect_traces.py <csv_file> <proxy_host>")
        sys.exit(1)
        
    csv_file = sys.argv[1]
    proxy_host = sys.argv[2]
    
    # CSVファイルを読み込む
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        images = [(row['Base Image'], row['Command'], row['Dockerfile URL']) for row in reader]
    
    successful = 0
    failed = 0
    
    # 各イメージを処理
    for base_image, first_command, dockerfile_url in images:
        try:
            if process_image(base_image, first_command, proxy_host, dockerfile_url):
                successful += 1
            else:
                failed += 1
        except Exception as e:
            print(f"Failed to process {base_image}: {str(e)}")
            failed += 1
    
    # オプティマイザーを無効化
    print("\nDisabling optimizer...")
    run_command("sudo ctr-starlight optimizer off")
    
    # トレースをレポート
    print("Reporting traces...")
    run_command("sudo ctr-starlight report --profile myproxy")
    
    print(f"\nProcessing complete:")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")

if __name__ == "__main__":
    main()
