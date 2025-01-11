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

STEP_COLUMNS = [
    "Step1 convert",
    "Step2 optimizer on",
    "Step3 pull",
    "Step4 create",
    "Step5 start",
    "Step6 remove",
    "Step7 optimizer off",
    "Step8 report"
]

def process_image(base_image, first_command, proxy_host, dockerfile_url):
    step_results = {col: "" for col in STEP_COLUMNS}
    image_name = base_image.replace(":", "-").replace("/", "-")
    owner, repo = extract_repo_info(dockerfile_url)
    if owner and repo:
        repo = repo.split('/')[0]  # リポジトリ名のみを取得（blob以降を除去）
        original_tag = base_image.split(':')[1] if ':' in base_image else 'latest'
        starlight_tag = f"{proxy_host}/{image_name}:{original_tag}-starlight-{owner}-{repo}"
    else:
        starlight_tag = f"{proxy_host}/{image_name}:starlight"
    
    print(f"\nProcessing image: {base_image}")
    
    convert_cmd = f"sudo ctr-starlight convert --insecure-destination --notify --profile myproxy --platform linux/amd64 {base_image} {starlight_tag}"
    pull_cmd = f"sudo ctr-starlight pull --profile myproxy {starlight_tag}"
    
    try:
        # Step1～Step3: イメージが既にあれば "skipped", なければ実行
        if run_command(f"sudo ctr image ls | grep {starlight_tag}", allow_failure=True, shell=True) == 0:
            step_results["Step1 convert"] = "skipped"
            step_results["Step2 optimizer on"] = "skipped"
            step_results["Step3 pull"] = "skipped"
        else:
            # Step1 convert
            try:
                run_command(convert_cmd, shell=True)
                step_results["Step1 convert"] = "success"
            except:
                step_results["Step1 convert"] = "failure"
                raise

            # Step2 optimizer on
            try:
                run_command("sudo ctr-starlight optimizer on")
                step_results["Step2 optimizer on"] = "success"
            except:
                step_results["Step2 optimizer on"] = "failure"
                raise

            # Step3 pull
            try:
                run_command(pull_cmd, shell=True)
                step_results["Step3 pull"] = "success"
            except:
                step_results["Step3 pull"] = "failure"
                raise
        
        # Step4 create
        try:
            print("Creating and running container...")
            container_name = f"trace-{image_name}-{starlight_tag.split(':')[-1]}"
            # 既に存在するコンテナがいれば削除
            while run_command(f"sudo ctr task ps {container_name}", allow_failure=True) == 0:
                try:
                    run_command(f"sudo ctr container rm -f {container_name}")
                except:
                    pass
            create_cmd = f"sudo ctr c create --snapshotter=starlight {starlight_tag} {container_name} sh -c {shlex.quote(first_command)}"
            run_command(create_cmd, shell=True)
            step_results["Step4 create"] = "success"
        except:
            step_results["Step4 create"] = "failure"
            raise

        # Step5 start
        try:
            start_cmd = f"sudo ctr task start {container_name}"
            run_command(start_cmd, shell=True, timeout=30)
            step_results["Step5 start"] = "success"
        except:
            step_results["Step5 start"] = "failure"
            raise

        # Step6 remove
        try:
            print("Stopping container...")
            try:
                run_command(f"sudo ctr task kill {container_name}")
            except:
                pass
            print("Removing container...")
            run_command(f"sudo ctr container rm {container_name}", shell=True, allow_failure=True)
            step_results["Step6 remove"] = "success"
        except:
            step_results["Step6 remove"] = "failure"
            raise

        # Step7 optimizer off
        try:
            print("\nDisabling optimizer...")
            run_command("sudo ctr-starlight optimizer off")
            step_results["Step7 optimizer off"] = "success"
        except:
            step_results["Step7 optimizer off"] = "failure"
            raise

        # Step8 report
        try:
            print("Reporting traces...")
            run_command("sudo ctr-starlight report --profile myproxy")
            step_results["Step8 report"] = "success"
        except:
            step_results["Step8 report"] = "failure"
            raise
        
    except Exception as e:
        print(f"Error processing image {base_image}: {str(e)}")
        # 失敗ステップを "failure" とし、それ以降は空欄のまま
        # すでに "skipped" or "success" のステップ以外は空欄のまま
        return False, step_results
    
    return True, step_results

def main():
    if len(sys.argv) != 3:
        print("Usage: collect_traces.py <csv_file> <proxy_host>")
        sys.exit(1)
        
    csv_file = sys.argv[1]
    proxy_host = sys.argv[2]

    start_datetime = time.strftime("%Y-%m-%d_%H%M%S")
    
    # CSVファイルを読み込む
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        images = [(row['Base Image'], row['Command'], row['Dockerfile URL']) for row in reader]
    
    successful = 0
    failed = 0
    results_for_csv = []
    
    # 各イメージを処理
    for base_image, first_command, dockerfile_url in images:
        try:
            success, step_results = process_image(base_image, first_command, proxy_host, dockerfile_url)
            row_dict = {
                "Base Image": base_image,
                "Command": first_command,
                "Intermediate Dockerfile Command": "",  # 必要に応じて代入
                "Dockerfile URL": dockerfile_url
            }
            row_dict.update(step_results)
            results_for_csv.append(row_dict)
            if success:
                successful += 1
            else:
                failed += 1
        except Exception as e:
            print(f"Failed to process {base_image}: {str(e)}")
            failed += 1
    
    
    # すべての処理完了後、新しい CSV を書き出す
    output_file = f"result_{start_datetime}_{csv_file}_with_steps.csv"
    fieldnames = ["Base Image", "Command", "Intermediate Dockerfile Command", "Dockerfile URL"] + STEP_COLUMNS
    with open(output_file, 'w', newline='') as out_f:
        writer = csv.DictWriter(out_f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results_for_csv)

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
