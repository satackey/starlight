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


def run_command(command, shell=False, timeout=900, allow_failure=False, quiet=False):
    """コマンドを実行し、出力を返す"""
    if quiet:
        print(f"\n>> Executing (quiet): {command}")
    else:
        print(f"\n>> Executing: {command}")

    global current_process
    try:
        stdout = subprocess.DEVNULL if quiet else None
        stderr = subprocess.DEVNULL if quiet else subprocess.STDOUT
        
        if shell:
            current_process = subprocess.Popen(
                command,
                shell=True,
                preexec_fn=os.setsid,  # 新しいプロセスグループを作る
                stdout=stdout,
                stderr=stderr
            )
        else:
            current_process = subprocess.Popen(
                command.split(),
                preexec_fn=os.setsid,
                stdout=stdout,
                stderr=stderr
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
    registry_repo = base_image.split(':')[0]
    image_name = base_image.replace(":", "-").replace("/", "-")
    owner, repo = extract_repo_info(dockerfile_url)
    if owner and repo:
        pass
    else:
        raise ValueError(f"Invalid Dockerfile URL: {dockerfile_url}")
    
    repo = repo.split('/')[0]  # リポジトリ名のみを取得（blob以降を除去）
    original_tag = base_image.split(':')[1] if ':' in base_image else 'latest'
    # lowercase owner and repo
    starlight_tag = f"{proxy_host}/{registry_repo.lower()}-{original_tag.lower()}/{owner.lower()}-{repo.lower()}:starlight"
    image_name = f"{registry_repo.lower()}-{original_tag.lower()}-{owner.lower()}-{repo.lower()}"
    print(f"\nProcessing image: {base_image}")
    
    # mirrored_base_image = f"cloud.cluster.local:5000/mirror/{base_image}"
    # convert_cmd = f"sudo ctr-starlight convert --insecure-destination --notify --profile myproxy --platform linux/amd64 {mirrored_base_image} {starlight_tag}"
    
    convert_cmd = f"sudo ctr-starlight convert --insecure-destination --notify --profile myproxy --platform linux/amd64 {base_image} {starlight_tag}"
    pull_cmd = f"sudo ctr-starlight pull --profile myproxy {starlight_tag}"
    
    try:
        # Step1～Step3: イメージが既にあれば "skipped", なければ実行
        if run_command(f"sudo ctr image ls | grep {starlight_tag}", allow_failure=True, shell=True, quiet=True) == 0:
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
            while run_command(f"sudo ctr container ls | grep {container_name}", allow_failure=True, shell=True, quiet=True) == 0:
                try:
                    run_command(f"sudo ctr container rm {container_name}", quiet=True)
                    time.sleep(1)
                except:
                    pass
            # すでに存在するスナップショットがあれば削除
            while run_command(f"sudo ctr snapshot --snapshotter=starlight ls | grep {container_name}", allow_failure=True, shell=True, quiet=True) == 0:
                try:
                    run_command(f"sudo ctr task kill -f {container_name}", quiet=True)
                    run_command(f"sudo ctr snapshot --snapshotter=starlight rm {container_name}", quiet=True)
                    time.sleep(1)
                except:
                    pass

            create_cmd = f"sudo ctr c create --snapshotter=starlight {starlight_tag} {container_name} sh -c {shlex.quote(first_command)}"
            run_command(create_cmd, shell=True)
            step_results["Step4 create"] = "success"
        except:
            step_results["Step4 create"] = "failure"
            raise

        # Step5 start
        started_at = time.time()
        try:
            start_cmd = f"sudo ctr task start {container_name}"
            run_command(start_cmd, shell=True, timeout=30)
            
            elapsed = time.time() - started_at
            elapsed_second_str = f"{elapsed:.1f} sec"

            step_results["Step5 start"] = f"success ({elapsed_second_str})"
        
        # expect timeout, non-zero exit code
        except subprocess.TimeoutExpired:
            print("Command timed out.")

            elapsed = time.time() - started_at
            elapsed_second_str = f"{elapsed:.1f} sec"

            step_results["Step5 start"] = f"timeout ({elapsed_second_str})"
        except subprocess.CalledProcessError as e:
            print(f"Command failed with exit code {e.returncode}.")

            elapsed = time.time() - started_at
            elapsed_second_str = f"{elapsed:.1f} sec"

            step_results["Step5 start"] = f"non-zero exit({e.returncode}) ({elapsed_second_str})"
        except:


            elapsed = time.time() - started_at
            elapsed_second_str = f"{elapsed:.1f} sec"

            step_results["Step5 start"] = f"failure ({elapsed_second_str})"
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
        error_class = e.__class__.__name__
        print(f"Error processing image {base_image}: `{error_class}` {str(e)}")
        # 失敗ステップを "failure" とし、それ以降は空欄のまま
        # すでに "skipped" or "success" のステップ以外は空欄のまま
        return False, step_results, starlight_tag
    
    return True, step_results, starlight_tag

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
    

    # すべての処理完了後、新しい CSV を書き出す
    output_file = f"result_{start_datetime}_{csv_file}_with_steps.csv"
    fieldnames = ["Base Image", "Command", "Intermediate Dockerfile Command", "Dockerfile URL", "Converted Repotag"] + STEP_COLUMNS
    with open(output_file, 'w', newline='') as out_f:
        writer = csv.DictWriter(out_f, fieldnames=fieldnames)
        writer.writeheader()

        # 各イメージを処理
        for base_image, first_command, dockerfile_url in images:
            try:
                time.sleep(1)
                success, step_results, starlight_tag = process_image(base_image, first_command, proxy_host, dockerfile_url)
                row_dict = {
                    "Base Image": base_image,
                    "Command": first_command,
                    "Intermediate Dockerfile Command": "",  # 必要に応じて代入
                    "Dockerfile URL": dockerfile_url,
                    "Converted Repotag": starlight_tag
                }
                row_dict.update(step_results)
                writer.writerows([row_dict])

                if success:
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
