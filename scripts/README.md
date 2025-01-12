```shell
source .venv/bin/activate

# csv を作成
GITHUB_TOKEN=$(gh auth token) python dockerfile_analyzer.py --all-output dockerfile_commands_10.csv --first-output dockerfile_commands_10_first.csv --count 10 --image node:lts --image ubuntu:22.04

# mirror にコピー
python -u mirror_images.py dockerfile_commands_10_first.csv 

# trace データを収集
sudo systemctl start containerd starlight
python -u collect_traces.py dockerfile_commands_10_first.csv cloud.cluster.local:5000 2>&1 | tee result_$(date "+%Y-%m-%d_%H%M%S").log
```
