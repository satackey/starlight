```shell
source .venv/bin/activate

# csv を作成
GITHUB_TOKEN=$(gh auth token) python dockerfile_analyzer.py --all-output dockerfile_commands_10.csv --first-output dockerfile_commands_10_first.csv --count 10 --image node:lts --image ubuntu:22.04

# mirror にコピー
python -u mirror_images.py dockerfile_commands_10_first.csv 

# trace データを収集
sudo systemctl start containerd starlight
python -u collect_traces.py dockerfile_commands_10_first.csv cloud.cluster.local:5000 2>&1 | tee result_$(date "+%Y-%m-%d_%H%M%S").log

# 収集した trace データは postgres に保存されるので、csv に変換
POSTGRES_CONNECTION_STRING="postgresql://postgres:postgres@cloud.cluster.local:5432/postgres" python convert_traces_to_csv.py result_2025-01-12_075226_dockerfile_commands_200_first.csv_with_steps.csv result_2025-01-12_075226_processed_traces_v2.csv

# モデルの学習・評価
python train_gru_model_v3.py result_2025-01-12_075226_processed_traces.csv 2>&1 | tee train_gru_model_v3_$(date "+%Y-%m-%d_%H%M%S").log
```
