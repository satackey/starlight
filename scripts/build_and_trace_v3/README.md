# Build and Trace V3

Dockerfileのビルドプロセスを追跡し、ファイルアクセスパターンを収集・分析するためのツール。
機械学習を用いたコンテナイメージのpull最適化のための学習データを生成します。

## 機能

- **Dockerfile解析**
  - GitHubからのDockerfile取得
  - ビルドプロセスの追跡
  - ファイルアクセスパターンの収集

- **最適化機能**
  - トレースの収集と管理
  - グループ化されたデータ管理
  - PostgreSQLへのデータ保存

- **並行処理**
  - 複数Dockerfileの同時処理
  - リソース制限の管理
  - エラーハンドリング

## インストール

1. 依存パッケージのインストール：
```bash
pip install -r requirements.txt
```

2. 必要なツールの確認：
- buildctl
- ctr-starlight
- git

3. 環境変数の設定：
```bash
export GITHUB_TOKEN=<your-token>  # GitHubのアクセストークン
export BUILDCTL_PATH=/usr/bin/buildctl  # buildctlのパス
export CTR_STARLIGHT_PATH=/usr/bin/ctr-starlight  # ctr-starlightのパス
```

## 使用方法

1. CSVファイルの準備：
```csv
Dockerfile URL
https://github.com/user/repo/blob/main/Dockerfile
https://github.com/user/repo/blob/main/services/api/Dockerfile
```

2. スクリプトの実行：
```bash
python -m build_and_trace_v3 <csv_file> [options]
```

オプション：
- `--concurrency N`: 同時処理数（デフォルト: 1）
- `--log-dir PATH`: ログ出力ディレクトリ
- `--log-level LEVEL`: ログレベル（DEBUG/INFO/WARNING/ERROR/CRITICAL）

3. optimizerの制御：
```bash
# optimizerの開始
sudo ctr-starlight --group <group_name> optimizer on

# optimizerの停止
sudo ctr-starlight --group <group_name> optimizer off
```

## 開発

1. テストの実行：
```bash
python -m pytest
```

2. カバレッジレポートの確認：
```bash
python -m pytest --cov-report=html
open htmlcov/index.html
```

3. コードフォーマット：
```bash
black .
isort .
```

## プロジェクト構造

```
build_and_trace_v3/
├── __init__.py
├── main.py                   # エントリーポイント
├── executors/               # コマンド実行関連
│   ├── base.py             # 基本実行機能
│   ├── buildctl.py         # buildctl実行
│   └── optimizer.py        # optimizer実行
├── processors/             # 処理ロジック
│   ├── dockerfile.py       # Dockerfile処理
│   └── github.py          # GitHub操作
├── utils/                 # ユーティリティ
│   ├── logging.py        # ログ設定
│   └── async_helpers.py  # 非同期ヘルパー
└── tests/                # テストコード
    ├── conftest.py      # テスト共通設定
    ├── test_main.py     # メインのテスト
    ├── executors/       # executorsのテスト
    ├── processors/      # processorsのテスト
    └── utils/           # utilsのテスト
```

## データ形式

トレースデータは以下の形式で管理されます：

```python
{
    "group": "mloptimizer-{user}-{repo}-{sha}-{basename}",
    "image": "image_digest",
    "start_time": "ISO8601 timestamp",
    "end_time": "ISO8601 timestamp",
    "traces": [
        {
            "filename": "path/to/file",
            "access_time": "duration from start",
            "wait_time": "duration"
        }
        ...
    ]
}
```

このデータはPostgreSQLのfileテーブルに保存され、"order"カラムにアクセス順序が記録されます。

## エラーハンドリング

- **ファイル関連**
  - ファイルが見つからない場合
  - 無効なフォーマット
  - アクセス権限の問題

- **GitHub関連**
  - レート制限
  - 認証エラー
  - リポジトリアクセスエラー

- **ビルド関連**
  - ビルドエラー
  - タイムアウト
  - リソース制限

## 貢献

1. このリポジトリをフォーク
2. 機能ブランチを作成 (`git checkout -b feature/amazing-feature`)
3. 変更をコミット (`git commit -m 'Add some amazing feature'`)
4. ブランチをプッシュ (`git push origin feature/amazing-feature`)
5. プルリクエストを作成

## ライセンス

このプロジェクトはApache License 2.0の下で公開されています。
