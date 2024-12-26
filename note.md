build_and_trace_v3プロジェクトについて：

このプロジェクトは、コンテナイメージのpull処理を最適化するための機械学習モデル構築を目的としています。

1. 解決する問題：
- コンテナイメージのpull処理が遅い
- ファイルアクセスの順序が最適化されていない
- ビルド時のリソース利用が非効率

2. アプローチ：
- Dockerfileのビルドプロセスを追跡
- ファイルアクセスパターンを収集・分析
- 機械学習モデルのための学習データを生成
- 収集したデータを元に最適なpull順序を予測

3. 期待される効果：
- pull処理の高速化
- リソース利用の効率化
- ビルド時間の短縮

このプロジェクトは、実際のDockerfileビルドから得られるトレースデータを使用して、機械学習モデルを訓練し、最適なファイルアクセス順序を学習することで、コンテナイメージのpull処理を最適化することを目指しています。

現在の実装状況：
1. プロジェクト構造
- executors/: コマンド実行管理（base.py, buildctl.py, optimizer.py）
- processors/: ビジネスロジック（dockerfile.py, github.py）
- utils/: ユーティリティ（logging.py, async_helpers.py）
- tests/: テストコード（各コンポーネントのテスト）
- main.py: エントリーポイント
- requirements.txt, pytest.ini: 設定ファイル

2. 主な機能
- Dockerfileのビルドプロセス追跡
- ファイルアクセスパターンの収集
- 機械学習用データの生成（PostgreSQLに保存）

3. 次のステップ
- テストの実行と検証
- コードフォーマットの適用
- 実際のDockerfileでの動作確認
- 機械学習モデルの実装

4. 環境要件
- Python 3.9+
- buildctl
- ctr-starlight
- PostgreSQL
- GitHub Personal Access Token

現在のディレクトリ: /home/vagrant/starlight
作業ディレクトリ: scripts/build_and_trace_v3/
