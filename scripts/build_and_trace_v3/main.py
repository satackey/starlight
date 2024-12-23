"""
Build and Trace Main

メインエントリーポイント。
以下の機能を実装：
- コマンドライン引数の処理
- CSVファイルの読み込み
- Dockerfileの処理実行
"""

import argparse
import asyncio
import csv
import os
import sys
from pathlib import Path
from typing import List, Optional

from processors.dockerfile import BatchDockerfileProcessor
from utils.logging import setup_logging, get_logger

logger = get_logger()

def parse_args() -> argparse.Namespace:
    """
    コマンドライン引数をパース
    
    Returns:
        パース済みの引数
    """
    parser = argparse.ArgumentParser(
        description='Dockerfile Build and Trace Collector',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        'csv_file',
        type=str,
        help='処理対象のDockerfile URLを含むCSVファイル'
    )
    
    parser.add_argument(
        '--concurrency',
        type=int,
        default=1,
        help='同時処理数（デフォルト: 1）'
    )
    
    parser.add_argument(
        '--log-dir',
        type=str,
        default=None,
        help='ログファイルの出力ディレクトリ'
    )
    
    parser.add_argument(
        '--log-level',
        type=str,
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        default='INFO',
        help='ログレベル（デフォルト: INFO）'
    )
    
    return parser.parse_args()

def read_urls_from_csv(csv_file: str) -> List[str]:
    """
    CSVファイルからDockerfile URLを読み込み
    
    Args:
        csv_file: CSVファイルのパス
        
    Returns:
        Dockerfile URLのリスト
        
    Raises:
        FileNotFoundError: ファイルが見つからない場合
        ValueError: CSVフォーマットが不正な場合
    """
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"CSVファイルが見つかりません: {csv_file}")
    
    urls = []
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        
        # 必要なカラムの存在確認
        if 'Dockerfile URL' not in reader.fieldnames:
            raise ValueError("CSVファイルに'Dockerfile URL'カラムが必要です")
        
        for row in reader:
            url = row['Dockerfile URL'].strip()
            if url:
                urls.append(url)
    
    return urls

async def process_dockerfiles(
    urls: List[str],
    concurrency: int,
    log_dir: Optional[str] = None
) -> None:
    """
    Dockerfileを処理
    
    Args:
        urls: Dockerfile URLのリスト
        concurrency: 同時処理数
        log_dir: ログ出力ディレクトリ
    """
    processor = BatchDockerfileProcessor(concurrency=concurrency)
    
    try:
        await processor.process_dockerfiles(urls)
    except Exception as e:
        logger.error(f"Dockerfile処理中にエラーが発生しました: {str(e)}", exc_info=True)
        raise

async def main() -> None:
    """メイン処理"""
    args = parse_args()
    
    # ログ設定
    setup_logging(
        log_dir=args.log_dir,
        log_level=args.log_level
    )
    
    try:
        # URLの読み込み
        logger.info(f"CSVファイルを読み込み中: {args.csv_file}")
        urls = read_urls_from_csv(args.csv_file)
        logger.info(f"{len(urls)}件のDockerfile URLを読み込みました")
        
        if not urls:
            logger.warning("処理対象のURLがありません")
            return
        
        # Dockerfileの処理
        logger.info(f"Dockerfileの処理を開始（同時処理数: {args.concurrency}）")
        await process_dockerfiles(
            urls=urls,
            concurrency=args.concurrency,
            log_dir=args.log_dir
        )
        
        logger.info("全ての処理が完了しました")
        
    except Exception as e:
        logger.error(f"エラーが発生しました: {str(e)}", exc_info=True)
        sys.exit(1)

if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("処理を中断しました")
        sys.exit(0)
    except Exception as e:
        logger.error(f"予期せぬエラーが発生しました: {str(e)}", exc_info=True)
        sys.exit(1)
