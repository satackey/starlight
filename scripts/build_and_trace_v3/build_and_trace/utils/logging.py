"""
Logging Utilities

ログ管理のためのユーティリティモジュール。
以下の機能を実装：
- ファイルとコンソールへの出力
- カラー出力
- 構造化ログ
- コンテキスト情報の管理
"""

import json
import logging
import os
import sys
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional, Dict, Any, Union

# ANSIカラーコード
COLORS = {
    'DEBUG': '\033[36m',     # シアン
    'INFO': '\033[32m',      # 緑
    'WARNING': '\033[33m',   # 黄
    'ERROR': '\033[31m',     # 赤
    'CRITICAL': '\033[35m',  # マゼンタ
    'RESET': '\033[0m'       # リセット
}

class ColorFormatter(logging.Formatter):
    """カラー出力対応のフォーマッター"""
    
    def __init__(self, fmt: Optional[str] = None, datefmt: Optional[str] = None):
        super().__init__(fmt, datefmt)
    
    def format(self, record: logging.LogRecord) -> str:
        """
        ログレコードをフォーマット
        
        Args:
            record: ログレコード
            
        Returns:
            フォーマットされたログメッセージ
        """
        # 元のメッセージを保存
        original_msg = record.msg
        
        # 辞書型の場合はJSONに変換
        if isinstance(record.msg, dict):
            record.msg = json.dumps(record.msg, ensure_ascii=False, indent=2)
        
        # カラーコードを追加
        levelname = record.levelname
        if sys.stderr.isatty():  # ターミナルの場合のみカラー出力
            color = COLORS.get(levelname, COLORS['RESET'])
            record.levelname = f"{color}{levelname}{COLORS['RESET']}"
        
        # 基本のフォーマットを適用
        formatted = super().format(record)
        
        # 元のメッセージを復元
        record.msg = original_msg
        record.levelname = levelname
        
        return formatted

class ContextLogger(logging.Logger):
    """コンテキスト情報を含むロガー"""
    
    def __init__(self, name: str, level: Union[int, str] = logging.NOTSET):
        super().__init__(name, level)
        self.context: Dict[str, Any] = {}
    
    def bind(self, **kwargs) -> 'ContextLogger':
        """
        コンテキスト情報を追加
        
        Args:
            **kwargs: キーワード引数でコンテキスト情報を指定
            
        Returns:
            self: メソッドチェーン用
        """
        self.context.update(kwargs)
        return self
    
    def _log(self, level: int, msg: Any, args: tuple, exc_info: Optional[Exception] = None, extra: Optional[dict] = None, **kwargs):
        """
        コンテキスト情報を含めてログを出力
        
        Args:
            level: ログレベル
            msg: ログメッセージ
            args: フォーマット引数
            exc_info: 例外情報
            extra: 追加情報
            **kwargs: その他のキーワード引数
        """
        if isinstance(msg, dict):
            # 辞書型の場合はコンテキストとマージ
            msg = {**self.context, **msg}
        elif isinstance(msg, str):
            # 文字列の場合はコンテキストを追加
            if self.context:
                msg = f"{msg} {json.dumps(self.context)}"
        
        super()._log(level, msg, args, exc_info, extra, **kwargs)

def setup_logging(
    log_dir: Optional[str] = None,
    log_level: Union[int, str] = logging.INFO,
    max_bytes: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5
) -> ContextLogger:
    """
    ロギングの設定
    
    Args:
        log_dir: ログファイルのディレクトリ
        log_level: ログレベル
        max_bytes: ログファイルの最大サイズ
        backup_count: 保持するバックアップファイル数
        
    Returns:
        設定されたロガー
    """
    # ロガーを作成
    logging.setLoggerClass(ContextLogger)
    logger = logging.getLogger('build_and_trace')
    logger.setLevel(log_level)
    
    # 既存のハンドラをクリア
    logger.handlers.clear()
    
    # フォーマッターを作成
    formatter = ColorFormatter(
        fmt='%(asctime)s [%(levelname)s] %(processName)s:%(process)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # コンソール出力用のハンドラを追加
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # ファイル出力用のハンドラを追加（指定がある場合）
    if log_dir:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = log_dir / f'build_trace_{timestamp}.log'
        
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

def get_logger() -> ContextLogger:
    """
    ロガーを取得
    
    Returns:
        設定済みのロガー
    """
    return logging.getLogger('build_and_trace')

# デフォルトのロガーを設定
logger = setup_logging()
