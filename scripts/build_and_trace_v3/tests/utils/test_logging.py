"""
Logging Tests

ロギング機能のテスト。
以下の項目をテスト：
- ログ設定
- カラー出力
- コンテキスト管理
"""

import json
import logging
import os
from pathlib import Path
from typing import List, Dict, Any

import pytest
from _pytest.logging import LogCaptureFixture

from build_and_trace_v3.utils.logging import (
    setup_logging,
    get_logger,
    ContextLogger,
    ColorFormatter,
    COLORS
)

class TestColorFormatter:
    """ColorFormatterのテスト"""
    
    @pytest.fixture
    def formatter(self) -> ColorFormatter:
        """テスト用のフォーマッターを提供"""
        return ColorFormatter(
            fmt='%(levelname)s - %(message)s'
        )
    
    def test_format_simple_message(self, formatter: ColorFormatter):
        """通常メッセージのフォーマットテスト"""
        record = logging.LogRecord(
            name='test',
            level=logging.INFO,
            pathname='test.py',
            lineno=1,
            msg='Test message',
            args=(),
            exc_info=None
        )
        
        formatted = formatter.format(record)
        
        # ターミナルでない場合は色なし
        assert 'INFO - Test message' in formatted
    
    def test_format_dict_message(self, formatter: ColorFormatter):
        """辞書メッセージのフォーマットテスト"""
        data = {'key': 'value', 'number': 42}
        record = logging.LogRecord(
            name='test',
            level=logging.INFO,
            pathname='test.py',
            lineno=1,
            msg=data,
            args=(),
            exc_info=None
        )
        
        formatted = formatter.format(record)
        
        # JSONとしてフォーマットされていることを確認
        assert json.loads(formatted.split(' - ')[1])['key'] == 'value'
    
    def test_color_codes(self, formatter: ColorFormatter):
        """カラーコードのテスト"""
        for level_name, color in COLORS.items():
            if level_name == 'RESET':
                continue
                
            level = getattr(logging, level_name)
            record = logging.LogRecord(
                name='test',
                level=level,
                pathname='test.py',
                lineno=1,
                msg='Test message',
                args=(),
                exc_info=None
            )
            
            # 強制的にターミナルモードを有効化
            with patch('sys.stderr.isatty', return_value=True):
                formatted = formatter.format(record)
                assert color in formatted
                assert COLORS['RESET'] in formatted

class TestContextLogger:
    """ContextLoggerのテスト"""
    
    @pytest.fixture
    def logger(self) -> ContextLogger:
        """テスト用のロガーを提供"""
        logger = ContextLogger('test')
        logger.setLevel(logging.DEBUG)
        return logger
    
    def test_bind_context(self, logger: ContextLogger, caplog: LogCaptureFixture):
        """コンテキストバインドのテスト"""
        caplog.set_level(logging.INFO)
        
        logger.bind(user='test_user', request_id='123')
        logger.info('Test message')
        
        assert 'user' in caplog.text
        assert 'test_user' in caplog.text
        assert 'request_id' in caplog.text
        assert '123' in caplog.text
    
    def test_bind_chain(self, logger: ContextLogger):
        """メソッドチェーンのテスト"""
        result = logger.bind(a=1).bind(b=2)
        
        assert result is logger
        assert logger.context == {'a': 1, 'b': 2}
    
    def test_dict_message_with_context(
        self,
        logger: ContextLogger,
        caplog: LogCaptureFixture
    ):
        """辞書メッセージとコンテキストの統合テスト"""
        caplog.set_level(logging.INFO)
        
        logger.bind(context_key='context_value')
        logger.info({'message_key': 'message_value'})
        
        log_message = json.loads(caplog.records[-1].message)
        assert log_message['context_key'] == 'context_value'
        assert log_message['message_key'] == 'message_value'
    
    def test_string_message_with_context(
        self,
        logger: ContextLogger,
        caplog: LogCaptureFixture
    ):
        """文字列メッセージとコンテキストの統合テスト"""
        caplog.set_level(logging.INFO)
        
        logger.bind(key='value')
        logger.info('Test message')
        
        assert 'Test message' in caplog.text
        assert '"key": "value"' in caplog.text

class TestLoggingSetup:
    """ロギング設定のテスト"""
    
    def test_setup_console_only(self, caplog: LogCaptureFixture):
        """コンソール出力のみの設定テスト"""
        logger = setup_logging()
        
        logger.info('Test message')
        
        assert 'Test message' in caplog.text
    
    def test_setup_with_file(self, temp_dir: Path):
        """ファイル出力ありの設定テスト"""
        log_dir = temp_dir / 'logs'
        logger = setup_logging(log_dir=str(log_dir))
        
        logger.info('Test message')
        
        log_files = list(log_dir.glob('*.log'))
        assert len(log_files) == 1
        assert 'Test message' in log_files[0].read_text()
    
    def test_log_rotation(self, temp_dir: Path):
        """ログローテーションのテスト"""
        log_dir = temp_dir / 'logs'
        logger = setup_logging(
            log_dir=str(log_dir),
            max_bytes=10,  # 小さいサイズで設定
            backup_count=2
        )
        
        # ログファイルが複数作成されるまで書き込み
        for i in range(100):
            logger.info(f'Test message {i}')
        
        log_files = list(log_dir.glob('*.log*'))
        assert 1 <= len(log_files) <= 3  # メインログ + バックアップ2つまで
    
    def test_log_level_setting(self, caplog: LogCaptureFixture):
        """ログレベル設定のテスト"""
        logger = setup_logging(log_level=logging.WARNING)
        
        logger.debug('Debug message')
        logger.info('Info message')
        logger.warning('Warning message')
        
        assert 'Debug message' not in caplog.text
        assert 'Info message' not in caplog.text
        assert 'Warning message' in caplog.text
    
    def test_get_logger(self):
        """ロガー取得のテスト"""
        logger1 = get_logger()
        logger2 = get_logger()
        
        assert logger1 is logger2  # 同じインスタンスを返す
        assert isinstance(logger1, ContextLogger)
