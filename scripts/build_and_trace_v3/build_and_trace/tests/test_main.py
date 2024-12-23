"""
Main Tests

メインモジュールのテスト。
以下の項目をテスト：
- コマンドライン引数
- CSVファイル処理
- メイン処理フロー
"""

import asyncio
import logging
import sys
from pathlib import Path
from typing import List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from build_and_trace.main import (
    parse_args,
    read_urls_from_csv,
    process_dockerfiles,
    main
)

class TestParseArgs:
    """引数パースのテスト"""
    
    def test_required_args(self):
        """必須引数のテスト"""
        with patch('sys.argv', ['script.py', 'test.csv']):
            args = parse_args()
            
            assert args.csv_file == 'test.csv'
            assert args.concurrency == 1  # デフォルト値
            assert args.log_dir is None  # デフォルト値
            assert args.log_level == 'INFO'  # デフォルト値
    
    def test_optional_args(self):
        """オプション引数のテスト"""
        with patch('sys.argv', [
            'script.py',
            'test.csv',
            '--concurrency', '2',
            '--log-dir', '/tmp/logs',
            '--log-level', 'DEBUG'
        ]):
            args = parse_args()
            
            assert args.csv_file == 'test.csv'
            assert args.concurrency == 2
            assert args.log_dir == '/tmp/logs'
            assert args.log_level == 'DEBUG'
    
    def test_invalid_concurrency(self):
        """無効な同時実行数のテスト"""
        with patch('sys.argv', ['script.py', 'test.csv', '--concurrency', '0']):
            with pytest.raises(SystemExit):
                parse_args()
    
    def test_invalid_log_level(self):
        """無効なログレベルのテスト"""
        with patch('sys.argv', ['script.py', 'test.csv', '--log-level', 'INVALID']):
            with pytest.raises(SystemExit):
                parse_args()

class TestReadUrlsFromCsv:
    """CSVファイル読み込みのテスト"""
    
    @pytest.fixture
    def csv_file(self, temp_dir: Path) -> Path:
        """テスト用のCSVファイルを提供"""
        csv_path = temp_dir / "test.csv"
        csv_path.write_text(
            "Dockerfile URL\n"
            "https://github.com/test/repo1/blob/main/Dockerfile\n"
            "https://github.com/test/repo2/blob/main/Dockerfile\n"
        )
        return csv_path
    
    def test_read_valid_csv(self, csv_file: Path):
        """正常なCSVファイルの読み込みテスト"""
        urls = read_urls_from_csv(str(csv_file))
        
        assert len(urls) == 2
        assert all(url.startswith('https://github.com/') for url in urls)
    
    def test_file_not_found(self, temp_dir: Path):
        """存在しないファイルのテスト"""
        with pytest.raises(FileNotFoundError):
            read_urls_from_csv(str(temp_dir / "nonexistent.csv"))
    
    def test_invalid_format(self, temp_dir: Path):
        """無効なフォーマットのテスト"""
        csv_path = temp_dir / "invalid.csv"
        csv_path.write_text("Invalid Header\nsome_url\n")
        
        with pytest.raises(ValueError) as exc_info:
            read_urls_from_csv(str(csv_path))
        
        assert "Dockerfile URL" in str(exc_info.value)
    
    def test_empty_file(self, temp_dir: Path):
        """空ファイルのテスト"""
        csv_path = temp_dir / "empty.csv"
        csv_path.write_text("")
        
        with pytest.raises(ValueError):
            read_urls_from_csv(str(csv_path))

class TestProcessDockerfiles:
    """Dockerfile処理のテスト"""
    
    @pytest.mark.asyncio
    async def test_process_dockerfiles(self, temp_dir: Path):
        """Dockerfile処理のテスト"""
        urls = [
            "https://github.com/test/repo1/blob/main/Dockerfile",
            "https://github.com/test/repo2/blob/main/Dockerfile"
        ]
        
        # BatchDockerfileProcessorをモック
        with patch('build_and_trace.main.BatchDockerfileProcessor') as mock_processor:
            processor_instance = mock_processor.return_value
            processor_instance.process_dockerfiles = AsyncMock()
            
            await process_dockerfiles(
                urls=urls,
                concurrency=2,
                log_dir=str(temp_dir)
            )
            
            # プロセッサが正しく設定されていることを確認
            mock_processor.assert_called_once_with(concurrency=2)
            processor_instance.process_dockerfiles.assert_called_once_with(urls)

class TestMain:
    """メイン処理のテスト"""
    
    @pytest.mark.asyncio
    async def test_main_success(self, temp_dir: Path):
        """正常実行のテスト"""
        # CSVファイルを作成
        csv_path = temp_dir / "test.csv"
        csv_path.write_text(
            "Dockerfile URL\n"
            "https://github.com/test/repo/blob/main/Dockerfile\n"
        )
        
        # コマンドライン引数をモック
        with patch('sys.argv', ['script.py', str(csv_path)]):
            # 処理をモック
            with patch('build_and_trace.main.process_dockerfiles') as mock_process:
                mock_process.return_value = None
                
                # 実行
                await main()
                
                # 処理が呼び出されたことを確認
                mock_process.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_main_no_urls(self, temp_dir: Path, caplog: pytest.LogCaptureFixture):
        """URLなしの実行テスト"""
        # 空のCSVファイルを作成
        csv_path = temp_dir / "empty.csv"
        csv_path.write_text("Dockerfile URL\n")
        
        # コマンドライン引数をモック
        with patch('sys.argv', ['script.py', str(csv_path)]):
            caplog.set_level(logging.WARNING)
            
            # 実行
            await main()
            
            # 警告が出力されることを確認
            assert "処理対象のURLがありません" in caplog.text
    
    @pytest.mark.asyncio
    async def test_main_error(self, temp_dir: Path):
        """エラー発生時のテスト"""
        # 無効なCSVファイルを作成
        csv_path = temp_dir / "invalid.csv"
        csv_path.write_text("Invalid Header\n")
        
        # コマンドライン引数をモック
        with patch('sys.argv', ['script.py', str(csv_path)]):
            # 実行
            with pytest.raises(SystemExit) as exc_info:
                await main()
            
            assert exc_info.value.code == 1
    
    @pytest.mark.asyncio
    async def test_keyboard_interrupt(self, temp_dir: Path, caplog: pytest.LogCaptureFixture):
        """キーボード割り込みのテスト"""
        # CSVファイルを作成
        csv_path = temp_dir / "test.csv"
        csv_path.write_text(
            "Dockerfile URL\n"
            "https://github.com/test/repo/blob/main/Dockerfile\n"
        )
        
        # コマンドライン引数をモック
        with patch('sys.argv', ['script.py', str(csv_path)]):
            # 処理中にKeyboardInterruptを発生させる
            with patch('build_and_trace.main.process_dockerfiles', side_effect=KeyboardInterrupt):
                caplog.set_level(logging.INFO)
                
                # 実行
                with pytest.raises(SystemExit) as exc_info:
                    await main()
                
                assert exc_info.value.code == 0
                assert "処理を中断しました" in caplog.text
