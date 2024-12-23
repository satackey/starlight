"""
Pytest Configuration

テストの共通設定とフィクスチャを提供。
"""

import asyncio
import logging
import os
import tempfile
from pathlib import Path
from typing import AsyncGenerator, Generator

import pytest
from _pytest.logging import LogCaptureFixture

from build_and_trace.utils.logging import setup_logging

# テスト用の一時ディレクトリ
@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """一時ディレクトリを提供"""
    with tempfile.TemporaryDirectory() as tmp:
        yield Path(tmp)

# テスト用のロガー
@pytest.fixture
def test_logger(caplog: LogCaptureFixture) -> Generator[logging.Logger, None, None]:
    """テスト用のロガーを提供"""
    caplog.set_level(logging.DEBUG)
    logger = setup_logging(log_level=logging.DEBUG)
    yield logger

# 非同期テスト用のイベントループ
@pytest.fixture
async def event_loop() -> AsyncGenerator[asyncio.AbstractEventLoop, None]:
    """非同期テスト用のイベントループを提供"""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

# モック用のコマンド実行結果
@pytest.fixture
def command_output() -> Generator[str, None, None]:
    """モック用のコマンド実行結果を提供"""
    yield "Test command output"

# モック用のエラー出力
@pytest.fixture
def command_error() -> Generator[str, None, None]:
    """モック用のエラー出力を提供"""
    yield "Test command error"

# テスト用のDockerfile URL
@pytest.fixture
def dockerfile_url() -> Generator[str, None, None]:
    """テスト用のDockerfile URLを提供"""
    yield "https://github.com/test/repo/blob/main/Dockerfile"

# テスト用のGitHubリポジトリ情報
@pytest.fixture
def github_info() -> Generator[dict, None, None]:
    """テスト用のGitHubリポジトリ情報を提供"""
    yield {
        "user": "test",
        "repo": "repo",
        "sha": "main",
        "filepath": "Dockerfile"
    }

# テスト用のCSVデータ
@pytest.fixture
def csv_data(temp_dir: Path) -> Generator[Path, None, None]:
    """テスト用のCSVデータを提供"""
    csv_path = temp_dir / "test.csv"
    csv_content = "Dockerfile URL\nhttps://github.com/test/repo/blob/main/Dockerfile"
    csv_path.write_text(csv_content)
    yield csv_path

# テスト用の環境変数
@pytest.fixture(autouse=True)
def env_setup() -> Generator[None, None, None]:
    """テスト用の環境変数を設定"""
    original_env = dict(os.environ)
    
    # テスト用の環境変数を設定
    os.environ.update({
        "GITHUB_TOKEN": "test_token",
        "BUILDCTL_PATH": "/usr/bin/buildctl",
        "CTR_STARLIGHT_PATH": "/usr/bin/ctr-starlight"
    })
    
    yield
    
    # 環境変数を元に戻す
    os.environ.clear()
    os.environ.update(original_env)

# 非同期モック用のコンテキストマネージャ
@pytest.fixture
def async_context():
    """非同期モック用のコンテキストマネージャを提供"""
    class AsyncContextManager:
        async def __aenter__(self):
            return self
        
        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass
    
    return AsyncContextManager()
