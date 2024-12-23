"""
GitHub Processor Tests

GitHubManagerの機能テスト。
以下の項目をテスト：
- リポジトリ情報の解析
- リポジトリ操作
- レート制限の管理
"""

import asyncio
import time
from pathlib import Path
from typing import Generator

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from build_and_trace.processors.github import (
    GitHubManager,
    GitHubRepository,
    RepositoryInfo,
    RateLimiter
)

class TestRepositoryInfo:
    """RepositoryInfoのテスト"""
    
    def test_from_url_with_file(self):
        """ファイルパス付きURLの解析テスト"""
        url = "https://github.com/test/repo/blob/main/path/to/file.txt"
        info = RepositoryInfo.from_url(url)
        
        assert info.url == url
        assert info.user == "test"
        assert info.repo == "repo"
        assert info.sha == "main"
        assert info.filepath == "path/to/file.txt"
    
    def test_from_url_without_file(self):
        """ファイルパスなしURLの解析テスト"""
        url = "https://github.com/test/repo/tree/main"
        info = RepositoryInfo.from_url(url)
        
        assert info.url == url
        assert info.user == "test"
        assert info.repo == "repo"
        assert info.sha == "main"
        assert info.filepath is None
    
    def test_invalid_urls(self):
        """無効なURLのテスト"""
        invalid_urls = [
            "https://example.com/test/repo",  # GitHubではないドメイン
            "https://github.com/test",  # 不完全なパス
            "not-a-url",  # URLではない
        ]
        
        for url in invalid_urls:
            with pytest.raises(ValueError):
                RepositoryInfo.from_url(url)
    
    def test_clone_url(self):
        """クローンURL生成のテスト"""
        info = RepositoryInfo(
            url="https://github.com/test/repo/blob/main/file.txt",
            user="test",
            repo="repo",
            sha="main",
            filepath="file.txt"
        )
        
        assert info.clone_url == "https://github.com/test/repo.git"
    
    def test_full_name(self):
        """フルネーム生成のテスト"""
        info = RepositoryInfo(
            url="https://github.com/test/repo/blob/main/file.txt",
            user="test",
            repo="repo",
            sha="main"
        )
        
        assert info.full_name == "test/repo"

class TestRateLimiter:
    """RateLimiterのテスト"""
    
    @pytest.fixture
    def limiter(self) -> RateLimiter:
        """テスト用のRateLimiterを提供"""
        return RateLimiter(requests_per_hour=60)  # 1分間に1リクエスト
    
    @pytest.mark.asyncio
    async def test_wait_timing(self, limiter: RateLimiter):
        """待機時間の計算テスト"""
        start_time = time.time()
        
        # 1回目のリクエスト（待機なし）
        await limiter.wait()
        first_duration = time.time() - start_time
        assert first_duration < 0.1  # ほぼ即時実行
        
        # 2回目のリクエスト（待機あり）
        start_time = time.time()
        await limiter.wait()
        second_duration = time.time() - start_time
        assert second_duration >= (3600 / 60) - 0.1  # 待機時間の検証

class TestGitHubRepository:
    """GitHubRepositoryのテスト"""
    
    @pytest.fixture
    def info(self) -> RepositoryInfo:
        """テスト用のRepositoryInfoを提供"""
        return RepositoryInfo(
            url="https://github.com/test/repo/blob/main/file.txt",
            user="test",
            repo="repo",
            sha="main",
            filepath="file.txt"
        )
    
    @pytest.fixture
    def repository(self, info: RepositoryInfo) -> GitHubRepository:
        """テスト用のGitHubRepositoryを提供"""
        return GitHubRepository(info)
    
    @pytest.mark.asyncio
    async def test_clone(self, repository: GitHubRepository, temp_dir: Path):
        """クローン処理のテスト"""
        with patch('git.Repo.clone_from') as mock_clone:
            mock_repo = MagicMock()
            mock_repo.working_dir = str(temp_dir)
            mock_clone.return_value = mock_repo
            
            repo_path = await repository._clone(str(temp_dir))
            
            assert repo_path == str(temp_dir)
            mock_clone.assert_called_once()
            mock_repo.git.checkout.assert_called_once_with(repository.info.sha)
    
    def test_get_file_path(self, repository: GitHubRepository, temp_dir: Path):
        """ファイルパス取得のテスト"""
        repository._path = str(temp_dir)
        
        # 指定されたファイルパス
        path = repository.get_file_path("test.txt")
        assert path == str(temp_dir / "test.txt")
        
        # info.filepathを使用
        path = repository.get_file_path()
        assert path == str(temp_dir / "file.txt")
    
    def test_get_file_path_errors(self, repository: GitHubRepository):
        """ファイルパス取得のエラーテスト"""
        # リポジトリがクローンされていない
        with pytest.raises(ValueError):
            repository.get_file_path()
        
        # パスが指定されていない
        repository._path = "/tmp"
        repository.info.filepath = None
        with pytest.raises(ValueError):
            repository.get_file_path()

class TestGitHubManager:
    """GitHubManagerのテスト"""
    
    @pytest.fixture
    def manager(self) -> GitHubManager:
        """テスト用のGitHubManagerを提供"""
        return GitHubManager()
    
    @pytest.mark.asyncio
    async def test_get_repository(self, manager: GitHubManager):
        """リポジトリ取得のテスト"""
        url = "https://github.com/test/repo/blob/main/file.txt"
        
        repo = await manager.get_repository(url)
        
        assert isinstance(repo, GitHubRepository)
        assert repo.info.user == "test"
        assert repo.info.repo == "repo"
    
    @pytest.mark.asyncio
    async def test_repository_reuse(self, manager: GitHubManager):
        """リポジトリの再利用テスト"""
        url1 = "https://github.com/test/repo/blob/main/file1.txt"
        url2 = "https://github.com/test/repo/blob/main/file2.txt"
        
        repo1 = await manager.get_repository(url1)
        repo2 = await manager.get_repository(url2)
        
        assert repo1 is repo2  # 同じリポジトリインスタンスを再利用
    
    @pytest.mark.asyncio
    async def test_repository_different_sha(self, manager: GitHubManager):
        """異なるSHAのリポジトリテスト"""
        url1 = "https://github.com/test/repo/blob/main/file.txt"
        url2 = "https://github.com/test/repo/blob/dev/file.txt"
        
        repo1 = await manager.get_repository(url1)
        repo2 = await manager.get_repository(url2)
        
        assert repo1 is not repo2  # 異なるSHAは別インスタンス
    
    def test_close_all(self, manager: GitHubManager):
        """全リポジトリのクローズテスト"""
        # モックリポジトリを作成
        mock_repos = [MagicMock() for _ in range(3)]
        manager._repos = {
            f"test/repo{i}": repo
            for i, repo in enumerate(mock_repos)
        }
        
        manager.close_all()
        
        # 全てのリポジトリがクローズされたことを確認
        for repo in mock_repos:
            repo.close.assert_called_once()
        assert not manager._repos
    
    @pytest.mark.asyncio
    async def test_context_manager(self, manager: GitHubManager):
        """コンテキストマネージャのテスト"""
        url = "https://github.com/test/repo/blob/main/file.txt"
        
        async with manager as m:
            repo = await m.get_repository(url)
            assert repo.info.full_name in manager._repos
        
        assert not manager._repos  # 全てクリーンアップされている
