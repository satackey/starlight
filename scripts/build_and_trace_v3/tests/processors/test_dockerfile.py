"""
Dockerfile Processor Tests

DockerfileProcessorの機能テスト。
以下の項目をテスト：
- Dockerfile情報の解析
- ビルドプロセスの管理
- バッチ処理
"""

import asyncio
from pathlib import Path
from typing import List

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from build_and_trace_v3.processors.dockerfile import (
    DockerfileProcessor,
    BatchDockerfileProcessor,
    DockerfileInfo
)
from build_and_trace_v3.executors.base import CommandError

class TestDockerfileInfo:
    """DockerfileInfoのテスト"""
    
    def test_from_github_url(self):
        """GitHub URLからの情報抽出テスト"""
        url = "https://github.com/test/repo/blob/main/path/to/Dockerfile"
        info = DockerfileInfo.from_github_url(url)
        
        assert info.url == url
        assert info.user == "test"
        assert info.repo == "repo"
        assert info.sha == "main"
        assert info.filepath == "path/to/Dockerfile"
    
    def test_invalid_github_url(self):
        """無効なGitHub URLのテスト"""
        invalid_urls = [
            "https://example.com/test/repo",  # GitHubではないドメイン
            "https://github.com/test",  # パスが不完全
            "not-a-url",  # URLではない
        ]
        
        for url in invalid_urls:
            with pytest.raises(ValueError):
                DockerfileInfo.from_github_url(url)
    
    def test_group_name_generation(self):
        """グループ名生成のテスト"""
        info = DockerfileInfo(
            url="https://github.com/test/repo/blob/abc123/path/to/Dockerfile",
            user="test",
            repo="repo",
            sha="abc123",
            filepath="path/to/Dockerfile"
        )
        
        group_name = info.group_name
        
        assert "mloptimizer" in group_name
        assert "test" in group_name
        assert "repo" in group_name
        assert "abc123" in group_name[:20]  # SHA7文字のみ使用
        assert "to" in group_name  # ベースディレクトリ名

class TestDockerfileProcessor:
    """DockerfileProcessorのテスト"""
    
    @pytest.fixture
    def processor(self) -> DockerfileProcessor:
        """テスト用のProcessorを提供"""
        return DockerfileProcessor()
    
    @pytest.fixture
    def mock_repo(self, temp_dir: Path) -> Path:
        """モック用のリポジトリを提供"""
        repo_dir = temp_dir / "test-repo"
        repo_dir.mkdir()
        
        # テスト用のDockerfileを作成
        dockerfile_dir = repo_dir / "path" / "to"
        dockerfile_dir.mkdir(parents=True)
        dockerfile = dockerfile_dir / "Dockerfile"
        dockerfile.write_text("FROM python:3.9")
        
        return repo_dir
    
    @pytest.mark.asyncio
    async def test_process_dockerfile(self, processor: DockerfileProcessor, mock_repo: Path):
        """Dockerfile処理の基本テスト"""
        url = "https://github.com/test/repo/blob/main/path/to/Dockerfile"
        
        # GitHubリポジトリのクローンをモック
        with patch('git.Repo.clone_from') as mock_clone:
            mock_clone.return_value.working_dir = str(mock_repo)
            
            await processor.process_dockerfile(url)
    
    @pytest.mark.asyncio
    async def test_process_dockerfile_not_found(
        self,
        processor: DockerfileProcessor,
        mock_repo: Path
    ):
        """存在しないDockerfileのテスト"""
        url = "https://github.com/test/repo/blob/main/nonexistent/Dockerfile"
        
        with patch('git.Repo.clone_from') as mock_clone:
            mock_clone.return_value.working_dir = str(mock_repo)
            
            with pytest.raises(ValueError) as exc_info:
                await processor.process_dockerfile(url)
            
            assert "not found" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_process_dockerfile_build_error(
        self,
        processor: DockerfileProcessor,
        mock_repo: Path
    ):
        """ビルド失敗のテスト"""
        url = "https://github.com/test/repo/blob/main/path/to/Dockerfile"
        
        # 無効なDockerfileを作成
        dockerfile = mock_repo / "path" / "to" / "Dockerfile"
        dockerfile.write_text("INVALID")
        
        with patch('git.Repo.clone_from') as mock_clone:
            mock_clone.return_value.working_dir = str(mock_repo)
            
            with pytest.raises(CommandError):
                await processor.process_dockerfile(url)
    
    @pytest.mark.asyncio
    async def test_optimizer_lifecycle(self, processor: DockerfileProcessor, mock_repo: Path):
        """optimizerのライフサイクルテスト"""
        url = "https://github.com/test/repo/blob/main/path/to/Dockerfile"
        
        # optimizerの状態を追跡
        optimizer_state = {"running": False, "group": None}
        
        def mock_start_optimizer(group_name: str):
            optimizer_state["running"] = True
            optimizer_state["group"] = group_name
        
        def mock_stop_optimizer():
            optimizer_state["running"] = False
            optimizer_state["group"] = None
        
        # optimizerメソッドをモック
        processor.optimizer.start_optimizer = AsyncMock(side_effect=mock_start_optimizer)
        processor.optimizer.stop_optimizer = AsyncMock(side_effect=mock_stop_optimizer)
        
        with patch('git.Repo.clone_from') as mock_clone:
            mock_clone.return_value.working_dir = str(mock_repo)
            
            await processor.process_dockerfile(url)
            
            # optimizerが適切に停止されていることを確認
            assert not optimizer_state["running"]
            assert optimizer_state["group"] is None

class TestBatchDockerfileProcessor:
    """BatchDockerfileProcessorのテスト"""
    
    @pytest.fixture
    def processor(self) -> BatchDockerfileProcessor:
        """テスト用のProcessorを提供"""
        return BatchDockerfileProcessor(concurrency=2)
    
    @pytest.mark.asyncio
    async def test_batch_processing(self, processor: BatchDockerfileProcessor):
        """バッチ処理のテスト"""
        urls = [
            "https://github.com/test/repo1/blob/main/Dockerfile",
            "https://github.com/test/repo2/blob/main/Dockerfile",
            "https://github.com/test/repo3/blob/main/Dockerfile"
        ]
        
        processed_urls = []
        
        # 個別のDockerfile処理をモック
        async def mock_process(url: str):
            await asyncio.sleep(0.1)  # 処理時間をシミュレート
            processed_urls.append(url)
        
        processor.processor.process_dockerfile = AsyncMock(side_effect=mock_process)
        
        await processor.process_dockerfiles(urls)
        
        assert len(processed_urls) == len(urls)
        assert set(processed_urls) == set(urls)
    
    @pytest.mark.asyncio
    async def test_batch_processing_with_errors(
        self,
        processor: BatchDockerfileProcessor,
        caplog
    ):
        """エラーを含むバッチ処理のテスト"""
        urls = [
            "https://github.com/test/repo1/blob/main/Dockerfile",
            "invalid-url",
            "https://github.com/test/repo2/blob/main/Dockerfile"
        ]
        
        successful_urls = []
        
        async def mock_process(url: str):
            if "invalid" in url:
                raise ValueError("Invalid URL")
            successful_urls.append(url)
        
        processor.processor.process_dockerfile = AsyncMock(side_effect=mock_process)
        
        # エラーが発生してもバッチ処理は継続
        await processor.process_dockerfiles(urls)
        
        assert len(successful_urls) == 2
        assert "invalid-url" not in successful_urls
        assert "Invalid URL" in caplog.text
    
    @pytest.mark.asyncio
    async def test_concurrency_limit(self, processor: BatchDockerfileProcessor):
        """同時実行数制限のテスト"""
        urls = [f"https://github.com/test/repo{i}/blob/main/Dockerfile" for i in range(5)]
        
        active_count = 0
        max_active = 0
        
        async def mock_process(url: str):
            nonlocal active_count, max_active
            active_count += 1
            max_active = max(max_active, active_count)
            await asyncio.sleep(0.1)
            active_count -= 1
        
        processor.processor.process_dockerfile = AsyncMock(side_effect=mock_process)
        
        await processor.process_dockerfiles(urls)
        
        assert max_active <= processor.concurrency
