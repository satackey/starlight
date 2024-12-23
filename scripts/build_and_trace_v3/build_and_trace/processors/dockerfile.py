"""
Dockerfile Processor

Dockerfileの処理を管理するモジュール。
以下の機能を実装：
- GitHubからのDockerfile取得
- ビルドプロセスの管理
- トレース収集の制御
"""

import asyncio
import logging
import os
import re
import tempfile
from dataclasses import dataclass
from typing import Optional, List
from urllib.parse import urlparse

import git

from ..executors.buildctl import BuildctlExecutor, BuildOptions
from ..executors.optimizer import OptimizerExecutor

logger = logging.getLogger(__name__)

@dataclass
class DockerfileInfo:
    """Dockerfileの情報を格納するデータクラス"""
    url: str
    user: str
    repo: str
    sha: str
    filepath: str
    
    @classmethod
    def from_github_url(cls, url: str) -> 'DockerfileInfo':
        """
        GitHub URLからDockerfileInfo を作成
        
        Args:
            url: GitHub URL
            
        Returns:
            DockerfileInfo オブジェクト
            
        Raises:
            ValueError: 無効なURLが指定された場合
        """
        try:
            parsed = urlparse(url)
            path_parts = parsed.path.split('/')
            
            # GitHub URLのフォーマット: /user/repo/blob/sha/path/to/Dockerfile
            return cls(
                url=url,
                user=path_parts[1],
                repo=path_parts[2],
                sha=path_parts[4],
                filepath='/'.join(path_parts[5:])
            )
        except Exception as e:
            raise ValueError(f"Invalid GitHub URL: {url}") from e
    
    @property
    def group_name(self) -> str:
        """最適化グループ名を生成"""
        # ベースディレクトリ名を取得
        basename = os.path.basename(os.path.dirname(self.filepath))
        
        # グループ名を生成（特殊文字を除去）
        components = [
            "mloptimizer",
            re.sub(r'[^a-zA-Z0-9-]', '', self.user),
            re.sub(r'[^a-zA-Z0-9-]', '', self.repo),
            self.sha[:7],
            re.sub(r'[^a-zA-Z0-9-]', '', basename)
        ]
        
        return '-'.join(components)

class DockerfileProcessor:
    """
    Dockerfileの処理を管理するクラス
    
    使用例:
    ```python
    processor = DockerfileProcessor()
    await processor.process_dockerfile(url)
    ```
    """
    
    def __init__(self):
        self.buildctl = BuildctlExecutor()
        self.optimizer = OptimizerExecutor()
    
    @staticmethod
    def clone_repository(info: DockerfileInfo, work_dir: str) -> str:
        """
        リポジトリをクローンし、特定のSHAでチェックアウト
        
        Args:
            info: Dockerfile情報
            work_dir: 作業ディレクトリ
            
        Returns:
            クローンしたリポジトリのパス
            
        Raises:
            git.GitError: Git操作に失敗した場合
        """
        repo_url = f"https://github.com/{info.user}/{info.repo}.git"
        repo_path = os.path.join(work_dir, info.repo)
        
        logger.info(f"Cloning repository: {repo_url}")
        git.Repo.clone_from(repo_url, repo_path)
        
        # 特定のSHAをチェックアウト
        repo = git.Repo(repo_path)
        repo.git.checkout(info.sha)
        
        return repo_path
    
    async def process_dockerfile(self, url: str) -> None:
        """
        Dockerfileを処理
        
        Args:
            url: Dockerfile のGitHub URL
            
        Raises:
            ValueError: 無効なURLまたはファイルが見つからない場合
            CommandError: コマンド実行に失敗した場合
        """
        # Dockerfile情報を解析
        info = DockerfileInfo.from_github_url(url)
        logger.info(f"Processing Dockerfile: {info.filepath}")
        
        with tempfile.TemporaryDirectory() as work_dir:
            try:
                # リポジトリをクローン
                repo_path = self.clone_repository(info, work_dir)
                dockerfile_path = os.path.join(repo_path, info.filepath)
                
                if not os.path.exists(dockerfile_path):
                    raise ValueError(f"Dockerfile not found: {dockerfile_path}")
                
                # 非同期コンテキストマネージャを使用
                async with self.buildctl, self.optimizer:
                    # optimizerを開始
                    await self.optimizer.start_optimizer(info.group_name)
                    
                    try:
                        # Dockerfileをビルド
                        await self.buildctl.build(
                            context_dir=os.path.dirname(dockerfile_path),
                            dockerfile=dockerfile_path
                        )
                    finally:
                        # 必ずoptimizerを停止
                        await self.optimizer.stop_optimizer()
                    
                    # トレースを報告
                    await self.optimizer.report_traces()
                
            except git.GitError as e:
                raise ValueError(f"Git operation failed: {str(e)}") from e
            except Exception as e:
                logger.error(f"Error processing {url}: {str(e)}")
                raise

class BatchDockerfileProcessor:
    """
    複数のDockerfileを処理するクラス
    
    使用例:
    ```python
    processor = BatchDockerfileProcessor()
    await processor.process_dockerfiles(urls)
    ```
    """
    
    def __init__(self, concurrency: int = 1):
        """
        Args:
            concurrency: 同時処理数
        """
        self.concurrency = concurrency
        self.processor = DockerfileProcessor()
    
    async def process_dockerfiles(self, urls: List[str]) -> None:
        """
        複数のDockerfileを処理
        
        Args:
            urls: Dockerfile のGitHub URLのリスト
        """
        # セマフォで同時実行数を制限
        semaphore = asyncio.Semaphore(self.concurrency)
        
        async def process_with_semaphore(url: str):
            async with semaphore:
                await self.processor.process_dockerfile(url)
                # レート制限を避けるために少し待機
                await asyncio.sleep(2)
        
        # タスクを作成
        tasks = [
            asyncio.create_task(process_with_semaphore(url))
            for url in urls
        ]
        
        # 全タスクの完了を待つ
        await asyncio.gather(*tasks)
