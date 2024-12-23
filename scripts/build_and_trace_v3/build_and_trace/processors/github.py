"""
GitHub Processor

GitHub関連の処理を管理するモジュール。
以下の機能を実装：
- リポジトリ操作（クローン、チェックアウト）
- URLの解析と検証
- レート制限の管理
"""

import asyncio
import logging
import os
import re
from dataclasses import dataclass
from typing import Optional, Dict, Any
from urllib.parse import urlparse

import git
from git.exc import GitError

logger = logging.getLogger(__name__)

@dataclass
class RepositoryInfo:
    """リポジトリ情報を格納するデータクラス"""
    url: str
    user: str
    repo: str
    sha: str
    filepath: Optional[str] = None
    
    @classmethod
    def from_url(cls, url: str) -> 'RepositoryInfo':
        """
        GitHub URLからRepositoryInfoを作成
        
        Args:
            url: GitHub URL
            
        Returns:
            RepositoryInfo オブジェクト
            
        Raises:
            ValueError: 無効なURLが指定された場合
        """
        try:
            parsed = urlparse(url)
            if not parsed.netloc == 'github.com':
                raise ValueError("Not a GitHub URL")
            
            path_parts = parsed.path.strip('/').split('/')
            if len(path_parts) < 3:
                raise ValueError("Invalid GitHub URL format")
            
            user = path_parts[0]
            repo = path_parts[1]
            
            # blob/SHA/path/to/file の形式をパース
            if len(path_parts) > 3 and path_parts[2] == 'blob':
                sha = path_parts[3]
                filepath = '/'.join(path_parts[4:]) if len(path_parts) > 4 else None
            else:
                sha = path_parts[2]
                filepath = None
            
            return cls(
                url=url,
                user=user,
                repo=repo,
                sha=sha,
                filepath=filepath
            )
        except Exception as e:
            raise ValueError(f"Invalid GitHub URL: {url}") from e
    
    @property
    def clone_url(self) -> str:
        """クローン用のURLを取得"""
        return f"https://github.com/{self.user}/{self.repo}.git"
    
    @property
    def full_name(self) -> str:
        """リポジトリのフルネームを取得"""
        return f"{self.user}/{self.repo}"

class RateLimiter:
    """GitHub APIのレート制限を管理するクラス"""
    
    def __init__(self, requests_per_hour: int = 60):
        """
        Args:
            requests_per_hour: 1時間あたりのリクエスト数制限
        """
        self.min_interval = 3600.0 / requests_per_hour
        self.last_request_time = 0.0
    
    async def wait(self):
        """レート制限に基づいて待機"""
        import time
        
        current_time = time.time()
        elapsed = current_time - self.last_request_time
        
        if elapsed < self.min_interval:
            wait_time = self.min_interval - elapsed
            logger.debug(f"Rate limit: waiting {wait_time:.2f} seconds")
            await asyncio.sleep(wait_time)
        
        self.last_request_time = time.time()

class GitHubRepository:
    """
    GitHubリポジトリの操作を管理するクラス
    
    使用例:
    ```python
    repo = GitHubRepository(info)
    async with repo.clone(work_dir) as repo_path:
        # リポジトリを使用
        ...
    ```
    """
    
    def __init__(self, info: RepositoryInfo):
        """
        Args:
            info: リポジトリ情報
        """
        self.info = info
        self.rate_limiter = RateLimiter()
        self._repo: Optional[git.Repo] = None
        self._path: Optional[str] = None
    
    async def _clone(self, work_dir: str) -> str:
        """
        リポジトリをクローン
        
        Args:
            work_dir: 作業ディレクトリ
            
        Returns:
            クローンしたリポジトリのパス
            
        Raises:
            GitError: Git操作に失敗した場合
        """
        # レート制限を考慮
        await self.rate_limiter.wait()
        
        # .cloned_reposディレクトリを作成
        clone_dir = os.path.join(work_dir, '.cloned_repos')
        os.makedirs(clone_dir, exist_ok=True)
        
        repo_path = os.path.join(clone_dir, self.info.repo)
        logger.info(f"Cloning repository: {self.info.clone_url}")
        
        try:
            self._repo = git.Repo.clone_from(self.info.clone_url, repo_path)
            self._path = repo_path
            
            # 特定のSHAをチェックアウト
            self._repo.git.checkout(self.info.sha)
            
            return repo_path
            
        except GitError as e:
            logger.error(f"Failed to clone repository: {str(e)}")
            if self._repo:
                self._repo.close()
            raise
    
    def close(self):
        """リポジトリをクリーンアップ"""
        if self._repo:
            try:
                self._repo.close()
            except Exception as e:
                logger.warning(f"Error closing repository: {str(e)}")
            finally:
                self._repo = None
                self._path = None
    
    @property
    def path(self) -> Optional[str]:
        """リポジトリのパスを取得"""
        return self._path
    
    def get_file_path(self, filepath: Optional[str] = None) -> str:
        """
        リポジトリ内のファイルパスを取得
        
        Args:
            filepath: ファイルパス（指定がない場合はinfo.filepathを使用）
            
        Returns:
            完全なファイルパス
            
        Raises:
            ValueError: パスが指定されていない場合
        """
        if not self._path:
            raise ValueError("Repository is not cloned")
        
        target_path = filepath or self.info.filepath
        if not target_path:
            raise ValueError("No filepath specified")
        
        return os.path.join(self._path, target_path)
    
    async def __aenter__(self) -> 'GitHubRepository':
        """非同期コンテキストマネージャのエントリーポイント"""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """非同期コンテキストマネージャの終了処理"""
        self.close()

class GitHubManager:
    """
    GitHub操作を管理するクラス
    
    使用例:
    ```python
    manager = GitHubManager()
    repo = await manager.get_repository("https://github.com/user/repo/blob/sha/path/to/file")
    ```
    """
    
    def __init__(self):
        self._repos: Dict[str, GitHubRepository] = {}
    
    async def get_repository(self, url: str) -> GitHubRepository:
        """
        URLからリポジトリを取得
        
        Args:
            url: GitHub URL
            
        Returns:
            GitHubRepository オブジェクト
        """
        info = RepositoryInfo.from_url(url)
        
        # 既存のリポジトリがあれば再利用
        if info.full_name in self._repos:
            repo = self._repos[info.full_name]
            if repo.info.sha != info.sha:
                # SHAが異なる場合は新しいリポジトリを作成
                repo.close()
                repo = GitHubRepository(info)
        else:
            repo = GitHubRepository(info)
        
        self._repos[info.full_name] = repo
        return repo
    
    def close_all(self):
        """全てのリポジトリをクリーンアップ"""
        for repo in self._repos.values():
            repo.close()
        self._repos.clear()
    
    async def __aenter__(self) -> 'GitHubManager':
        """非同期コンテキストマネージャのエントリーポイント"""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """非同期コンテキストマネージャの終了処理"""
        self.close_all()
