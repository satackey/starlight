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
from typing import Optional, List, Tuple
from urllib.parse import urlparse

import git

from ..executors.buildctl import BuildctlExecutor, BuildOptions
from ..executors.convert import ConvertExecutor
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
    processor = DockerfileProcessor(profile="myproxy")
    await processor.process_dockerfile(url)
    ```
    """
    
    def __init__(
        self,
        profile: Optional[str] = None,
        registry: str = "cloud.cluster.local:5000"
    ):
        """
        Args:
            profile: Starlightプロファイル名（例：myproxy）
            registry: レジストリのアドレス
        """
        self.buildctl = BuildctlExecutor()
        self.convert = ConvertExecutor()
        self.optimizer = OptimizerExecutor()
        self.profile = profile
        self.registry = registry
    
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
    
    def _extract_base_image(self, dockerfile_path: str) -> Tuple[str, str]:
        """
        Dockerfileからベースイメージとタグを抽出
        
        Args:
            dockerfile_path: Dockerfileのパス
            
        Returns:
            (イメージ名, タグ)のタプル
            
        Raises:
            ValueError: FROMが見つからない場合
        """
        with open(dockerfile_path, 'r') as f:
            content = f.read()
        
        # FROMを探す（マルチステージビルドの場合は最初のFROMを使用）
        match = re.search(r'FROM\s+([^:\s]+):?([^\s]*)', content)
        if not match:
            raise ValueError("No FROM instruction found in Dockerfile")
        
        image = match.group(1)
        tag = match.group(2) or "latest"
        
        return image, tag
    
    def _rewrite_dockerfile(
        self,
        dockerfile_path: str,
        source_image: str,
        destination_image: str
    ) -> None:
        """
        Dockerfileのベースイメージを書き換え
        
        Args:
            dockerfile_path: Dockerfileのパス
            source_image: 元のイメージ名
            destination_image: 新しいイメージ名
        """
        with open(dockerfile_path, 'r') as f:
            content = f.read()
        
        # FROMを書き換え
        new_content = re.sub(
            f'FROM\\s+{re.escape(source_image)}',
            f'FROM {destination_image}',
            content
        )
        
        with open(dockerfile_path, 'w') as f:
            f.write(new_content)
    
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
                
                # ベースイメージを抽出
                base_image, tag = self._extract_base_image(dockerfile_path)
                source_image = f"{base_image}:{tag}"
                
                # ファイル名（拡張子を除く）を取得
                filename = os.path.splitext(os.path.basename(info.filepath))[0]
                
                # 特殊文字を除去
                safe_owner = re.sub(r'[^a-zA-Z0-9-]', '', info.user)
                safe_repo = re.sub(r'[^a-zA-Z0-9-]', '', info.repo)
                safe_filename = re.sub(r'[^a-zA-Z0-9-]', '', filename)
                
                # 変換後のイメージ名を生成
                destination_image = (
                    f"{self.registry}/{base_image}:{tag}-starlight"
                    f"-{safe_owner}-{safe_repo}-{safe_filename}"
                )
                
                # イメージを変換
                logger.info(f"Converting image: {source_image} -> {destination_image}")
                async with self.convert:
                    await self.convert.convert(
                        source_image=source_image,
                        destination_image=destination_image,
                        profile=self.profile,
                        insecure_destination=True,
                        notify=True
                    )
                
                # Dockerfileを書き換え
                self._rewrite_dockerfile(
                    dockerfile_path=dockerfile_path,
                    source_image=source_image,
                    destination_image=destination_image
                )
                
                # 非同期コンテキストマネージャを使用
                async with self.buildctl, self.optimizer:
                    # optimizerを開始（グループ名は任意）
                    await self.optimizer.start_optimizer(info.group_name if info.group_name else "default")
                    
                    try:
                        # Dockerfileをビルド
                        await self.buildctl.build(
                            context_dir=os.path.dirname(dockerfile_path),
                            dockerfile=dockerfile_path
                        )
                    finally:
                        # 必ずoptimizerを停止
                        await self.optimizer.stop_optimizer()
                    
                    # トレースを報告（プロファイルを指定）
                    await self.optimizer.report_traces(profile=self.profile)
                
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
    processor = BatchDockerfileProcessor(profile="myproxy")
    await processor.process_dockerfiles(urls)
    ```
    """
    
    def __init__(
        self,
        concurrency: int = 1,
        profile: Optional[str] = None,
        registry: str = "cloud.cluster.local:5000"
    ):
        """
        Args:
            concurrency: 同時処理数
            profile: Starlightプロファイル名（例：myproxy）
            registry: レジストリのアドレス
        """
        self.concurrency = concurrency
        self.processor = DockerfileProcessor(
            profile=profile,
            registry=registry
        )
    
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
