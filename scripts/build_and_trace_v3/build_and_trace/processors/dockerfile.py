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

from ..executors.container import ContainerExecutor, ContainerOptions
from ..executors.convert import ConvertExecutor
from ..executors.optimizer import OptimizerExecutor
from ..executors.pull import PullExecutor

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
        self.container = ContainerExecutor()
        self.convert = ConvertExecutor()
        self.optimizer = OptimizerExecutor()
        self.pull = PullExecutor()
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

    def _extract_first_run(self, dockerfile_path: str) -> Optional[str]:
        """
        Dockerfileから最初のRUN命令を抽出
        
        Args:
            dockerfile_path: Dockerfileのパス
            
        Returns:
            RUN命令の文字列、見つからない場合はNone
        """
        with open(dockerfile_path, 'r') as f:
            content = f.read()
        
        # RUN命令を探す（最初のRUNを使用）
        matches = list(re.finditer(r'RUN\s+(.+?)(?:\n|$)', content))
        if not matches:
            return None
        
        # 最初のRUN命令を取得
        first_run = matches[0].group(1).strip()
        
        # バックスラッシュで続く行を結合
        if first_run.endswith('\\'):
            lines = []
            current_line = first_run[:-1].strip()
            for line in content[matches[0].end():].split('\n'):
                line = line.strip()
                if not line:
                    continue
                if line.endswith('\\'):
                    lines.append(line[:-1].strip())
                else:
                    lines.append(line)
                    break
            first_run = ' '.join([current_line] + lines)
        
        return first_run
    
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
                async with self.container, self.optimizer, self.pull:
                    # optimizerを開始（グループ名は任意）
                    await self.optimizer.start_optimizer(info.group_name if info.group_name else "default")
                    
                    try:
                        # 変換したイメージをpull
                        logger.info(f"Pulling image: {destination_image}")
                        await self.pull.pull(
                            image=destination_image,
                            profile=self.profile
                        )
                        
                        # RUNを抽出（見つからない場合はスキップ）
                        run_cmd = self._extract_first_run(dockerfile_path)
                        if run_cmd is None:
                            logger.info(f"No RUN instruction found in Dockerfile: {dockerfile_path}, skipping...")
                            return
                        
                        logger.info(f"Found RUN instruction: {run_cmd}")
                        
                        # # データディレクトリを作成（一時的なディレクトリ）
                        # data_dir = f"/tmp/test-{info.repo}-data"
                        # os.makedirs(data_dir, exist_ok=True)
                        # logger.info(f"Created data directory: {data_dir}")
                        
                        try:
                            # コンテナを作成して起動
                            options = ContainerOptions(
                                image=destination_image,
                                instance=f"{info.repo}-{info.sha[:7]}",  # リポジトリ名を含める
                                command=run_cmd,  # 最初のRUN命令を使用
                                # mount_src=data_dir,
                                # mount_dst="/data",
                                env_file="../demo/config/all.env"
                            )
                            
                            # コンテナを作成して起動（インスタンス名を取得）
                            instance = await self.container.create_and_start(options=options)
                            
                            try:
                                # Ctrl-Cで停止されるまで待機
                                while True:
                                    await asyncio.sleep(1)
                            except asyncio.CancelledError:
                                # コンテナを削除してから例外を再送出
                                await self.container.remove_container(instance)
                                raise
                            
                        finally:
                            # データディレクトリのクリーンアップ
                            try:
                                import shutil
                                shutil.rmtree(data_dir)
                                logger.info(f"Cleaned up data directory: {data_dir}")
                            except Exception as e:
                                logger.warning(f"Failed to clean up data directory {data_dir}: {e}")
                        
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
