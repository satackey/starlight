"""
BuildCtl Executor

buildctlコマンドの実行を管理するモジュール。
以下の機能を実装：
- Dockerfileのビルド実行
- ビルドオプションの管理
- ビルド固有のエラーハンドリング
"""

import asyncio
import logging
import os
import time
from dataclasses import dataclass
from typing import Optional, List, Dict

from .base import BaseExecutor, CommandError, ExecutionResult

logger = logging.getLogger(__name__)

@dataclass
class BuildOptions:
    """buildctlコマンドのオプション設定"""
    
    # コンテキストとDockerfile
    context_dir: str
    dockerfile: str
    
    # ビルド設定
    frontend: str = "dockerfile.v0"
    output_type: str = "image"
    image_name: str = "build-output"
    push: bool = False
    
    # トレース設定
    trace_mode: int = 1
    
    # 進捗表示
    progress: str = "plain"
    
    @property
    def dockerfile_name(self) -> str:
        """Dockerfileの名前を取得"""
        return os.path.basename(self.dockerfile)
    
    @property
    def dockerfile_dir(self) -> str:
        """Dockerfileのディレクトリを取得"""
        return os.path.dirname(self.dockerfile)

class BuildctlExecutor(BaseExecutor):
    """
    buildctlコマンドの実行を管理するクラス
    
    使用例:
    ```python
    options = BuildOptions(
        context_dir="/path/to/context",
        dockerfile="/path/to/Dockerfile"
    )
    
    async with BuildctlExecutor() as executor:
        result = await executor.execute(options=options)
    ```
    """
    
    def __init__(self, buildctl_path: str = "buildctl"):
        """
        Args:
            buildctl_path: buildctlコマンドのパス
        """
        super().__init__()
        self.buildctl_path = buildctl_path
    
    def _build_command(self, **kwargs) -> List[str]:
        """
        buildctlコマンドを構築
        
        Args:
            **kwargs: コマンドパラメータ
                - options: BuildOptions オブジェクト
        
        Returns:
            コマンドとその引数のリスト
        """
        options: BuildOptions = kwargs.get('options')
        if not options:
            raise ValueError("BuildOptions is required")
        
        cmd = [
            "sudo", self.buildctl_path, "build",
            # フロントエンド設定
            f"--frontend={options.frontend}",
            # "--frontend-opt", f"filename={options.dockerfile_name}",
            
            # コンテキストとDockerfile
            f"--local=context={options.context_dir}",
            f"--local=dockerfile={options.dockerfile_dir}",
            
            # 出力設定
            "--output", f"type={options.output_type},"
                       f"name={options.image_name},"
                       f"push={str(options.push).lower()}",
            
            # トレース設定
            # "--trace", f"mode={options.trace_mode}",
            
            # 進捗表示
            f"--progress={options.progress}"
        ]
        
        return cmd
    
    async def execute(self, **kwargs) -> ExecutionResult:
        """
        コマンドを実行（環境変数を設定）
        
        Args:
            **kwargs: コマンドのパラメータ
            
        Returns:
            ExecutionResult: 実行結果
            
        Raises:
            CommandError: コマンド実行に失敗した場合
        """
        # 現在の環境変数をコピー
        env = os.environ.copy()
        # CONTAINERD_SNAPSHOTTERを設定
        env["CONTAINERD_SNAPSHOTTER"] = "starlight"
        
        start_time = time.time()
        cmd = self._build_command(**kwargs)
        logger.info(f"Executing command: {' '.join(cmd)}")
        
        try:
            # プロセス作成（環境変数を設定）
            self.process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env  # 環境変数を渡す
            )
            
            # 出力ストリームの処理を開始
            stdout_task = asyncio.create_task(
                self._stream_output(self.process.stdout, "stdout")
            )
            stderr_task = asyncio.create_task(
                self._stream_output(self.process.stderr, "stderr")
            )
            
            # プロセスの完了を待つ
            return_code = await self.process.wait()
            
            # 出力の収集を完了
            stdout_lines = await stdout_task
            stderr_lines = await stderr_task
            
            duration = time.time() - start_time
            
            # 実行結果を作成
            result = ExecutionResult(
                return_code=return_code,
                stdout=stdout_lines,
                stderr=stderr_lines,
                duration=duration,
                command=' '.join(cmd)
            )
            
            if not result.success:
                raise CommandError(
                    f"Command failed with return code {return_code}",
                    result=result
                )
            
            return result
            
        except asyncio.CancelledError:
            await self.stop()
            raise
        
        finally:
            self.process = None
            self._stop_event.clear()
    
    async def build(self, context_dir: str, dockerfile: str, **kwargs) -> None:
        """
        Dockerfileをビルド
        
        Args:
            context_dir: ビルドコンテキストのディレクトリ
            dockerfile: Dockerfileのパス
            **kwargs: その他のビルドオプション
        
        Raises:
            CommandError: ビルドに失敗した場合
            ValueError: 無効なオプションが指定された場合
        """
        options = BuildOptions(
            context_dir=context_dir,
            dockerfile=dockerfile,
            **kwargs
        )
        
        try:
            await self.execute(options=options)
        except CommandError as e:
            # ビルドエラーの詳細を含めて再送出
            error_details = "\n".join(e.result.stderr) if e.result else "No error details available"
            raise CommandError(
                f"Build failed: {error_details}",
                result=e.result
            ) from e
