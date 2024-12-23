"""
BuildCtl Executor

buildctlコマンドの実行を管理するモジュール。
以下の機能を実装：
- Dockerfileのビルド実行
- ビルドオプションの管理
- ビルド固有のエラーハンドリング
"""

import os
from dataclasses import dataclass
from typing import Optional, List

from .base import BaseExecutor, CommandError

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
            "--trace", f"mode={options.trace_mode}",
            
            # 進捗表示
            f"--progress={options.progress}"
        ]
        
        return cmd
    
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
