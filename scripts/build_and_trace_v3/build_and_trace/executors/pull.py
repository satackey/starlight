"""
Pull Executor

Starlightのpullコマンドの実行を管理するモジュール。
以下の機能を実装：
- コンテナイメージのpull
"""

from dataclasses import dataclass
from typing import Optional, List

from .base import BaseExecutor, CommandError

@dataclass
class PullOptions:
    """pullコマンドのオプション設定"""
    image: str
    profile: Optional[str] = None

class PullExecutor(BaseExecutor):
    """
    pullコマンドの実行を管理するクラス
    
    使用例:
    ```python
    async with PullExecutor() as executor:
        await executor.pull(
            image="cloud.cluster.local/redis:6.2.1-starlight",
            profile="myproxy"
        )
    ```
    """
    
    def __init__(self, ctr_starlight_path: str = "ctr-starlight"):
        """
        Args:
            ctr_starlight_path: ctr-starlightコマンドのパス
        """
        super().__init__()
        self.ctr_starlight_path = ctr_starlight_path
    
    def _build_command(self, **kwargs) -> List[str]:
        """
        pullコマンドを構築
        
        Args:
            **kwargs: コマンドパラメータ
                - options: PullOptions オブジェクト
        
        Returns:
            コマンドとその引数のリスト
        """
        options: PullOptions = kwargs.get('options')
        if not options:
            raise ValueError("PullOptions is required")
        
        cmd = ["sudo", self.ctr_starlight_path, "pull"]
        
        if options.profile:
            cmd.extend(["--profile", options.profile])
        
        cmd.append(options.image)
        
        return cmd
    
    async def pull(
        self,
        image: str,
        profile: Optional[str] = None
    ) -> None:
        """
        コンテナイメージをpull
        
        Args:
            image: イメージ名
            profile: Starlightプロファイル名
        
        Raises:
            CommandError: コマンド実行に失敗した場合
        """
        options = PullOptions(
            image=image,
            profile=profile
        )
        
        try:
            await self.execute(options=options)
        except CommandError as e:
            error_details = "\n".join(e.result.stderr) if e.result else "No error details available"
            raise CommandError(
                f"Failed to pull image: {error_details}",
                result=e.result
            ) from e
