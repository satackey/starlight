"""
Convert Executor

Starlightのconvertコマンドの実行を管理するモジュール。
以下の機能を実装：
- コンテナイメージの変換
- プロキシへの通知
"""

from dataclasses import dataclass
from typing import Optional, List

from .base import BaseExecutor, CommandError

@dataclass
class ConvertOptions:
    """convertコマンドのオプション設定"""
    source_image: str
    destination_image: str
    profile: Optional[str] = None
    platform: str = "linux/amd64"
    insecure_destination: bool = False
    notify: bool = True

class ConvertExecutor(BaseExecutor):
    """
    convertコマンドの実行を管理するクラス
    
    使用例:
    ```python
    async with ConvertExecutor() as executor:
        await executor.convert(
            source_image="docker.io/library/redis:6.2.1",
            destination_image="cloud.cluster.local:5000/redis:6.2.1-starlight",
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
        convertコマンドを構築
        
        Args:
            **kwargs: コマンドパラメータ
                - options: ConvertOptions オブジェクト
        
        Returns:
            コマンドとその引数のリスト
        """
        options: ConvertOptions = kwargs.get('options')
        if not options:
            raise ValueError("ConvertOptions is required")
        
        cmd = ["sudo", self.ctr_starlight_path, "convert"]
        
        # if options.insecure_destination:
        #     cmd.append("--insecure-destination")

        # とりあえず 常に insecure にしておく
        cmd.append("--insecure-destination")
        
        if options.notify:
            cmd.append("--notify")
        
        if options.profile:
            cmd.extend(["--profile", options.profile])

        # # とりあえず 常に myproxy にしておく
        # cmd.extend(["--profile", "myproxy"])
        
        cmd.extend(["--platform", options.platform])
        cmd.extend([options.source_image, options.destination_image])
        
        return cmd
    
    async def convert(
        self,
        source_image: str,
        destination_image: str,
        profile: Optional[str] = None,
        platform: str = "linux/amd64",
        insecure_destination: bool = False,
        notify: bool = True
    ) -> None:
        """
        コンテナイメージを変換
        
        Args:
            source_image: 変換元イメージ名
            destination_image: 変換先イメージ名
            profile: Starlightプロファイル名
            platform: プラットフォーム（例：linux/amd64）
            insecure_destination: 安全でない接続を許可
            notify: プロキシに通知
        
        Raises:
            CommandError: コマンド実行に失敗した場合
        """
        options = ConvertOptions(
            source_image=source_image,
            destination_image=destination_image,
            profile=profile,
            platform=platform,
            insecure_destination=insecure_destination,
            notify=notify
        )
        
        try:
            await self.execute(options=options)
        except CommandError as e:
            error_details = "\n".join(e.result.stderr) if e.result else "No error details available"
            raise CommandError(
                f"Failed to convert image: {error_details}",
                result=e.result
            ) from e
