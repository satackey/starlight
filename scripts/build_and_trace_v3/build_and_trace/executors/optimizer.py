"""
Optimizer Executor

Starlightのoptimizerコマンドの実行を管理するモジュール。
以下の機能を実装：
- optimizerのon/off制御
- トレースの収集と報告
- グループ名の管理
"""

import re
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional, List

from .base import BaseExecutor, CommandError

class OptimizerAction(Enum):
    """optimizerの操作種別"""
    ON = auto()
    OFF = auto()
    REPORT = auto()

@dataclass
class OptimizerOptions:
    """optimizerコマンドのオプション設定"""
    
    action: OptimizerAction
    group_name: Optional[str] = None
    
    def validate(self):
        """オプションの検証"""
        if self.action == OptimizerAction.ON and not self.group_name:
            raise ValueError("Group name is required for optimizer ON action")
        
        if self.group_name:
            # グループ名の検証
            if not re.match(r'^[a-zA-Z0-9-]+$', self.group_name):
                raise ValueError(
                    "Invalid group name. Use only alphanumeric characters and hyphens."
                )

class OptimizerExecutor(BaseExecutor):
    """
    optimizerコマンドの実行を管理するクラス
    
    使用例:
    ```python
    async with OptimizerExecutor() as executor:
        # optimizerを開始
        await executor.start_optimizer("my-group")
        
        try:
            # 処理を実行
            ...
        finally:
            # optimizerを停止
            await executor.stop_optimizer()
            
        # トレースを報告
        await executor.report_traces()
    ```
    """
    
    def __init__(self, ctr_starlight_path: str = "ctr-starlight"):
        """
        Args:
            ctr_starlight_path: ctr-starlightコマンドのパス
        """
        super().__init__()
        self.ctr_starlight_path = ctr_starlight_path
        self._current_group: Optional[str] = None
    
    def _build_command(self, **kwargs) -> List[str]:
        """
        optimizerコマンドを構築
        
        Args:
            **kwargs: コマンドパラメータ
                - options: OptimizerOptions オブジェクト
        
        Returns:
            コマンドとその引数のリスト
        """
        options: OptimizerOptions = kwargs.get('options')
        if not options:
            raise ValueError("OptimizerOptions is required")
        
        # オプションの検証
        options.validate()
        
        cmd = ["sudo", self.ctr_starlight_path]
        
        if options.action == OptimizerAction.REPORT:
            cmd.extend(["report"])
        else:
            if options.group_name:
                cmd.extend(["--group", options.group_name])
            cmd.extend(["optimizer", options.action.name.lower()])
        
        return cmd
    
    async def start_optimizer(self, group_name: str) -> None:
        """
        optimizerを開始
        
        Args:
            group_name: 最適化グループ名
        
        Raises:
            CommandError: コマンド実行に失敗した場合
            ValueError: 無効なグループ名が指定された場合
        """
        options = OptimizerOptions(
            action=OptimizerAction.ON,
            group_name=group_name
        )
        
        try:
            await self.execute(options=options)
            self._current_group = group_name
        except CommandError as e:
            error_details = "\n".join(e.result.stderr) if e.result else "No error details available"
            raise CommandError(
                f"Failed to start optimizer: {error_details}",
                result=e.result
            ) from e
    
    async def stop_optimizer(self) -> None:
        """
        optimizerを停止
        
        Raises:
            CommandError: コマンド実行に失敗した場合
        """
        options = OptimizerOptions(action=OptimizerAction.OFF)
        
        try:
            await self.execute(options=options)
        except CommandError as e:
            error_details = "\n".join(e.result.stderr) if e.result else "No error details available"
            raise CommandError(
                f"Failed to stop optimizer: {error_details}",
                result=e.result
            ) from e
        finally:
            self._current_group = None
    
    async def report_traces(self) -> None:
        """
        トレースを報告
        
        Raises:
            CommandError: コマンド実行に失敗した場合
        """
        options = OptimizerOptions(action=OptimizerAction.REPORT)
        
        try:
            await self.execute(options=options)
        except CommandError as e:
            error_details = "\n".join(e.result.stderr) if e.result else "No error details available"
            raise CommandError(
                f"Failed to report traces: {error_details}",
                result=e.result
            ) from e
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """コンテキストマネージャの終了処理"""
        if self._current_group:
            # 実行中のoptimizerがある場合は停止
            await self.stop_optimizer()
        await super().__aexit__(exc_type, exc_val, exc_tb)
