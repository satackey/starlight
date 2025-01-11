"""
Container Executor

ctr c create と ctr task start を使用してコンテナを管理するモジュール。
以下の機能を実装：
- コンテナの作成
- コンテナの起動
- コンテナ固有のエラーハンドリング
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
class ContainerOptions:
    """コンテナ作成と起動のオプション設定"""
    
    # コンテナ設定
    image: str
    instance: str
    command: Optional[str] = None
    
    # マウント設定
    mount_type: str = "bind"
    mount_src: Optional[str] = None
    mount_dst: Optional[str] = None
    mount_options: str = "rbind:rw"
    
    # 環境変数設定
    env_file: Optional[str] = None
    
    # ネットワーク設定
    network: str = "host"

class ContainerExecutor(BaseExecutor):
    """
    ctr c create と ctr task start を使用してコンテナを管理するクラス
    
    使用例:
    ```python
    options = ContainerOptions(
        image="registry.example.com/myapp:latest",
        instance="instance1",
        command="/usr/local/bin/myapp"
    )
    
    async with ContainerExecutor() as executor:
        await executor.create_and_start(options=options)
    ```
    """
    
    def __init__(self, ctr_path: str = "ctr"):
        """
        Args:
            ctr_path: ctrコマンドのパス
        """
        super().__init__()
        self.ctr_path = ctr_path
    
    def _build_command(self, **kwargs) -> List[str]:
        """
        コマンドを構築（BaseExecutorの要件を満たすため）
        
        Args:
            **kwargs: コマンドのパラメータ
                - action: "create" または "start" または "remove"
                - options: ContainerOptions オブジェクト（createの場合）
                - instance: インスタンス名（startとremoveの場合）
            
        Returns:
            コマンドとその引数のリスト
            
        Raises:
            ValueError: 無効なアクションまたは必要なパラメータが不足している場合
        """
        action = kwargs.get('action')
        if not action:
            raise ValueError("Action is required")
        
        if action == "create":
            return self._build_create_command(**kwargs)
        elif action == "start":
            instance = kwargs.get('instance')
            if not instance:
                raise ValueError("Instance name is required for start action")
            return self._build_start_command(instance)
        elif action == "remove":
            instance = kwargs.get('instance')
            if not instance:
                raise ValueError("Instance name is required for remove action")
            return self._build_remove_command(instance)
        else:
            raise ValueError(f"Invalid action: {action}")
    
    def _build_create_command(self, **kwargs) -> List[str]:
        """
        ctr c create コマンドを構築
        
        Args:
            **kwargs: コマンドパラメータ
                - options: ContainerOptions オブジェクト
        
        Returns:
            コマンドとその引数のリスト
        """
        options: ContainerOptions = kwargs.get('options')
        if not options:
            raise ValueError("ContainerOptions is required")
        
        cmd = [
            "sudo", self.ctr_path, "c", "create",
            "--snapshotter=starlight"
        ]
        
        # マウント設定
        if options.mount_src and options.mount_dst:
            cmd.extend([
                "--mount",
                f"type={options.mount_type},"
                f"src={options.mount_src},"
                f"dst={options.mount_dst},"
                f"options={options.mount_options}"
            ])
        
        # 環境変数設定
        if options.env_file:
            cmd.extend(["--env-file", options.env_file])
        
        # ネットワーク設定
        if options.network:
            cmd.extend(["--net-host"])
        
        # イメージとインスタンス名
        cmd.extend([options.image, options.instance])
        
        # コマンド
        if options.command:
            cmd.append(options.command)
        
        return cmd
    
    def _build_start_command(self, instance: str) -> List[str]:
        """
        ctr task start コマンドを構築
        
        Args:
            instance: インスタンス名
        
        Returns:
            コマンドとその引数のリスト
        """
        return ["sudo", self.ctr_path, "task", "start", instance]
    
    def _build_remove_command(self, instance: str) -> List[str]:
        """
        ctr container rm コマンドを構築
        
        Args:
            instance: インスタンス名
        
        Returns:
            コマンドとその引数のリスト
        """
        return ["sudo", self.ctr_path, "container", "rm", instance]
    
    async def create_container(self, options: ContainerOptions) -> ExecutionResult:
        """
        コンテナを作成
        
        Args:
            **kwargs: コマンドのパラメータ
            
        Returns:
            ExecutionResult: 実行結果
            
        Raises:
            CommandError: コマンド実行に失敗した場合
        """
        env = os.environ.copy()
        env["CONTAINERD_SNAPSHOTTER"] = "starlight"
        
        start_time = time.time()
        cmd = self._build_command(action="create", options=options)
        print("====================================")
        print(cmd)
        print("====================================")
        logger.info(f"Creating container: {' '.join(cmd)}")
        
        try:
            self.process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env
            )
            
            stdout_task = asyncio.create_task(
                self._stream_output(self.process.stdout, "stdout")
            )
            stderr_task = asyncio.create_task(
                self._stream_output(self.process.stderr, "stderr")
            )
            
            return_code = await self.process.wait()
            
            stdout_lines = await stdout_task
            stderr_lines = await stderr_task
            
            duration = time.time() - start_time
            
            result = ExecutionResult(
                return_code=return_code,
                stdout=stdout_lines,
                stderr=stderr_lines,
                duration=duration,
                command=' '.join(cmd)
            )
            
            if not result.success:
                raise CommandError(
                    f"Container creation failed with return code {return_code}",
                    result=result
                )
            
            return result
            
        except asyncio.CancelledError:
            await self.stop()
            raise
        
        finally:
            self.process = None
            self._stop_event.clear()
    
    async def start_container(self, instance: str) -> ExecutionResult:
        """
        コンテナを起動
        
        Args:
            instance: インスタンス名
            
        Returns:
            ExecutionResult: 実行結果
            
        Raises:
            CommandError: コマンド実行に失敗した場合
        """
        start_time = time.time()
        cmd = self._build_command(action="start", instance=instance)
        logger.info(f"Starting container: {' '.join(cmd)}")
        
        try:
            self.process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout_task = asyncio.create_task(
                self._stream_output(self.process.stdout, "stdout")
            )
            stderr_task = asyncio.create_task(
                self._stream_output(self.process.stderr, "stderr")
            )
            
            return_code = await self.process.wait()
            
            stdout_lines = await stdout_task
            stderr_lines = await stderr_task
            
            duration = time.time() - start_time
            
            result = ExecutionResult(
                return_code=return_code,
                stdout=stdout_lines,
                stderr=stderr_lines,
                duration=duration,
                command=' '.join(cmd)
            )
            
            if not result.success:
                raise CommandError(
                    f"Container start failed with return code {return_code}",
                    result=result
                )
            
            return result
            
        except asyncio.CancelledError:
            await self.stop()
            raise
        
        finally:
            self.process = None
            self._stop_event.clear()
    
    async def remove_container(self, instance: str) -> ExecutionResult:
        """
        コンテナを削除
        
        Args:
            instance: インスタンス名
            
        Returns:
            ExecutionResult: 実行結果
            
        Raises:
            CommandError: コマンド実行に失敗した場合
        """
        start_time = time.time()
        cmd = self._build_command(action="remove", instance=instance)
        logger.info(f"Removing container: {' '.join(cmd)}")
        
        try:
            self.process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout_task = asyncio.create_task(
                self._stream_output(self.process.stdout, "stdout")
            )
            stderr_task = asyncio.create_task(
                self._stream_output(self.process.stderr, "stderr")
            )
            
            return_code = await self.process.wait()
            
            stdout_lines = await stdout_task
            stderr_lines = await stderr_task
            
            duration = time.time() - start_time
            
            result = ExecutionResult(
                return_code=return_code,
                stdout=stdout_lines,
                stderr=stderr_lines,
                duration=duration,
                command=' '.join(cmd)
            )
            
            if not result.success:
                raise CommandError(
                    f"Container removal failed with return code {return_code}",
                    result=result
                )
            
            return result
            
        except asyncio.CancelledError:
            await self.stop()
            raise
        
        finally:
            self.process = None
            self._stop_event.clear()
    
    async def create_and_start(self, options: ContainerOptions) -> str:
        """
        コンテナを作成して起動
        
        Args:
            **kwargs: コマンドのパラメータ
                - options: ContainerOptions オブジェクト
        
        Returns:
            str: インスタンス名
        
        Raises:
            CommandError: コマンド実行に失敗した場合
            ValueError: 無効なオプションが指定された場合
        """
        try:
            # コンテナを作成
            await self.create_container(options)
            
            # コンテナを起動（Ctrl-Cで停止されるまで待機）
            try:
                await self.start_container(options.instance)
            except asyncio.CancelledError:
                # コンテナを削除
                await self.remove_container(options.instance)
                raise
            
            return options.instance
            
        except CommandError as e:
            # エラーの詳細を含めて再送出
            error_details = "\n".join(e.result.stderr) if e.result else "No error details available"
            raise CommandError(
                f"Container operation failed: {error_details}",
                result=e.result
            ) from e
