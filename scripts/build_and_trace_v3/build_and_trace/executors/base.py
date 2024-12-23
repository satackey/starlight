"""
Base Command Executor

非同期コマンド実行の基本機能を提供するモジュール。
以下の機能を実装：
- 非同期コマンド実行
- 標準出力/エラー出力のストリーム処理
- シグナルハンドリング
- エラー管理
"""

import asyncio
import logging
import signal
import sys
from abc import ABC, abstractmethod
from asyncio.subprocess import Process
from dataclasses import dataclass
from typing import Optional, Dict, Any, Callable, Coroutine, List

logger = logging.getLogger(__name__)

@dataclass
class ExecutionResult:
    """コマンド実行結果を格納するデータクラス"""
    return_code: int
    stdout: List[str]
    stderr: List[str]
    duration: float  # 実行時間（秒）
    command: str    # 実行されたコマンド
    
    @property
    def success(self) -> bool:
        """コマンドが成功したかどうか"""
        return self.return_code == 0

class CommandError(Exception):
    """コマンド実行に関連するエラー"""
    def __init__(self, message: str, result: Optional[ExecutionResult] = None):
        super().__init__(message)
        self.result = result

class BaseExecutor(ABC):
    """
    コマンド実行の基本クラス
    
    このクラスは非同期コマンド実行の基本機能を提供します。
    サブクラスは特定のコマンド（buildctl, optimizerなど）用の実装を提供します。
    """
    
    def __init__(self):
        self.process: Optional[Process] = None
        self._stop_event = asyncio.Event()
        self._output_handlers: List[Callable[[str, str], Coroutine[Any, Any, None]]] = []
        
    def add_output_handler(self, handler: Callable[[str, str], Coroutine[Any, Any, None]]):
        """
        出力ハンドラを追加
        
        Args:
            handler: 非同期コールバック関数。引数は(出力テキスト, ストリーム種別["stdout"/"stderr"])
        """
        self._output_handlers.append(handler)
        
    async def _handle_output(self, text: str, stream: str):
        """
        出力を登録されたハンドラに送信
        
        Args:
            text: 出力テキスト
            stream: ストリーム種別（"stdout" or "stderr"）
        """
        for handler in self._output_handlers:
            try:
                await handler(text, stream)
            except Exception as e:
                logger.error(f"Error in output handler: {str(e)}")
    
    async def _stream_output(self, pipe: asyncio.StreamReader, stream_type: str) -> List[str]:
        """
        ストリームからの出力を処理
        
        Args:
            pipe: 出力パイプ
            stream_type: ストリーム種別（"stdout" or "stderr"）
            
        Returns:
            処理された出力行のリスト
        """
        output_lines = []
        try:
            while not self._stop_event.is_set():
                line = await pipe.readline()
                if not line:
                    break
                    
                text = line.decode().strip()
                output_lines.append(text)
                
                # ハンドラに出力を送信
                await self._handle_output(text, stream_type)
                
                # ログ出力
                if stream_type == "stderr":
                    logger.error(text)
                else:
                    logger.info(text)
                    
        except Exception as e:
            logger.error(f"Error reading {stream_type}: {str(e)}")
            
        return output_lines
    
    def _setup_signal_handlers(self):
        """シグナルハンドラを設定"""
        def handler(signum, frame):
            logger.info(f"Received signal {signum}")
            if self.process:
                # プロセスの停止をスケジュール
                asyncio.create_task(self.stop())
        
        signal.signal(signal.SIGINT, handler)
        signal.signal(signal.SIGTERM, handler)
    
    async def stop(self):
        """実行中のプロセスを停止"""
        self._stop_event.set()
        if self.process:
            try:
                self.process.terminate()
                await asyncio.wait_for(self.process.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                logger.warning("Process did not terminate, killing...")
                self.process.kill()
            except Exception as e:
                logger.error(f"Error stopping process: {str(e)}")
    
    @abstractmethod
    def _build_command(self, **kwargs) -> List[str]:
        """
        コマンドを構築
        
        Args:
            **kwargs: コマンドのパラメータ
            
        Returns:
            コマンドとその引数のリスト
        """
        pass
    
    async def execute(self, **kwargs) -> ExecutionResult:
        """
        コマンドを実行
        
        Args:
            **kwargs: コマンドのパラメータ
            
        Returns:
            ExecutionResult: 実行結果
            
        Raises:
            CommandError: コマンド実行に失敗した場合
        """
        import time
        
        cmd = self._build_command(**kwargs)
        start_time = time.time()
        
        try:
            # プロセス作成
            self.process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
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
            
    async def __aenter__(self):
        """コンテキストマネージャのエントリーポイント"""
        self._setup_signal_handlers()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """コンテキストマネージャの終了処理"""
        if self.process:
            await self.stop()
