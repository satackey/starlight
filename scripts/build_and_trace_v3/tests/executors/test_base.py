"""
Base Executor Tests

BaseExecutorの機能テスト。
以下の項目をテスト：
- コマンド実行
- 出力処理
- エラーハンドリング
- シグナルハンドリング
"""

import asyncio
import signal
from typing import List

import pytest

from build_and_trace_v3.executors.base import BaseExecutor, CommandError, ExecutionResult

class TestExecutor(BaseExecutor):
    """テスト用のExecutor実装"""
    
    def __init__(self, command_output: str = "", command_error: str = "", return_code: int = 0):
        super().__init__()
        self.command_output = command_output
        self.command_error = command_error
        self.return_code = return_code
        self.built_command: List[str] = []
    
    def _build_command(self, **kwargs) -> List[str]:
        """テスト用のコマンド構築"""
        self.built_command = ["test", "command"]
        if "arg" in kwargs:
            self.built_command.append(kwargs["arg"])
        return self.built_command

class TestBaseExecutor:
    """BaseExecutorのテスト"""
    
    @pytest.mark.asyncio
    async def test_successful_execution(self, command_output: str):
        """正常実行のテスト"""
        executor = TestExecutor(command_output=command_output)
        
        result = await executor.execute(arg="test")
        
        assert result.return_code == 0
        assert result.stdout == [command_output]
        assert result.stderr == []
        assert result.command == "test command test"
        assert executor.built_command == ["test", "command", "test"]
    
    @pytest.mark.asyncio
    async def test_failed_execution(self, command_error: str):
        """失敗実行のテスト"""
        executor = TestExecutor(command_error=command_error, return_code=1)
        
        with pytest.raises(CommandError) as exc_info:
            await executor.execute()
        
        assert exc_info.value.result is not None
        assert exc_info.value.result.return_code == 1
        assert exc_info.value.result.stderr == [command_error]
    
    @pytest.mark.asyncio
    async def test_output_handler(self):
        """出力ハンドラのテスト"""
        executor = TestExecutor(command_output="test output")
        output_lines = []
        
        async def handler(text: str, stream: str):
            output_lines.append((text, stream))
        
        executor.add_output_handler(handler)
        await executor.execute()
        
        assert output_lines == [("test output", "stdout")]
    
    @pytest.mark.asyncio
    async def test_context_manager(self):
        """コンテキストマネージャのテスト"""
        async with TestExecutor() as executor:
            result = await executor.execute()
            assert result.success
    
    @pytest.mark.asyncio
    async def test_signal_handling(self):
        """シグナルハンドリングのテスト"""
        executor = TestExecutor()
        
        # 長時間実行するタスクをシミュレート
        async def long_running_task():
            await executor.execute()
            await asyncio.sleep(10)
        
        # タスクを開始
        task = asyncio.create_task(long_running_task())
        
        # 少し待ってからシグナルを送信
        await asyncio.sleep(0.1)
        os.kill(os.getpid(), signal.SIGINT)
        
        # タスクが中断されることを確認
        with pytest.raises(asyncio.CancelledError):
            await task
    
    @pytest.mark.asyncio
    async def test_multiple_output_handlers(self):
        """複数の出力ハンドラのテスト"""
        executor = TestExecutor(command_output="test")
        outputs1 = []
        outputs2 = []
        
        async def handler1(text: str, stream: str):
            outputs1.append((text, stream))
        
        async def handler2(text: str, stream: str):
            outputs2.append((text, stream))
        
        executor.add_output_handler(handler1)
        executor.add_output_handler(handler2)
        
        await executor.execute()
        
        assert outputs1 == outputs2 == [("test", "stdout")]
    
    @pytest.mark.asyncio
    async def test_handler_error(self, caplog):
        """出力ハンドラのエラー処理テスト"""
        executor = TestExecutor(command_output="test")
        
        async def failing_handler(text: str, stream: str):
            raise Exception("Handler error")
        
        executor.add_output_handler(failing_handler)
        await executor.execute()
        
        # ハンドラのエラーがログに記録されることを確認
        assert "Error in output handler" in caplog.text
    
    @pytest.mark.asyncio
    async def test_stop(self):
        """停止処理のテスト"""
        executor = TestExecutor()
        
        # 実行を開始
        task = asyncio.create_task(executor.execute())
        
        # 少し待ってから停止
        await asyncio.sleep(0.1)
        await executor.stop()
        
        # プロセスが停止することを確認
        with pytest.raises(asyncio.CancelledError):
            await task
    
    @pytest.mark.asyncio
    async def test_execution_result_properties(self):
        """ExecutionResultのプロパティテスト"""
        result = ExecutionResult(
            return_code=0,
            stdout=["output"],
            stderr=["error"],
            duration=1.0,
            command="test"
        )
        
        assert result.success
        assert result.return_code == 0
        assert result.stdout == ["output"]
        assert result.stderr == ["error"]
        assert result.duration == 1.0
        assert result.command == "test"
    
    @pytest.mark.asyncio
    async def test_command_error_properties(self):
        """CommandErrorのプロパティテスト"""
        result = ExecutionResult(
            return_code=1,
            stdout=[],
            stderr=["error"],
            duration=1.0,
            command="test"
        )
        
        error = CommandError("Test error", result=result)
        
        assert str(error) == "Test error"
        assert error.result == result
        assert not error.result.success
