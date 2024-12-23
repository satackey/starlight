"""
Optimizer Executor Tests

OptimizerExecutorの機能テスト。
以下の項目をテスト：
- optimizer操作（on/off）
- グループ名の管理
- エラーハンドリング
"""

import pytest

from build_and_trace.executors.optimizer import (
    OptimizerExecutor,
    OptimizerOptions,
    OptimizerAction,
    CommandError
)

class TestOptimizerOptions:
    """OptimizerOptionsのテスト"""
    
    def test_valid_group_name(self):
        """有効なグループ名のテスト"""
        options = OptimizerOptions(
            action=OptimizerAction.ON,
            group_name="test-group-123"
        )
        
        options.validate()  # 例外が発生しないことを確認
    
    def test_invalid_group_name(self):
        """無効なグループ名のテスト"""
        invalid_names = [
            "test group",  # スペースを含む
            "test/group",  # スラッシュを含む
            "test_group",  # アンダースコアを含む
            "テストグループ",  # 非ASCII文字を含む
            "",  # 空文字
        ]
        
        for name in invalid_names:
            options = OptimizerOptions(
                action=OptimizerAction.ON,
                group_name=name
            )
            with pytest.raises(ValueError):
                options.validate()
    
    def test_group_name_required_for_on(self):
        """ONアクション時のグループ名必須チェック"""
        options = OptimizerOptions(
            action=OptimizerAction.ON,
            group_name=None
        )
        
        with pytest.raises(ValueError):
            options.validate()
    
    def test_group_name_optional_for_off_and_report(self):
        """OFF/REPORTアクション時のグループ名オプション"""
        for action in [OptimizerAction.OFF, OptimizerAction.REPORT]:
            options = OptimizerOptions(
                action=action,
                group_name=None
            )
            options.validate()  # 例外が発生しないことを確認

class TestOptimizerExecutor:
    """OptimizerExecutorのテスト"""
    
    @pytest.fixture
    def executor(self) -> OptimizerExecutor:
        """テスト用のExecutorを提供"""
        return OptimizerExecutor(ctr_starlight_path="/usr/bin/ctr-starlight")
    
    def test_build_command_on(self, executor: OptimizerExecutor):
        """ONコマンドの構築テスト"""
        options = OptimizerOptions(
            action=OptimizerAction.ON,
            group_name="test-group"
        )
        
        command = executor._build_command(options=options)
        
        assert command == [
            "sudo",
            "/usr/bin/ctr-starlight",
            "optimizer",
            "on",
            "--group",
            "test-group"
        ]
    
    def test_build_command_off(self, executor: OptimizerExecutor):
        """OFFコマンドの構築テスト"""
        options = OptimizerOptions(action=OptimizerAction.OFF)
        
        command = executor._build_command(options=options)
        
        assert command == [
            "sudo",
            "/usr/bin/ctr-starlight",
            "optimizer",
            "off"
        ]
    
    def test_build_command_report(self, executor: OptimizerExecutor):
        """REPORTコマンドの構築テスト"""
        options = OptimizerOptions(action=OptimizerAction.REPORT)
        
        command = executor._build_command(options=options)
        
        assert command == [
            "sudo",
            "/usr/bin/ctr-starlight",
            "report"
        ]
    
    @pytest.mark.asyncio
    async def test_start_optimizer(self, executor: OptimizerExecutor):
        """optimizer開始のテスト"""
        await executor.start_optimizer("test-group")
        assert executor._current_group == "test-group"
    
    @pytest.mark.asyncio
    async def test_stop_optimizer(self, executor: OptimizerExecutor):
        """optimizer停止のテスト"""
        await executor.start_optimizer("test-group")
        await executor.stop_optimizer()
        assert executor._current_group is None
    
    @pytest.mark.asyncio
    async def test_report_traces(self, executor: OptimizerExecutor):
        """トレース報告のテスト"""
        await executor.report_traces()  # 例外が発生しないことを確認
    
    @pytest.mark.asyncio
    async def test_context_manager_cleanup(self, executor: OptimizerExecutor):
        """コンテキストマネージャのクリーンアップテスト"""
        async with executor:
            await executor.start_optimizer("test-group")
            assert executor._current_group == "test-group"
        
        assert executor._current_group is None
    
    @pytest.mark.asyncio
    async def test_context_manager_cleanup_on_error(self, executor: OptimizerExecutor):
        """エラー時のコンテキストマネージャのクリーンアップテスト"""
        with pytest.raises(ValueError):
            async with executor:
                await executor.start_optimizer("test-group")
                raise ValueError("Test error")
        
        assert executor._current_group is None
    
    @pytest.mark.asyncio
    async def test_start_optimizer_invalid_group(self, executor: OptimizerExecutor):
        """無効なグループ名でのoptimizerの開始テスト"""
        with pytest.raises(ValueError):
            await executor.start_optimizer("invalid group name")
    
    @pytest.mark.asyncio
    async def test_stop_optimizer_when_not_running(self, executor: OptimizerExecutor):
        """未実行状態でのoptimizer停止テスト"""
        await executor.stop_optimizer()  # 例外が発生しないことを確認
    
    @pytest.mark.asyncio
    async def test_multiple_start_stop(self, executor: OptimizerExecutor):
        """複数回の開始/停止テスト"""
        await executor.start_optimizer("group1")
        assert executor._current_group == "group1"
        
        await executor.stop_optimizer()
        assert executor._current_group is None
        
        await executor.start_optimizer("group2")
        assert executor._current_group == "group2"
        
        await executor.stop_optimizer()
        assert executor._current_group is None
    
    @pytest.mark.asyncio
    async def test_output_handling(self, executor: OptimizerExecutor):
        """出力処理のテスト"""
        output_lines = []
        
        async def handler(text: str, stream: str):
            output_lines.append((text, stream))
        
        executor.add_output_handler(handler)
        
        await executor.start_optimizer("test-group")
        await executor.stop_optimizer()
        
        assert len(output_lines) > 0  # 何らかの出力があることを確認
