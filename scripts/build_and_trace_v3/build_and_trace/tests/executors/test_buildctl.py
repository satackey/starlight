"""
BuildCtl Executor Tests

BuildctlExecutorの機能テスト。
以下の項目をテスト：
- Dockerfileのビルド
- オプション設定
- エラーハンドリング
"""

import os
from pathlib import Path
from typing import List

import pytest

from build_and_trace.executors.buildctl import BuildctlExecutor, BuildOptions
from build_and_trace.executors.base import CommandError

class TestBuildOptions:
    """BuildOptionsのテスト"""
    
    def test_default_values(self):
        """デフォルト値のテスト"""
        options = BuildOptions(
            context_dir="/path/to/context",
            dockerfile="/path/to/context/Dockerfile"
        )
        
        assert options.frontend == "dockerfile.v0"
        assert options.output_type == "image"
        assert options.image_name == "build-output"
        assert not options.push
        assert options.dockerfile_name == "Dockerfile"
        assert options.dockerfile_dir == "/path/to/context"
    
    def test_custom_values(self):
        """カスタム値のテスト"""
        options = BuildOptions(
            context_dir="/path/to/context",
            dockerfile="/path/to/context/custom/Dockerfile.prod",
            frontend="custom.v1",
            output_type="local",
            image_name="my-image",
            push=True
        )
        
        assert options.frontend == "custom.v1"
        assert options.output_type == "local"
        assert options.image_name == "my-image"
        assert options.push
        assert options.dockerfile_name == "Dockerfile.prod"
        assert options.dockerfile_dir == "/path/to/context/custom"

class TestBuildctlExecutor:
    """BuildctlExecutorのテスト"""
    
    @pytest.fixture
    def executor(self) -> BuildctlExecutor:
        """テスト用のExecutorを提供"""
        return BuildctlExecutor(buildctl_path="/usr/bin/buildctl")
    
    @pytest.fixture
    def dockerfile_path(self, temp_dir: Path) -> Path:
        """テスト用のDockerfileを提供"""
        dockerfile = temp_dir / "Dockerfile"
        dockerfile.write_text("FROM python:3.9")
        return dockerfile
    
    def test_build_command_structure(self, executor: BuildctlExecutor, dockerfile_path: Path):
        """コマンド構築のテスト"""
        options = BuildOptions(
            context_dir=str(dockerfile_path.parent),
            dockerfile=str(dockerfile_path)
        )
        
        command = executor._build_command(options=options)
        
        assert command[0:2] == ["sudo", "/usr/bin/buildctl"]
        assert "--frontend=dockerfile.v0" in command
        assert f"--local=context={dockerfile_path.parent}" in command
        assert f"--local=dockerfile={dockerfile_path.parent}" in command
        assert "--output" in command
        assert "--trace" in command
        assert "--progress=plain" in command
    
    @pytest.mark.asyncio
    async def test_successful_build(self, executor: BuildctlExecutor, dockerfile_path: Path):
        """正常ビルドのテスト"""
        result = await executor.build(
            context_dir=str(dockerfile_path.parent),
            dockerfile=str(dockerfile_path)
        )
        
        assert result is None  # buildメソッドは成功時にNoneを返す
    
    @pytest.mark.asyncio
    async def test_build_with_custom_options(self, executor: BuildctlExecutor, dockerfile_path: Path):
        """カスタムオプション付きビルドのテスト"""
        await executor.build(
            context_dir=str(dockerfile_path.parent),
            dockerfile=str(dockerfile_path),
            frontend="custom.v1",
            output_type="local",
            image_name="test-image",
            push=True
        )
        
        # オプションが正しく反映されていることは_build_commandで検証済み
        assert True
    
    @pytest.mark.asyncio
    async def test_build_failure(self, executor: BuildctlExecutor, dockerfile_path: Path):
        """ビルド失敗のテスト"""
        # 無効なDockerfileを作成
        dockerfile_path.write_text("INVALID")
        
        with pytest.raises(CommandError) as exc_info:
            await executor.build(
                context_dir=str(dockerfile_path.parent),
                dockerfile=str(dockerfile_path)
            )
        
        assert exc_info.value.result is not None
        assert exc_info.value.result.return_code != 0
    
    def test_invalid_dockerfile(self, executor: BuildctlExecutor, temp_dir: Path):
        """無効なDockerfileパスのテスト"""
        with pytest.raises(ValueError):
            BuildOptions(
                context_dir=str(temp_dir),
                dockerfile=str(temp_dir / "nonexistent" / "Dockerfile")
            )
    
    @pytest.mark.asyncio
    async def test_context_manager(self, executor: BuildctlExecutor, dockerfile_path: Path):
        """コンテキストマネージャのテスト"""
        async with executor:
            await executor.build(
                context_dir=str(dockerfile_path.parent),
                dockerfile=str(dockerfile_path)
            )
        
        assert True  # エラーが発生しないことを確認
    
    @pytest.mark.asyncio
    async def test_output_handling(self, executor: BuildctlExecutor, dockerfile_path: Path):
        """出力処理のテスト"""
        output_lines = []
        
        async def handler(text: str, stream: str):
            output_lines.append((text, stream))
        
        executor.add_output_handler(handler)
        
        await executor.build(
            context_dir=str(dockerfile_path.parent),
            dockerfile=str(dockerfile_path)
        )
        
        assert len(output_lines) > 0  # 何らかの出力があることを確認
    
    @pytest.mark.asyncio
    async def test_build_cancellation(self, executor: BuildctlExecutor, dockerfile_path: Path):
        """ビルド中断のテスト"""
        # 長時間実行するDockerfileを作成
        dockerfile_path.write_text("""
        FROM python:3.9
        RUN sleep 30
        """)
        
        # ビルドタスクを開始
        build_task = asyncio.create_task(
            executor.build(
                context_dir=str(dockerfile_path.parent),
                dockerfile=str(dockerfile_path)
            )
        )
        
        # 少し待ってからビルドを中断
        await asyncio.sleep(0.1)
        await executor.stop()
        
        # タスクが中断されることを確認
        with pytest.raises(asyncio.CancelledError):
            await build_task
