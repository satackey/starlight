"""
Async Helpers Tests

非同期ヘルパー機能のテスト。
以下の項目をテスト：
- タイムアウト処理
- リトライ機能
- 並行処理制御
"""

import asyncio
import time
from typing import List, Optional

import pytest

from build_and_trace_v3.utils.async_helpers import (
    AsyncTimeout,
    with_timeout,
    RetryPolicy,
    with_retry,
    Limiter,
    gather_with_concurrency
)

class TestAsyncTimeout:
    """AsyncTimeoutのテスト"""
    
    @pytest.mark.asyncio
    async def test_no_timeout(self):
        """タイムアウトなしの正常実行テスト"""
        async with AsyncTimeout(seconds=1.0):
            await asyncio.sleep(0.1)
            # 例外が発生しないことを確認
    
    @pytest.mark.asyncio
    async def test_timeout(self):
        """タイムアウト発生のテスト"""
        with pytest.raises(TimeoutError):
            async with AsyncTimeout(seconds=0.1):
                await asyncio.sleep(1.0)
    
    @pytest.mark.asyncio
    async def test_nested_timeout(self):
        """ネストされたタイムアウトのテスト"""
        async with AsyncTimeout(seconds=1.0):
            async with AsyncTimeout(seconds=0.1):
                with pytest.raises(TimeoutError):
                    await asyncio.sleep(0.2)
            
            # 外側のタイムアウトはまだ有効
            await asyncio.sleep(0.1)
    
    @pytest.mark.asyncio
    async def test_timeout_cleanup(self):
        """タイムアウト後のクリーンアップテスト"""
        timeout = AsyncTimeout(seconds=0.1)
        
        with pytest.raises(TimeoutError):
            async with timeout:
                await asyncio.sleep(1.0)
        
        assert timeout._timeout_handle is None
        assert timeout._task is None

class TestWithTimeout:
    """with_timeoutデコレータのテスト"""
    
    @pytest.mark.asyncio
    async def test_decorated_function(self):
        """デコレータ付き関数の実行テスト"""
        @with_timeout(1.0)
        async def test_func():
            await asyncio.sleep(0.1)
            return "success"
        
        result = await test_func()
        assert result == "success"
    
    @pytest.mark.asyncio
    async def test_decorated_function_timeout(self):
        """デコレータ付き関数のタイムアウトテスト"""
        @with_timeout(0.1)
        async def test_func():
            await asyncio.sleep(1.0)
            return "success"
        
        with pytest.raises(TimeoutError):
            await test_func()

class TestRetryPolicy:
    """RetryPolicyのテスト"""
    
    def test_default_values(self):
        """デフォルト値のテスト"""
        policy = RetryPolicy()
        
        assert policy.max_attempts == 3
        assert policy.initial_delay == 1.0
        assert policy.max_delay == 60.0
        assert policy.exponential_base == 2.0
        assert isinstance(policy.exceptions, tuple)
        assert Exception in policy.exceptions
    
    def test_custom_values(self):
        """カスタム値のテスト"""
        policy = RetryPolicy(
            max_attempts=5,
            initial_delay=0.1,
            max_delay=10.0,
            exponential_base=3.0,
            exceptions=ValueError
        )
        
        assert policy.max_attempts == 5
        assert policy.initial_delay == 0.1
        assert policy.max_delay == 10.0
        assert policy.exponential_base == 3.0
        assert policy.exceptions == (ValueError,)
    
    def test_delay_calculation(self):
        """待機時間計算のテスト"""
        policy = RetryPolicy(
            initial_delay=1.0,
            max_delay=10.0,
            exponential_base=2.0
        )
        
        delays = [policy.get_delay(i) for i in range(1, 5)]
        assert delays == [1.0, 2.0, 4.0, 8.0]
        
        # 最大値を超えない
        assert policy.get_delay(10) == 10.0

class TestWithRetry:
    """with_retryデコレータのテスト"""
    
    @pytest.mark.asyncio
    async def test_successful_execution(self):
        """正常実行のテスト"""
        @with_retry()
        async def test_func():
            return "success"
        
        result = await test_func()
        assert result == "success"
    
    @pytest.mark.asyncio
    async def test_retry_on_error(self):
        """エラー時のリトライテスト"""
        attempts = 0
        
        @with_retry(RetryPolicy(initial_delay=0.1))
        async def test_func():
            nonlocal attempts
            attempts += 1
            if attempts < 3:
                raise ValueError("Test error")
            return "success"
        
        result = await test_func()
        assert result == "success"
        assert attempts == 3
    
    @pytest.mark.asyncio
    async def test_max_attempts_exceeded(self):
        """最大試行回数超過のテスト"""
        @with_retry(RetryPolicy(max_attempts=3, initial_delay=0.1))
        async def test_func():
            raise ValueError("Test error")
        
        with pytest.raises(ValueError):
            await test_func()
    
    @pytest.mark.asyncio
    async def test_specific_exception(self):
        """特定の例外のみリトライするテスト"""
        @with_retry(RetryPolicy(
            exceptions=ValueError,
            initial_delay=0.1
        ))
        async def test_func():
            raise KeyError("Different error")
        
        with pytest.raises(KeyError):
            await test_func()

class TestLimiter:
    """Limiterのテスト"""
    
    @pytest.fixture
    def limiter(self) -> Limiter:
        """テスト用のLimiterを提供"""
        return Limiter(max_concurrent=2)
    
    @pytest.mark.asyncio
    async def test_concurrent_execution(self, limiter: Limiter):
        """同時実行のテスト"""
        active_count = 0
        max_active = 0
        
        async def task():
            nonlocal active_count, max_active
            async with limiter:
                active_count += 1
                max_active = max(max_active, active_count)
                await asyncio.sleep(0.1)
                active_count -= 1
        
        tasks = [task() for _ in range(5)]
        await asyncio.gather(*tasks)
        
        assert max_active == 2  # 最大同時実行数
        assert limiter.active == 0  # 全て完了
        assert limiter.total == 5  # 総実行数
    
    @pytest.mark.asyncio
    async def test_error_handling(self, limiter: Limiter):
        """エラー発生時のテスト"""
        with pytest.raises(ValueError):
            async with limiter:
                raise ValueError("Test error")
        
        assert limiter.active == 0  # クリーンアップされている

class TestGatherWithConcurrency:
    """gather_with_concurrencyのテスト"""
    
    @pytest.mark.asyncio
    async def test_concurrent_execution(self):
        """同時実行のテスト"""
        active_count = 0
        max_active = 0
        results: List[int] = []
        
        async def task(i: int) -> int:
            nonlocal active_count, max_active
            active_count += 1
            max_active = max(max_active, active_count)
            await asyncio.sleep(0.1)
            active_count -= 1
            results.append(i)
            return i
        
        tasks = [lambda i=i: task(i) for i in range(5)]
        await gather_with_concurrency(2, *tasks)
        
        assert max_active == 2  # 最大同時実行数
        assert len(results) == 5  # 全タスクが完了
        assert set(results) == set(range(5))  # 全ての結果が含まれる
    
    @pytest.mark.asyncio
    async def test_error_propagation(self):
        """エラー伝播のテスト"""
        async def failing_task():
            raise ValueError("Test error")
        
        with pytest.raises(ValueError):
            await gather_with_concurrency(2, failing_task)
