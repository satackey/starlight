"""
Async Utilities

非同期処理のためのユーティリティモジュール。
以下の機能を実装：
- タイムアウト付き実行
- リトライ機能
- 並行処理の制御
- エラーハンドリング
"""

import asyncio
import functools
import logging
from datetime import datetime, timedelta
from typing import TypeVar, Callable, Optional, Type, Union, List, Any

from .logging import get_logger

logger = get_logger()

T = TypeVar('T')

class AsyncTimeout:
    """
    タイムアウト付き非同期実行のコンテキストマネージャ
    
    使用例:
    ```python
    async with AsyncTimeout(seconds=10):
        await long_running_task()
    ```
    """
    
    def __init__(self, seconds: float):
        """
        Args:
            seconds: タイムアウト時間（秒）
        """
        self.seconds = seconds
        self._task: Optional[asyncio.Task] = None
        self._timeout_handle: Optional[asyncio.Handle] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
    
    async def __aenter__(self):
        """非同期コンテキストマネージャのエントリーポイント"""
        self._loop = asyncio.get_running_loop()
        self._task = asyncio.current_task()
        
        if self._task is None:
            raise RuntimeError("No task found in current context")
        
        self._timeout_handle = self._loop.call_later(
            self.seconds,
            self._task.cancel
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """非同期コンテキストマネージャの終了処理"""
        if self._timeout_handle:
            self._timeout_handle.cancel()
        
        if exc_type is asyncio.CancelledError:
            raise TimeoutError(f"Operation timed out after {self.seconds} seconds")

def with_timeout(seconds: float) -> Callable:
    """
    タイムアウト付き実行のデコレータ
    
    使用例:
    ```python
    @with_timeout(10)
    async def long_running_task():
        ...
    ```
    
    Args:
        seconds: タイムアウト時間（秒）
        
    Returns:
        デコレータ関数
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            async with AsyncTimeout(seconds):
                return await func(*args, **kwargs)
        return wrapper
    return decorator

class RetryPolicy:
    """リトライポリシーの設定"""
    
    def __init__(
        self,
        max_attempts: int = 3,
        initial_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        exceptions: Optional[Union[Type[Exception], List[Type[Exception]]]] = None
    ):
        """
        Args:
            max_attempts: 最大試行回数
            initial_delay: 初回の待機時間（秒）
            max_delay: 最大待機時間（秒）
            exponential_base: 指数バックオフの底
            exceptions: リトライ対象の例外クラス
        """
        self.max_attempts = max_attempts
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        
        if exceptions is None:
            exceptions = Exception
        elif not isinstance(exceptions, (list, tuple)):
            exceptions = [exceptions]
        self.exceptions = tuple(exceptions)
    
    def get_delay(self, attempt: int) -> float:
        """
        待機時間を計算
        
        Args:
            attempt: 試行回数
            
        Returns:
            待機時間（秒）
        """
        delay = self.initial_delay * (self.exponential_base ** (attempt - 1))
        return min(delay, self.max_delay)

def with_retry(
    policy: Optional[RetryPolicy] = None,
    logger: Optional[logging.Logger] = None
) -> Callable:
    """
    リトライ機能付き実行のデコレータ
    
    使用例:
    ```python
    @with_retry(RetryPolicy(max_attempts=3))
    async def flaky_operation():
        ...
    ```
    
    Args:
        policy: リトライポリシー
        logger: ロガー
        
    Returns:
        デコレータ関数
    """
    if policy is None:
        policy = RetryPolicy()
    
    if logger is None:
        logger = get_logger()
    
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            last_exception = None
            
            for attempt in range(1, policy.max_attempts + 1):
                try:
                    return await func(*args, **kwargs)
                    
                except policy.exceptions as e:
                    last_exception = e
                    
                    if attempt == policy.max_attempts:
                        logger.error(
                            f"All {policy.max_attempts} attempts failed for {func.__name__}",
                            exc_info=True
                        )
                        raise
                    
                    delay = policy.get_delay(attempt)
                    logger.warning(
                        f"Attempt {attempt}/{policy.max_attempts} failed for {func.__name__}. "
                        f"Retrying in {delay:.1f} seconds..."
                    )
                    await asyncio.sleep(delay)
            
            raise last_exception
            
        return wrapper
    return decorator

class Limiter:
    """
    並行処理数を制限するクラス
    
    使用例:
    ```python
    limiter = Limiter(max_concurrent=3)
    async with limiter:
        await process_task()
    ```
    """
    
    def __init__(self, max_concurrent: int):
        """
        Args:
            max_concurrent: 最大同時実行数
        """
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.active = 0
        self.total = 0
    
    async def __aenter__(self):
        """非同期コンテキストマネージャのエントリーポイント"""
        await self.semaphore.acquire()
        self.active += 1
        self.total += 1
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """非同期コンテキストマネージャの終了処理"""
        self.active -= 1
        self.semaphore.release()

async def gather_with_concurrency(
    max_concurrent: int,
    *tasks: List[Callable[[], Any]]
) -> List[Any]:
    """
    並行処理数を制限してタスクを実行
    
    使用例:
    ```python
    results = await gather_with_concurrency(3, task1, task2, task3, task4)
    ```
    
    Args:
        max_concurrent: 最大同時実行数
        *tasks: 実行するタスクのリスト
        
    Returns:
        タスクの実行結果のリスト
    """
    limiter = Limiter(max_concurrent)
    
    async def wrapped_task(task: Callable[[], Any]) -> Any:
        async with limiter:
            return await task()
    
    return await asyncio.gather(
        *(wrapped_task(task) for task in tasks)
    )
