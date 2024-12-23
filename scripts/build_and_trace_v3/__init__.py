"""
Build and Trace V3

Dockerfileのビルドプロセスを追跡し、ファイルアクセスパターンを収集・分析するためのツール。
"""

__version__ = '3.0.0'
__author__ = 'Starlight Team'
__license__ = 'Apache License 2.0'

from . import executors
from . import processors
from . import utils

__all__ = ['executors', 'processors', 'utils']
