"""
日志工具模块 - 记录所有关键步骤到llm-log.txt
"""
import datetime
import sys


class Logger:
    """双输出日志记录器：同时输出到控制台和文件"""

    def __init__(self, log_file='llm-log.txt'):
        self.log_file = log_file
        self.terminal = sys.stdout

        # 清空或创建日志文件
        with open(self.log_file, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write(f"日志开始时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n\n")

    def log(self, message, level='INFO'):
        """
        记录日志消息

        Args:
            message: 日志消息
            level: 日志级别 (INFO, DEBUG, WARNING, ERROR)
        """
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        formatted_message = f"[{timestamp}] [{level}] {message}"

        # 输出到控制台
        print(formatted_message)

        # 输出到文件
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(formatted_message + '\n')

    def info(self, message):
        """记录INFO级别日志"""
        self.log(message, 'INFO')

    def debug(self, message):
        """记录DEBUG级别日志"""
        self.log(message, 'DEBUG')

    def warning(self, message):
        """记录WARNING级别日志"""
        self.log(message, 'WARNING')

    def error(self, message):
        """记录ERROR级别日志"""
        self.log(message, 'ERROR')

    def section(self, title):
        """记录分节标题"""
        separator = "=" * 80
        message = f"\n{separator}\n{title}\n{separator}"
        print(message)
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(message + '\n')

    def subsection(self, title):
        """记录子节标题"""
        separator = "-" * 60
        message = f"\n{separator}\n{title}\n{separator}"
        print(message)
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(message + '\n')


# 全局日志记录器实例
_global_logger = None


def get_logger():
    """获取全局日志记录器"""
    global _global_logger
    if _global_logger is None:
        _global_logger = Logger()
    return _global_logger


def reset_logger():
    """重置日志记录器"""
    global _global_logger
    _global_logger = Logger()
    return _global_logger
