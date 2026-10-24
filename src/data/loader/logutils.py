import enum

ANSI_RED = "\033[91m"
ANSI_GREEN = "\033[92m"
ANSI_YELLOW = "\033[93m"
ANSI_PINK = "\033[95m"
ANSI_DARK_RED = "\033[31m"
ANSI_BLUE = "\033[34m"
ANSI_RESET = "\033[0m"


class LogLevel(enum.Enum):
    VERBOSE = 0
    DEBUG = 10
    INFO = 20
    WARNING = 30
    ERROR = 40
    CRITICAL = 50
    NONE = 60


_log_level = LogLevel.INFO


def set_log_level(level: LogLevel) -> None:
    global _log_level
    _log_level = level


def _logprint(str: str, level: LogLevel) -> None:
    global _log_level
    if level.value < _log_level.value:
        return
    match level:
        case LogLevel.VERBOSE:
            print(f"{ANSI_BLUE}[VERBOSE]{ANSI_RESET} {str}")
        case LogLevel.DEBUG:
            print(f"{ANSI_PINK}[DEBUG]{ANSI_RESET} {str}")
        case LogLevel.INFO:
            print(f"{ANSI_GREEN}[INFO]{ANSI_RESET} {str}")
        case LogLevel.WARNING:
            print(f"{ANSI_YELLOW}[WARNING]{ANSI_RESET} {str}")
        case LogLevel.ERROR:
            print(f"{ANSI_RED}[ERROR]{ANSI_RESET} {str}")
        case LogLevel.CRITICAL:
            print(f"{ANSI_DARK_RED}[CRITICAL]{ANSI_RESET} {str}")
        case _:
            raise ValueError("Invalid log level.")


def verbose(str: str) -> None:
    _logprint(str, LogLevel.VERBOSE)


def debug(str: str) -> None:
    _logprint(str, LogLevel.DEBUG)


def info(str: str) -> None:
    _logprint(str, LogLevel.INFO)


def warn(str: str) -> None:
    _logprint(str, LogLevel.WARNING)


def error(str: str) -> None:
    _logprint(str, LogLevel.ERROR)


def critical(str: str) -> None:
    _logprint(str, LogLevel.CRITICAL)
