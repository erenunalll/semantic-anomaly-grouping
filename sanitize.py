import re

### Blacklist ###
ISO_TIMESTAMP = re.compile(
    r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:[.,]\d+)?(?:Z|[+-]\d{2}:\d{2})?"
)
SYSLOG_TIMESTAMP = re.compile(
    r"\b(?:(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun)\s+)?"
    r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)"
    r"\s+\d{1,2}\s+\d{2}:\d{2}:\d{2}"
    r"(?:\s+\d{4})?\b"
)
COMPACT_TIMESTAMP = re.compile(
    r"\b\d{6}\s+\d{6}\b"
)
DOT_TIMESTAMP = re.compile(
    r"\b\d+\.\d{2}\.\d+\b"
)
DOT_TIMESTAMP_MS = re.compile(
    r"\b\d{4}-\d{2}-\d{2}-\.\d+\b"
)
TIME = re.compile(
    r"\b\d{2}:\d{2}:\d{2}(?:\.\d+)?\b"
)
IDENTIFIER_FIELDS = (
    r"trace_id|request_id|session_id|user_id|"
    r"correlation_id|span_id|parent_span_id|"
    r"uuid|guid|pid|tid|uid|gid"
)
ID = re.compile(
    rf"(?i)\b(?:{IDENTIFIER_FIELDS})\b\s*[:=]\s*\S+", flags=re.IGNORECASE
)
MANY_DIGITS = re.compile(
    r"\b\d{10,}\b"
)
UUID = re.compile(
    r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"
)
HEX = re.compile(
    r"0x[0-9a-fA-F]+"
)
REF = re.compile(
    r"\b\w+@\d+\b"
)
IP = re.compile(
    r"\b(?:\d{1,3}\.){3}\d{1,3}(?::\d{1,5})?\b"
)
PARTIAL_IP = re.compile(
    r"(?<![\w-])(?:\d{1,3}\.){1,3}\d{0,3}(?![\w-])|(?<![\w-])\.\d{1,3}(?![\w-])"
)
UNIX_PATH_RE = re.compile(
    r"(?:/[A-Za-z0-9._-]+)+"
)

WINDOWS_PATH_RE = re.compile(
    r"[A-Za-z]:\\(?:[^\\/:*?\"<>|\r\n]+\\)*[^\\/:*?\"<>|\r\n]*"
)
FILE_RE = re.compile(
    r"\b[\w.-]+\.(?:c|cc|cpp|cxx|h|hpp|java|py|go|rs|js|ts|cs)\b",
    flags=re.IGNORECASE,
)
BRACKETS = re.compile(
    r"\[[^\[\]]*\]"
)
SPACES = re.compile(
    r"\s+"
)
DASH = re.compile(
    r"(^\s*-\s*)|(\s+-\s+)"
)
JAVA_LOGGER = re.compile(
    r"\b(?:[a-zA-Z_]\w*\.){2,}[A-Z]\w*\b"
)
NODE = re.compile(
    r"\bR\d{2}-M\d-N[A-Z0-9]+-[A-Z]:J\d{2}-U\d{2}\b"
)
KEY_VALUE = re.compile(
    r"\b\w+=\S*"
)
PORT = re.compile(
    r"\bport\s+\d+\b",
    flags=re.IGNORECASE
)
ERR_WARN = re.compile(
    r"\b(?:ERROR|WARN|WARNING)\b"
)

def sanitize(text: str) -> str:
    """
    Applies sanitization to given text by cleaning common dynamic noise in error logs.
    """

    text = BRACKETS.sub(" ", text)

    text = IP.sub("<IP>", text)
    text = ISO_TIMESTAMP.sub("<TIME>", text)
    text = SYSLOG_TIMESTAMP.sub("<TIME>", text)
    text = COMPACT_TIMESTAMP.sub("<TIME>", text)
    text = DOT_TIMESTAMP.sub("<TIME>", text)
    text = DOT_TIMESTAMP_MS.sub("<TIME>", text)
    text = TIME.sub("<TIME>", text)

    text = KEY_VALUE.sub("", text)

    text = WINDOWS_PATH_RE.sub("", text)
    text = UNIX_PATH_RE.sub("", text)

    text = ID.sub("", text)
    text = REF.sub("", text)
    text = HEX.sub("<HEX>", text)
    text = UUID.sub("<UUID>", text)
    text = PORT.sub("<PORT>", text)
    text = PARTIAL_IP.sub("<IP>", text)

    text = JAVA_LOGGER.sub("", text)
    text = NODE.sub("", text)

    text = MANY_DIGITS.sub("", text)
    text = ERR_WARN.sub("", text)
    text = DASH.sub(" ", text)
    text = SPACES.sub(" ", text)
    text = text.strip()

    return text

if __name__ == "__main__":
    pass