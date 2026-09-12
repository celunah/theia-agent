"""Limits, validation patterns, and policy data used by the App Server."""

import re


MAX_ATTACHMENT_BYTES = 32 * 1024 * 1024
MAX_ATTACHMENT_TEXT_BYTES = 100 * 1024
MAX_ATTACHMENTS_PER_REQUEST = 10
MAX_ATTACHMENT_BATCH_BYTES = 64 * 1024 * 1024
IMAGE_SUFFIXES = frozenset({".gif", ".jpeg", ".jpg", ".png", ".webp"})
MESSAGE_LEDGER_LIMIT = 2000
MESSAGE_LEDGER_RETRY_AFTER = 15 * 60
CHANNEL_CHECKPOINT_LIMIT = 200
MEMORY_FILE_LIMIT = 64 * 1024
MEMORY_SNAPSHOT_LIMIT = 256 * 1024

# The app-server protocol is newline-delimited JSON. Restoring a thread can
# produce a single event containing a large persisted item, which exceeds
# asyncio's default 64 KiB StreamReader limit. Keep a finite ceiling while
# allowing those history events through.
DEFAULT_CODEX_STDIO_LIMIT = 16 * 1024 * 1024
MIN_CODEX_STDIO_LIMIT = 64 * 1024
MAX_CODEX_STDIO_LIMIT = 64 * 1024 * 1024
CODEX_STDIO_LIMIT_ENV = "THEIA_CODEX_STDIO_LIMIT"
CODEX_MEMORY_WATCHDOG_ENV = "THEIA_CODEX_MEMORY_WATCHDOG"
CODEX_MAX_RSS_MB_ENV = "THEIA_CODEX_MAX_RSS_MB"
CODEX_MEMORY_CHECK_INTERVAL_ENV = "THEIA_CODEX_MEMORY_CHECK_INTERVAL"
CODEX_MEMORY_RESTART_GRACE_ENV = "THEIA_CODEX_MEMORY_RESTART_GRACE"
CODEX_MEMORY_BREACH_SAMPLES_ENV = "THEIA_CODEX_MEMORY_BREACH_SAMPLES"
SESSION_ARCHIVE_AFTER = 30 * 24 * 60 * 60
SESSION_DELETE_AFTER = 90 * 24 * 60 * 60
DEFAULT_ATTACHMENT_CACHE_LIMIT_BYTES = 512 * 1024 * 1024
DEFAULT_ATTACHMENT_CACHE_MAX_AGE = SESSION_DELETE_AFTER
DEFAULT_CODEX_MEMORY_WATCHDOG = True
DEFAULT_CODEX_MAX_RSS_MB = 6144.0
DEFAULT_CODEX_MEMORY_CHECK_INTERVAL = 15.0
DEFAULT_CODEX_MEMORY_RESTART_GRACE = 10.0
DEFAULT_CODEX_MEMORY_BREACH_SAMPLES = 3
WEB_SEARCH_ENV = "THEIA_WEB_SEARCH"
WEB_SEARCH_MODES = frozenset({"disabled", "indexed", "live"})

_APPROVAL_RISK_SAFE = "safe"
_APPROVAL_RISK_DANGEROUS = "dangerous"
_APPROVAL_RISK_VERY_DANGEROUS = "very_dangerous"
_APPROVAL_SAFE_COMMAND_RE = re.compile(
    r"^\s*(?:pwd|ls|find|rg|grep|head|tail|file|stat|"
    r"git\s+(?:status|diff|log|show|branch|rev-parse))\b",
    re.IGNORECASE,
)
_APPROVAL_VERY_DANGEROUS_RE = re.compile(
    r"(?:"
    r"\b(?:rm|rmdir|del|erase|sudo|su|doas|chmod|chown|chgrp|mkfs|dd|"
    r"shutdown|reboot|poweroff|kill|pkill|killall|mount|umount)\b|"
    r"\bgit\s+(?:reset|clean|push|checkout|restore|rebase|commit|merge|"
    r"apply|config)\b|"
    r"\b(?:delete|destroy|wipe|drop|truncate)\b|"
    r"\b(?:password|secret|token|credential|private\s+key)\b|"
    r"\b(?:bash|sh|zsh|fish|cmd|powershell|pwsh|python|python3|node|perl|"
    r"ruby|php|curl|wget|ssh|scp|rsync|nc|ncat|docker|make|cargo|go|npm|"
    r"npx|pip)\b|"
    r"(?:\.env\b|\.ssh\b|\.aws\b|/etc/(?:shadow|passwd)\b)|"
    r"(?:&&|\|\||[;|<>]|`|\$\()"
    r")",
    re.IGNORECASE,
)
_APPROVAL_PATH_RE = re.compile(
    r"(?<![A-Za-z0-9_])(?:file://)?/[\w.@%+~=:/-]+"
    r"|(?<![A-Za-z0-9_])[A-Za-z]:[\\/][^\s,;]+",
    re.IGNORECASE,
)

_SELF_IMPROVEMENT_MAX_UPDATES = 4
_SELF_IMPROVEMENT_MAX_UPDATE_BYTES = 4096
_SELF_IMPROVEMENT_MAX_TOTAL_BYTES = 16 * 1024
_SELF_IMPROVEMENT_SUMMARY_MAX_BYTES = 8 * 1024
_SELF_IMPROVEMENT_SUMMARY_ITEM_MAX_CHARACTERS = 800
_SELF_IMPROVEMENT_MAX_FILE_BYTES = 512 * 1024
_SELF_IMPROVEMENT_SKILL_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,79}$")
_PERSONALITY_SESSION_KEY_RE = re.compile(
    r"^guild:(?P<guild>[^:]+):channel:[^:]+:user:(?P<user>[^:]+)$"
)
_PERSONALITY_SCOPE_KEY_RE = re.compile(r"^(?:me|server):[1-9][0-9]*$|^everyone$")

_MOOD_MAX_CAUSES = 3
_MOOD_CAUSE_MAX_CHARACTERS = 180
_MOOD_TRAITS_MAX_CHARACTERS = 180
DEFAULT_MOOD_CLASSIFICATION_TIMEOUT = 5.0
DEFAULT_ATTENTION_CLASSIFICATION_TIMEOUT = 1.0
DEFAULT_WORKSPACE_REVIEW_TIMEOUT = 12.0
WORKSPACE_ENTRY_CATEGORIES = frozenset(
    {"goal", "decision", "constraint", "open_question", "context_note"}
)
WORKSPACE_MAX_ENTRIES = 24
WORKSPACE_ENTRY_MAX_CHARACTERS = 320
WORKSPACE_TOTAL_MAX_CHARACTERS = 8 * 1024
WORKSPACE_REVIEW_CONTEXT_MAX_CHARACTERS = 6000
WORKSPACE_KEY_MAX_CHARACTERS = 64
CONVERSATION_RELATIONS = (
    "CONTINUE",
    "RELATED_EXTENSION",
    "SIDETRACK",
    "TOPIC_SHIFT",
    "OFF_TOPIC",
    "RETURN",
    "NESTED_RETURN",
    "CLARIFICATION",
    "END",
)
ATTENTION_CONTEXT_LIMIT = 12
ATTENTION_RECENT_EXCHANGE_LIMIT = 6
ATTENTION_PARKED_METADATA_LIMIT = 8
ATTENTION_GLOBAL_MESSAGE_LIMIT = 6
ATTENTION_OPEN_LOOP_LIMIT = 3
ATTENTION_HISTORY_LIMIT = 12
ATTENTION_TITLE_MAX_CHARACTERS = 120
ATTENTION_SUMMARY_MAX_CHARACTERS = 480
ATTENTION_EXCHANGE_MAX_CHARACTERS = 500
ATTENTION_REASON_MAX_CHARACTERS = 180
_USAGE_DAILY_LIMIT = 400
_TOKEN_USAGE_KEYS = (
    "cacheWriteInputTokens",
    "cachedInputTokens",
    "inputTokens",
    "outputTokens",
    "reasoningOutputTokens",
    "totalTokens",
)
_MOOD_TRIVIAL_MESSAGES = frozenset(
    {
        "hi",
        "hello",
        "hey",
        "ok",
        "okay",
        "k",
        "yes",
        "no",
        "sure",
        "got it",
        "thanks",
        "thank you",
    }
)

_SELF_IMPROVEMENT_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "updates": {
            "type": "array",
            "maxItems": _SELF_IMPROVEMENT_MAX_UPDATES,
            "items": {
                "type": "object",
                "properties": {
                    "kind": {
                        "type": "string",
                        "enum": ["memory", "user_profile", "skill", "personality"],
                    },
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["kind", "path", "content"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["updates"],
    "additionalProperties": False,
}

_CODEX_CHILD_SECRET_ENV_NAMES = frozenset(
    {
        "TOKEN",
        "DISCORD_TOKEN",
        "THEIA_DISCORD_TOKEN",
        "STT_TOKEN",
        "THEIA_TRANSCRIPTION_API_KEY",
        "TTS_TOKEN",
        "THEIA_TTS_API_KEY",
        "OPENAI_API_KEY",
        "CODEX_API_KEY",
    }
)
TEXT_ATTACHMENT_SUFFIXES = frozenset(
    {
        ".c",
        ".cc",
        ".cpp",
        ".csv",
        ".go",
        ".h",
        ".hpp",
        ".html",
        ".ini",
        ".java",
        ".json",
        ".js",
        ".jsx",
        ".md",
        ".markdown",
        ".py",
        ".rs",
        ".sh",
        ".sql",
        ".text",
        ".toml",
        ".ts",
        ".tsx",
        ".txt",
        ".xml",
        ".yaml",
        ".yml",
    }
)
AUDIO_ATTACHMENT_SUFFIXES = frozenset(
    {
        ".aac",
        ".flac",
        ".m4a",
        ".mp3",
        ".mp4",
        ".mpeg",
        ".mpga",
        ".ogg",
        ".wav",
        ".webm",
    }
)

_PERSONALITY_SUMMARY_SOURCE_LIMIT = 16 * 1024
_PERSONALITY_SUMMARY_TIMEOUT = 15.0
_MEMORY_RETRIEVAL_SOURCE_LIMIT = 32 * 1024
_MEMORY_RETRIEVAL_REQUEST_LIMIT = 12 * 1024
_MEMORY_RETRIEVAL_TIMEOUT = 8.0
_MEMORY_RETRIEVAL_HINT_RE = re.compile(
    r"\b(?:remember|memory|previous|earlier|last\s+time|before|again|"
    r"discuss(?:ed|ion)|history|known|what\s+did\s+we|who\s+did)\b",
    re.IGNORECASE,
)
_MEMORY_ENTRY_RE = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+\S")
_MEMORY_USER_ID_RE = re.compile(
    r"<@!?([0-9]+)>|(?:discord\s+user\s+id|user_id)\s*[:=]\s*([0-9]+)",
    re.IGNORECASE,
)
