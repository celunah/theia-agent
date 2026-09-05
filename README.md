# Theia Agent

Theia Agent is a Discord bot that brings private, persistent Codex
conversations to Discord. It runs the Codex App Server locally and keeps each
user, channel, and thread in an isolated session.

The current release is `1.0.0`.

## Highlights

- Text conversations in DMs and guild channels, with optional response threads
  and bounded recent-channel context.
- Session recovery, `/undo`, inactivity retention, approvals, and sequential
  questions.
- Dynamic Codex model selection, adaptive reasoning, native web search, skills,
  memories, and Markdown personality profiles.
- Attachments for images, audio, and text-like files.
- Optional voice mode using OpenAI-compatible speech-to-text and text-to-speech
  services.
- Server-scoped Discord presentation customization and administrator controls.
- Read-only/safe tool access for regular users; administrators receive the
  configured full Codex policy.

## Discord commands

```text
/login       /about       /usage       /credits
/approve     /deny        /stop        /undo
/btw         /skill       /personality /model
/mode        /restart     /customize
```

Prefix commands are disabled. `/restart` and `/customize` are
administrator-only. `/mode voice` requires the voice dependencies and audio
services described below.
`/customize` also covers pagination controls, input modal labels, and embed
field labels in the account, usage, credits, and About views.
`/about` privately shows the running Theia version and revision, selected
Codex CLI version, invoking Discord account, Codex plan, and current session
mode and personality.
The default Codex model is `gpt-5.6-luna`; `/model` can select another
available model, and changing it starts the next request in a fresh
conversation.

## Installation

Requirements:

- Python 3.10 or newer
- [uv](https://docs.astral.sh/uv/)
- Node.js 16 or newer with npm
- A Discord bot token

From the project directory:

```bash
python scripts/bootstrap.py
```

This installs the Python dependencies and the project-local Codex CLI from the
checked-in `package-lock.json`. Run the setup wizard before the first launch:

```bash
python scripts/configure.py
```

It asks for text or voice mode, then collects the Discord token and (for voice
mode) the OpenAI-compatible STT/TTS URLs and optional service tokens. It writes
the local ignored `.env` atomically. You can also create `.env` from
`.env.example` and edit it manually.

Start the bot with:

```bash
uv run python main.py
```

An optional one-file executable can be built with Nuitka:

```bash
uv run --with nuitka python scripts/build_nuitka.py
```

This removes the need for a loose Python `.venv`; the executable is `dist/theia`
(`dist/theia.exe` on Windows). It still starts Codex as a child process and
therefore needs Node.js plus the bootstrapped or explicitly configured Codex CLI.
Nuitka and a supported platform compiler are build-time requirements.
`/restart` re-executes the original one-file binary, including after its
temporary extraction directory has been cleaned up.

Keep `.env` private. Theia stores its Codex login, session state, cached
attachments, personalities, and installed skills in a private home at
`~/.theia` by default. Use `THEIA_HOME` and `THEIA_STATE` to relocate that
home or state file. Theia reuses valid private authentication, imports another
Codex cache atomically when needed, and starts device-code login only when no
usable cache exists.

The source launcher reads `.env` from the working directory or project root.
The compiled one-file executable also searches for `.env` beside itself and in
its parent directory. Keep the file beside the executable when launching it
from another location; secrets are loaded at runtime and are never bundled.

## Configuration

The most useful optional settings are:

```dotenv
# Codex and runtime
THEIA_DEFAULT_MODE=text        # text or voice for new sessions
THEIA_APPROVAL_LEVEL=high      # high, medium, or low approval handling
THEIA_SELF_IMPROVEMENT=true    # review completed admin turns for durable updates
THEIA_SELF_IMPROVEMENT_TIMEOUT=90
THEIA_CODEX_CLI=
THEIA_HOME=
THEIA_STATE=
THEIA_WEB_SEARCH=indexed       # disabled, indexed, or live
THEIA_SAFE_WORKSPACE_ROOTS=     # optional extra read-only roots for regular users
THEIA_ATTACHMENT_CACHE_LIMIT_BYTES=536870912
CODEX_ADAPTIVE_REASONING=true

# Discord behavior
THEIA_REQUIRE_MENTION=true
THEIA_THREAD_REQUIRE_MENTION=true
THEIA_AUTO_THREAD=true
THEIA_CONTEXT_MESSAGES=12
THEIA_CONTEXT_MAX_CHARACTERS=8000

# Optional OpenAI-compatible audio services
STT_BASE_URL=
STT_TOKEN=
STT_MODEL=whisper-1
TTS_BASE_URL=
TTS_TOKEN=
TTS_MODEL=tts-1
TTS_VOICE=alloy
TTS_FORMAT=mp3
```

Voice mode also needs a system Opus library and `ffmpeg` on `PATH`. The audio
integrations remain disabled when their base URLs are empty.

## Security and privacy

Theia does not send ChatGPT credentials, raw tool calls, paths, or sensitive
payloads to Discord. Approvals are bound to the requesting Discord user,
session, turn, and item. Runtime diagnostics are sanitized by default.

Regular users are restricted to read-only/safe Codex tools and uploaded
attachments by default. Add only explicitly shared, non-sensitive directories
to `THEIA_SAFE_WORKSPACE_ROOTS`. Server administrators can use the configured
full tool policy, including approvals and the guarded Discord thread-creation
tool. Approval handling defaults to High (always request). Medium automatically
accepts safe requests, while Low automatically accepts most requests and keeps
very dangerous requests manual. File changes, permission changes, and
out-of-workspace or destructive commands remain manual. Codex approval requests
are surfaced as Discord embeds; requests that cannot be actionable under the
current policy are reported visibly and declined. Personality profiles are
style-only and cannot authorize source or configuration changes. After a
successful administrator turn, Theia performs a private, read-only review when
self-improvement is enabled. The review can append or create bounded entries in
Theia's private `MEMORY.md` and `USER.md`, create or update a direct-child
`skills/<name>/SKILL.md`, and refine the active personality profile. It cannot
write source code, configuration, authentication, session state, Git metadata,
or arbitrary workspace files. Set `THEIA_SELF_IMPROVEMENT=false` to disable it.
Applied changes are reported in Discord with compact statuses such as `Memory
created`, `Skill updated`, and `Personality updated`.

## Development

Install development dependencies and run the complete local CI contract:

```bash
uv sync --dev
uv run python scripts/run_ci.py
```

The contract checks Ruff, formatting, Pylint, Pyrefly, and the complete pytest
suite. For focused iteration, use `uv run poe test_basic` or
`uv run poe test_changed`.

For the fuller capability inventory and current app-server coverage, see
[`THEIA_FEATURES.md`](THEIA_FEATURES.md).
