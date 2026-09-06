# Theia Agent

Theia Agent is a Discord bot that brings private, persistent Codex
conversations to Discord. It runs the Codex App Server locally and keeps each
user, channel, and thread in an isolated session.

The current release is `1.0.1`.

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
- Model-generated Discord Rich Presence with debounced task updates and slower
  idle refreshes.
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
Successful `/restart`, `/model`, and `/customize` confirmations are public;
account, session, authentication, and permission-sensitive responses remain
ephemeral. `/stop`, `/btw`, and successful `/skill` responses are already
public conversation or action results.
`/about` privately shows the running Theia version and revision, selected
Codex CLI version, invoking Discord account, Codex plan, and current session
mode and personality.
Independent message, slash-command, and voice turns run as tracked background
tasks, so a long-running agentic action does not block other Discord requests.
A bare mention of Theia prompts a response to the bounded recent channel
context.
Ordinary conversation uses a spoken-first cadence with direct acknowledgment,
short natural paragraphs, and concrete progress updates. Code, reviews,
procedures, and explicit detail requests remain complete and can expand as needed.
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
mode) the OpenAI-compatible STT/TTS URLs, optional service tokens, and voice
parameters. It writes
the local ignored `.env` atomically. You can also create `.env` from
`.env.example` and edit it manually.

Start the bot with:

```bash
uv run python main.py
```

### Container deployment

The repository includes a Docker-compatible image and a Compose definition.
They work with Docker or Podman because the setup uses a standard Dockerfile
and does not require an exposed port; Theia connects to Discord outbound.

Create the ignored environment file and set at least `TOKEN`:

```bash
cp .env.example .env
```

Build and start Theia with Docker Compose:

```bash
docker compose up --build -d
docker compose logs -f theia
```

Podman Compose uses the same commands:

```bash
podman compose up --build -d
podman compose logs -f theia
```

The `theia-data` volume persists Theia's private Codex authentication,
sessions, attachments, memories, personalities, and skills. The default
`theia-workspace` volume is the administrator Codex working directory. To use
a host directory instead, replace that volume in `compose.yaml` with
`./workspace:/workspace`. Set `THEIA_SAFE_WORKSPACE_ROOTS=/workspace` in `.env`
only when regular users should receive read-only access to that directory.

The first `/login` in a new deployment may require Codex's device-code flow;
subsequent container replacements reuse the authentication stored in
`theia-data`. Keep `.env` out of image builds and source control. The image
also includes `ffmpeg` and the system Opus library for the optional voice mode.

For a direct engine invocation, use the same image and mount the persistent
volumes explicitly:

```bash
docker build --build-arg THEIA_COMMIT="$(git rev-parse --short=7 HEAD)" -t theia-agent:1.0.1 .
docker run -d --name theia-agent --restart unless-stopped \
  --env-file .env \
  -v theia-data:/data \
  -v theia-workspace:/workspace \
  theia-agent:1.0.1
```

Replace `docker` with `podman` for a Podman deployment. The image runs as a
non-root `theia` user. Rootless Podman users who bind-mount a host workspace
may need to adjust the host directory ownership or use a named volume.

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
The build embeds the short Git revision used to produce the executable so
`/about` remains accurate after extraction.

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
THEIA_ALWAYS_ADMIN_USERS=      # comma-delimited trusted Discord user IDs
THEIA_SELF_IMPROVEMENT=true    # review completed admin turns for durable updates
THEIA_SELF_IMPROVEMENT_TIMEOUT=90
THEIA_RICH_PRESENCE_ENABLED=true
THEIA_RICH_PRESENCE_ACTIVE_DEBOUNCE=3
THEIA_RICH_PRESENCE_IDLE_INTERVAL=900
THEIA_RICH_PRESENCE_RECENT_IDLE_INTERVAL=600
THEIA_RICH_PRESENCE_CONTEXT_MAX_AGE=1800
THEIA_RICH_PRESENCE_TIMEOUT=8
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
STT_PROTOCOL=openai-compatible
STT_BASE_URL=
STT_TOKEN=
STT_MODEL=whisper-1
TTS_PROTOCOL=openai-compatible
TTS_BASE_URL=
TTS_TOKEN=
TTS_MODEL=tts-1
TTS_VOICE=alloy
TTS_FORMAT=mp3
```

The setup wizard writes the custom provider's STT model, TTS model, TTS voice,
and TTS format. Advanced protocol aliases remain available through the
corresponding environment variables.

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
Set `THEIA_ALWAYS_ADMIN_USERS` to a comma-delimited list of trusted Discord user
IDs to grant those users Theia administrator access even without Discord server
administrator permission. This is a global deployment override and does not
replace the normal Codex login requirement.

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
