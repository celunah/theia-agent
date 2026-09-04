# Theia Agent

Theia Agent runs the Codex App Server with a private runtime home by default:

```text
~/.theia
```

## Installation

The Python application and Codex CLI are installed locally in the project. A
system-wide Codex CLI is not required:

```bash
python scripts/bootstrap.py
```

The bootstrap command runs `uv sync` and then `npm ci` from the checked-in
`package-lock.json`. Theia prefers the resulting project-local `codex`
executable and only uses a system installation as a compatibility fallback.
Use `THEIA_CODEX_CLI` to select a specific CLI executable explicitly.

This keeps Theia-installed skills and Theia-specific Codex configuration out
of the global `~/.codex` directory. Set `THEIA_HOME` to choose a different
private home, or `THEIA_STATE` to choose a different session-state file. The
legacy `CODEX_DISCORD_HOME` and `CODEX_DISCORD_STATE` overrides are still
supported.
Theia's Codex login is stored in its private home. If a global Codex login
already exists, Theia bootstraps a private copy on first startup without
modifying the global configuration; later restarts reuse the private copy.
When a server administrator completes `/login` in a server, the persisted
Codex authorization is granted to all members of that server. Other servers
and direct messages remain separately gated until authorized.

Global memory and skills remain discoverable as shared inputs. The standard
installer uses Theia's `CODEX_HOME` and therefore installs into the private
home:

```text
~/.theia/skills/<skill-name>/SKILL.md
```

Project-local skills are also discovered from:

```text
<project>/.agents/skills/
<project>/.codex/skills/
```

Theia also discovers Hermes roots from `HERMES_HOME` (default `~/.hermes`),
alongside the global Codex roots. Use `CODEX_MEMORY_ROOTS` and
`CODEX_SKILL_ROOTS` for additional roots. Paths are separated using the
platform path separator (`:` on Linux/macOS). Explicitly set
`THEIA_INCLUDE_GLOBAL_MEMORY=true` if the global Codex memory snapshot should
be injected into new threads; it remains available as a workspace root by
default.

Each text, `/btw`, `/skill`, and voice request receives a bounded snapshot of
recent messages from its current Discord channel, ordered oldest to newest.
Theia excludes its compact `-#` progress messages and continues without
channel context when Discord history is unavailable. Configure the bounds with
`THEIA_CONTEXT_MESSAGES` (default `12`, maximum `30`) and
`THEIA_CONTEXT_MAX_CHARACTERS` (default `8000`, maximum `16000`).

## Discord conversations

Direct messages are accepted without a mention. Guild channels require a
mention by default, while threads in which the bot has participated can
continue without repeated mentions. These controls can be configured with
`THEIA_REQUIRE_MENTION`, `THEIA_THREAD_REQUIRE_MENTION`,
`THEIA_FREE_RESPONSE_CHANNELS`, and `THEIA_AUTO_THREAD` (the corresponding
`DISCORD_*` names are also accepted). The bot stores bounded channel
checkpoints and request IDs in the private state file to recover missed
messages after a gateway resume and suppress duplicate deliveries.

Server administrators have the configured full Codex tool policy. Other users
may use read-only/safe tools, but cannot modify files, run state-changing
commands, send Discord messages through Codex, or perform external side
effects. This policy is applied when the Codex thread starts and a role-policy
change starts a fresh thread.

`/model` uses Codex model autocomplete to select the active model. Reasoning
remains adaptive and is not manually overridden.

`/undo` removes the most recent completed Codex turn for the current Discord
session. Sessions inactive for 30 days are archived automatically; activity
unarchives them and continues the conversation. Sessions inactive for 90 days
are deleted automatically; later activity starts a new conversation.

Messages and `/btw` requests may include an optional attachment. Theia caches
attachments privately, sends images/audio through Codex local-input types, and
includes bounded text content for text-like files. Generated files located in
approved runtime/workspace roots can be sent through Codex's Discord tool.

## Transcription and TTS

Theia can use separate OpenAI-compatible audio servers. Both integrations are
disabled when their base URL is empty. Set each base URL to the server's API
base (for example, `https://audio.example/v1`); Theia calls the standard
`/audio/transcriptions` and `/audio/speech` endpoints without adding `/v1`
automatically:

```text
STT_BASE_URL=https://transcription.example/v1
STT_TOKEN=                              # optional for local servers
STT_MODEL=whisper-1

TTS_BASE_URL=https://tts.example/v1
TTS_TOKEN=                              # optional for local servers
TTS_MODEL=tts-1
TTS_VOICE=alloy
TTS_FORMAT=mp3
```

The optional `STT_PROTOCOL`, `STT_MODEL`, `TTS_PROTOCOL`, `TTS_MODEL`,
`TTS_VOICE`, and `TTS_FORMAT` settings select compatible server details. The
older `THEIA_TRANSCRIPTION_*` and `THEIA_TTS_*` names remain accepted as
aliases.

Audio attachments are transcribed and the transcript is provided to Codex in
addition to the cached audio input. Completed responses are kept as normal
Discord text and, when TTS is configured, also include generated audio
attachments. Long responses are split into separate audio attachments to stay
within the standard speech input limit. TTS failures do not discard a
successful text response; transcription failures are reported as request
failures.

Use `/mode voice` to enable voice mode for the current Discord session. The
user must already be in a voice channel, and both `STT_BASE_URL` and
`TTS_BASE_URL` must be configured. Voice mode listens through the optional
`discord-ext-voice-recv` adapter, labels speech with the Discord speaker when
available, sends transcripts through the same Codex session, and plays
intermediate and final responses back into the voice channel. `/mode text`
returns the session to the default text-only behavior.

Voice receive also requires a system Opus library supported by discord.py. TTS
playback requires `ffmpeg` to be available on `PATH`; if either dependency is
missing, `/mode voice` reports that voice mode cannot be started.

## Personalities

The `/personality` command manages Markdown or UTF-8 text personality prompts:

```text
/personality file:<prompt.md> name:<profile-name>  upload and activate
/personality name:<profile-name>                  switch profiles
/personality name:none                            clear the active profile
```

The `name` option uses Discord autocomplete for profiles already stored in the
private Theia home. Providing only a file is rejected; providing no options
shows the command usage. The selected profile is kept per Discord session and
persisted across bot restarts.
The `none`, `default`, and `neutral` names clear the active profile.

Codex receives one consistent system instruction before the user request: the
base priors first, followed by the bounded memory snapshot and selected
personality. Changing or clearing
the personality resets that Discord session so old and new instructions are
never mixed in one Codex thread.

## Adaptive reasoning

Adaptive reasoning is enabled by default. Before each request, Codex runs a
hidden, ephemeral pre-assessment to decide whether external tools are needed
and how complex the task is. Requests that do not need a tool use the model's
lowest supported reasoning level (`low`, `light`, or `minimal`). Tool-backed
requests use `medium`, `high`, `xhigh`, or `max` when the active model supports
them. If the assessment or model capability data is unavailable, the request
uses `medium`.

Set `CODEX_ADAPTIVE_REASONING=false` to disable the pre-assessment and use the
`medium` default for every request. The assessment itself is not shown in
Discord and does not modify the user's Codex thread.

## Web search

Theia uses Codex's native hosted web-search tool. On startup it adds
`web_search = "indexed"` to the private `config.toml` when no search mode has
been configured. Codex then uses its search index to decide when external live
web access is appropriate. Set `THEIA_WEB_SEARCH=live` to always prefer live
results, or `THEIA_WEB_SEARCH=disabled` to turn search off. Existing Codex
settings are preserved when the environment override is not set.

## Discord presence

The bot presence is dynamic. It is `online` after a recent user interaction,
changes to `idle` after 15 minutes without interaction, and changes to `Do Not
Disturb` only when a real Codex tool-backed request has run for at least 60
seconds. Basic conversation never becomes DND, and context compaction is not
treated as a task. The thresholds can be customized:

```text
THEIA_PRESENCE_IDLE_AFTER=900
THEIA_PRESENCE_LONG_TASK_AFTER=60
THEIA_PRESENCE_UPDATE_INTERVAL=15
```

## Logging

The Codex layer uses the `theia.codex` Python logger and defaults to concise,
colored `INFO` output matching discord.py's timestamp, level, and logger
format. It reports the basic app-server lifecycle, request and turn progress,
tool activity, approvals, and failures without logging raw payloads. Set
`THEIA_CODEX_LOG_LEVEL=DEBUG` for protocol-level diagnostics, or `WARNING` to
show only warnings and errors. Set `THEIA_CODEX_LOG_COLORS=false` when plain
output is needed. For more detail in an embedding application:

```python
import logging

logging.getLogger("theia.codex").setLevel(logging.DEBUG)
```

Diagnostics intentionally omit prompts, credentials, user or session ids,
commands, paths, raw protocol payloads, and tool output. Exception traces are
represented by their type rather than their potentially sensitive message.

## Source layout

```text
main.py                    Compatibility launcher and re-exports
theia/core.py              Shared types, configuration, and safe formatting
theia/ui.py                Approval, input, and modal components
theia/app_server.py        Codex App Server protocol and session lifecycle
theia/delivery.py          Progress delivery and response pagination
theia/bot.py               Discord commands and gateway events
theia/personality.py       Personality profile validation and storage
```
