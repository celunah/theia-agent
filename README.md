# Theia Agent

Theia is a private, persistent Codex agent for Discord. Chat with her in DMs,
servers, and threads while she remembers the conversations, files, and
preferences that matter to you. Voice can use Codex Realtime, custom audio
services, or a configured Qwen Audio middleware endpoint.

## Features

- Natural text conversations with optional Codex Realtime or custom-provider voice.
- Persistent conversations, attachments, memories, skills, and personality
  character cards.
- Personality profiles can be selected for yourself, the current server, or
  everyone in Theia, with administrator protection for shared scopes.
- Model selection, web search, approvals, and safe controls for agent actions.
- Theia-only usage tracking, background work, daily recaps, and optional self-improvement.
- A temporary, personality-aware mood, Rich Presence, and server-specific customization.
- Conversational attention that preserves substantial topic changes and follows brief tangents naturally.
- Optional automatic Codex CLI updates with staged verification and rollback.

Theia can be installed to a Discord account as well as a server. Account
installations provide slash-command conversations in DMs, group DMs, and
servers; installing her to a server additionally enables message listening,
threads, and voice.

## Commands

```text
/login  /about  /usage  /debug  /credits  /model  /mode
/approve  /deny  /stop  /undo  /btw [prompt] [file]
/skill  /personality  /memory [scope]  /commitments [action] [id]
/improvements  /customize  /restart
```

`/debug`, `/improvements`, `/restart`, and `/customize` are for administrators.
`/memory` is a
private, owner-locked explorer: users can inspect their own entries, server
administrators can inspect the current server, and Super Admins can inspect
broader scopes. The full capability inventory is in
[`THEIA_FEATURES.md`](THEIA_FEATURES.md).

Leave the `/btw` prompt blank to enter it in a Discord modal. Generated images
and their response are delivered together with an owner-only `Follow up` control;
follow-ups update that same message.

## Setup

You need Python 3.10+, [uv](https://docs.astral.sh/uv/), Node.js with npm, and
a Discord bot token.

From the project directory:

```bash
python scripts/bootstrap.py
python scripts/configure.py
uv run python main.py
```

The setup wizard creates the local `.env` file and can configure text or voice
mode. Voice mode uses Codex Realtime by default, or custom OpenAI-compatible
STT/TTS services when both endpoints are configured. Keep `.env` private.

To use account installation, enable User Install for the Discord application
and use its account-install link. Install Theia to a server when you want
message listening, threads, or voice.

### Docker

Create `.env`, add your Discord bot token, then run:

```bash
cp .env.example .env
docker compose up --build -d
```

Theia keeps her private data and working files in the mounted directories.

Administrator requests can inspect the private `.theia` runtime when needed,
but operations there always require approval.

Set `THEIA_SUPER_ADMIN_USERS` to a comma-separated list of trusted Discord user
IDs when someone needs Super Admin access outside normal server permissions.

For Qwen Audio middleware, set `THEIA_AUDIO_PROVIDER=qwen` (or leave it as
`auto`) and provide a `ws://` or `wss://` `THEIA_QWEN_AUDIO_URL`. The endpoint
must implement Theia's bounded audio-provider event protocol; Qwen remains an
audio middleware service and does not receive Theia's personality or tools.

Set `THEIA_CODEX_AUTO_UPDATE=true` to let Theia check for and stage official
Codex CLI updates in her private runtime. Updates are disabled by default and
are deferred safely around active work. The update interval and timeout can be
adjusted with `THEIA_CODEX_UPDATE_INTERVAL` and `THEIA_CODEX_UPDATE_TIMEOUT`.
