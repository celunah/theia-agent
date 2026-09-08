# Theia Agent

Theia is a private, persistent Codex agent for Discord. Chat with her in DMs,
servers, and threads while she remembers the conversations, files, and
preferences that matter to you.

## Features

- Natural text conversations with optional voice replies.
- Persistent conversations, attachments, memories, skills, and personality
  character cards.
- Model selection, web search, approvals, and safe controls for agent actions.
- Theia-only usage tracking, background work, daily recaps, and optional self-improvement.
- A temporary, personality-aware mood, Rich Presence, and server-specific customization.

Theia can be installed to a Discord account as well as a server. Account
installations provide slash-command conversations in DMs, group DMs, and
servers; installing her to a server additionally enables message listening,
threads, and voice.

## Commands

```text
/login  /about  /usage  /credits  /model  /mode
/approve  /deny  /stop  /undo  /btw
/skill  /personality  /customize  /restart
```

`/restart` and `/customize` are for administrators. The full capability
inventory is in [`THEIA_FEATURES.md`](THEIA_FEATURES.md).

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
mode. Keep `.env` private.

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
