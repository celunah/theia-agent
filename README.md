# Theia Agent

Theia is a private, persistent Codex agent for Discord. Chat with her in DMs,
servers, and threads while she remembers the conversations, files, and
preferences that matter to you.

## Features

- Natural text conversations with optional voice replies.
- Persistent conversations, attachments, memories, skills, and personality
  character cards.
- Personality profiles can be selected for yourself, the current server, or
  everyone in Theia, with administrator protection for shared scopes.
- Model selection, web search, approvals, and safe controls for agent actions.
- Theia-only usage tracking, background work, daily recaps, and optional self-improvement.
- A temporary, personality-aware mood, Rich Presence, and server-specific customization.

Theia can be installed to a Discord account as well as a server. Account
installations provide slash-command conversations in DMs, group DMs, and
servers; installing her to a server additionally enables message listening,
threads, and voice.

## Commands

```text
/login  /about  /usage  /debug  /credits  /model  /mode
/approve  /deny  /stop  /undo  /btw [prompt] [file]
/skill  /personality  /customize  /restart
```

`/debug`, `/restart`, and `/customize` are for administrators. `/debug` shows a
sanitized live view of Theia's runtime while it is open. The full capability
inventory is in [`THEIA_FEATURES.md`](THEIA_FEATURES.md).

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
