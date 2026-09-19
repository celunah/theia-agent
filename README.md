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
- Theia-only usage tracking with compact and detailed views, background work, daily recaps, and optional self-improvement.
- A temporary, personality-aware mood, Rich Presence, and server-specific customization.
- Conversational attention that preserves substantial topic changes and follows brief tangents naturally.
- Optional automatic Codex CLI updates with staged verification and rollback.

Theia can be installed to a Discord account as well as a server. Account
installations provide slash-command conversations in DMs, group DMs, and
servers; installing her to a server additionally enables message listening,
threads, and voice.

## Commands

```text
/login  /about  /usage  /credits  /model  /mode
/approve  /deny  /stop  /undo  /btw [prompt] [file]
/skill  /personality  /memory [scope]  /commitments [action] [id]
/improvements  /customize  /restart
```

The Lighthouse View is an operator-facing terminal dashboard; `/improvements`,
`/restart`, and `/customize` are for administrators.
`/memory` is a
private, owner-locked explorer: users can inspect their own entries, server
administrators can inspect the current server, and Super Admins can inspect
broader scopes. The full capability inventory is in
[`THEIA_FEATURES.md`](THEIA_FEATURES.md).
Entries are shown newest-first with their UTC calendar date.

Its Recent events feed shows bounded timestamps, severity, and stable event
titles. Technical event details remain available through its diagnostic view.

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
On the first real launch, supported secret entries are migrated into the
encrypted vault and removed from the dotenv file. Once the vault exists,
`scripts/configure.py` asks for its passphrase and updates the vault directly.

To use account installation, enable User Install for the Discord application
and use its account-install link. Install Theia to a server when you want
message listening, threads, or voice.

### Docker

Create `.env`, add your Discord bot token, and create both bind-mount source
directories as the deploying account:

```bash
cp .env.example .env
mkdir -p "$HOME/.theia" "$HOME/theia-workspace"
export THEIA_UID="$(id -u)"
export THEIA_GID="$(id -g)"
docker compose up --build -d
```

On a systemd user session with an unlocked GNOME Keyring, the checked-in
`systemd/theia.service` can keep the container running and rebuild it when
needed:

```bash
install -Dm644 systemd/theia.service "$HOME/.config/systemd/user/theia.service"
systemctl --user daemon-reload
systemctl --user enable --now theia.service
```

The unit requires `gnome-keyring-daemon.service` and passes the user D-Bus
socket into the container for unattended vault unlocks. Starting the daemon
alone does not unlock a login keyring; the user session must unlock its Secret
Service collection first.

On native Windows PowerShell, use:

```powershell
Copy-Item .env.example .env
New-Item -ItemType Directory -Force "$HOME\.theia", "$HOME\theia-workspace"
docker compose up --build -d
```

Compose deliberately does not create the directories automatically, because
Docker may create missing bind sources as `root`, which prevents the non-root
Theia process from reading `/data`. On Linux or WSL, set `THEIA_UID` and
`THEIA_GID` to the account that owns those directories. Native Windows hosts
do not have a Unix UID/GID to mirror, so Compose defaults to Theia's container
identity (`1000:1000`) automatically. Theia repairs ownership of the runtime
and workspace mounts at startup, verifies access as that identity, and then
drops container privileges. If access still fails, or another startup component
cannot initialize, the Lighthouse remains available with a `FATAL` degraded
state instead of hiding the startup failure.

Administrator requests can inspect the private `.theia` runtime when needed,
but operations there always require approval.

Set `THEIA_SUPER_ADMIN_USERS` to a comma-separated list of trusted Discord user
IDs when someone needs Super Admin access outside normal server permissions.

For Qwen Audio middleware, set `THEIA_AUDIO_PROVIDER=qwen` (or leave it as
`auto`) and provide a `ws://` or `wss://` `THEIA_QWEN_AUDIO_URL`. The endpoint
must implement Theia's bounded audio-provider event protocol; Qwen remains an
audio middleware service and does not receive Theia's personality or tools.

Uploaded-media perception is a separate, opt-in path. Set
`THEIA_QWEN_PERCEPTION_ENABLED=true` and configure a Model Studio compatible
`THEIA_QWEN_PERCEPTION_BASE_URL` and private `THEIA_QWEN_PERCEPTION_API_KEY`.
The default model is `qwen3.8-omni-flash`. Before Qwen is called, Theia reads
the current Codex modality capabilities and sends natively supported media
directly to Codex. Qwen receives only unsupported, oversized, or explicitly
dedicated-perception media; the same item is never sent to both providers in
one request. The perception report is neutral JSON context, while Codex still
controls reasoning, personality, tools, and the final response. The current
Theia's real launcher migrates supported provider secrets from the initial
private deployment environment into an encrypted vault before connecting to
Discord. The vault uses an interactive passphrase by default; the Lighthouse
starts in a redacted `Locked` state until it is unlocked. Set
`THEIA_VAULT_KEYCHAIN=true` and `THEIA_VAULT_UNLOCK_MODE=auto` to try the OS
keychain first, or use `unattended` to fail without prompting when no keychain
credential is available. `THEIA_VAULT_IDLE_TIMEOUT` enables optional inactivity
locking in seconds. While the Lighthouse is running, Ctrl+L manually locks or
unlocks the vault from the terminal. The encrypted vault is stored at
`$THEIA_HOME/credentials.vault`; the passphrase is never stored there.

Set `THEIA_CODEX_AUTO_UPDATE=true` to let Theia check for and stage official
Codex CLI updates in her private runtime. In Docker, each update installs the
latest npm package into the persistent runtime with its `package.json`,
`package-lock.json`, and `node_modules`; the managed copy is selected before
the image-bundled fallback. Updates are disabled by default and are deferred
safely around active work. The update interval and timeout can be adjusted with
`THEIA_CODEX_UPDATE_INTERVAL` and `THEIA_CODEX_UPDATE_TIMEOUT`.
