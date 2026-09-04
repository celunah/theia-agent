# Theia capabilities

Theia is a Discord harness around a locally bundled Codex app-server, currently pinned to Codex `0.153.0`. It supports authenticated, persistent, isolated Codex conversations in Discord, with text, optional voice, skills, memories, personalities, approvals, model selection, web search, session recovery, and retention management.

The Codex app-server is based around threads, turns, items, streamed notifications, and bidirectional JSON-RPC/JSONL communication.

## Discord interface

The current commands are:

- `/login`
- `/about`
- `/usage`
- `/credits`
- `/approve`
- `/deny`
- `/stop`
- `/undo`
- `/btw <prompt> [file]`
- `/skill <skill_name>`
- `/personality [file] [name]`
- `/model <model>`
- `/mode text|voice`
- `/restart`
- `/customize [target] [element] [value]`

Prefix commands are disabled.

`/about` privately displays the running Theia version and short source
revision, selected Codex CLI version, invoking Discord account, Codex plan,
and current session mode and personality.

Theia also:

- Responds to DMs and configured guild messages.
- Supports Discord threads as separate Codex sessions.
- Optionally creates a response thread only when the user explicitly requests
  one, with fallback to the original channel when thread creation fails.
- Thread intent accepts common semantic wording such as keeping a discussion
  separate or giving it its own space, while rejecting informational questions
  and negated requests.
- Server administrators expose a guarded `discord.create_thread` Codex tool;
  Codex decides when a thread is needed, can assign the same name to Discord
  and the Codex session, and route the first response into it.
- Names threads from the initial request.
- Includes bounded recent Discord history when asked about channel context.
- Restores sessions after restart.
- Lets server administrators restart the bot process in place.
- Archives sessions after 30 days of inactivity and deletes them after 90 days.
- Unarchives an archived session when the user returns.
- Creates a new session if the old one was deleted.
- Supports `/undo` through `thread/rollback`.

Normal answers are plain text. Long messages use Discord buttons, emoji reactions as fallback, and finally message splitting if neither is available.

Command responses, approvals, and choices use embeds. Statuses use compact `-#` text:

- `Thinking` only appears for tool-backed work.
- `Thought for N seconds` or `Thought for M minutes and S seconds` remains after completion.
- Natural-language Codex preambles and intermediate messages are shown when Codex emits them.
- Updates are coalesced rather than token-streamed.
- Raw commands, tool calls, paths, output, credentials, and chain-of-thought are not exposed.
- Failure embeds include the underlying sanitized Codex error.

Typing indicators are shown while Theia is responding.

Server administrators can use `/customize` to change Discord-only embed
titles, embed content, embed colors, status labels, and interaction button
labels, including pagination controls, input modal labels, and embed field
labels. Targets can be commands such as `/usage` or frontend labels such as
`Thinking`. Values support Markdown and placeholders including `{user}`,
`{server}`, `{channel}`, `{model}`, `{mode}`, `{skill}`, `{personality}`,
`{prompt}`, `{reason}`, `{status}`, `{duration}`, `{text}`, `{balance}`, and
usage/limit fields. Use `default` to reset a customization. Preferences are
server-wide and stored separately from Codex sessions in
`~/.theia/discord-customizations.json`; they never enter Codex prompts or
agent state.

## Authentication and isolation

- `/login` uses Codex’s ChatGPT device-code login flow.
- Authentication is cached in Theia’s private runtime, normally `~/.theia`.
- A valid private cache is reused; another Codex cache is imported atomically
  when needed, and the device-code flow is used only when no usable cache remains.
- Login reports `Already logged in`, `Cached authentication imported`,
  `Device code required`, and `Authentication completed` for those states.
- A server administrator can authenticate the bot for server-wide use.
- Sessions are isolated by Discord user, channel, and thread.
- Approvals are bound to the Discord user, thread, turn, and approval item.
- Users cannot approve or deny another user’s request.
- Pending approvals are cleared after completion, denial, cancellation, timeout, or interruption.
- ChatGPT tokens and internal app-server credentials are never sent to Discord.

## Models and reasoning

- `/model` obtains available models dynamically through `model/list`.
- Model autocomplete is populated from the current Codex model catalog.
- Theia does not hardcode a fixed list of models.
- Actual models depend on the logged-in account, Codex build, configuration, and provider.
- Reasoning is automatic:
  - No-tool requests use low/light reasoning.
  - Tool-backed requests are pre-assessed as simple, moderate, complex, or very complex.
  - Theia then selects medium, high, or maximum supported reasoning.
  - Outside adaptive reasoning, the default is medium.
- A manual reasoning command is intentionally not exposed.
- `/model` is the only manual model-selection command.

## Codex API calls implemented

Theia currently calls or wraps these app-server methods.

### Account and discovery

```text
initialize
account/read
account/login/start
account/usage/read
account/rateLimits/read
model/list
modelProvider/capabilities/read
skills/list
skills/extraRoots/set
app/list
```

### Thread lifecycle and history

```text
thread/start
thread/resume
thread/read
thread/list
thread/fork
thread/loaded/list
thread/turns/list
thread/items/list
thread/name/set
thread/rollback
thread/archive
thread/unarchive
thread/delete
thread/compact/start
thread/goal/get
thread/goal/set
thread/goal/clear
```

### Turn control

```text
turn/start
turn/steer
turn/interrupt
```

Theia uses the history and loaded-session methods to reconstruct sessions after restart, and uses thread naming, archive, unarchive, rollback, and deletion internally.

## Codex events and requests handled

Theia processes events including:

```text
account/updated
account/rateLimits/updated
account/login/completed
thread/started
thread/status/changed
thread/closed
thread/deleted
thread/archived
thread/unarchived
skills/changed
item/started
item/completed
item/agentMessage/delta
item/commandExecution/outputDelta
turn/completed
context/compacted
thread/compacted
error
```

It handles Codex approval and interaction requests for:

- Command execution approval.
- File-change approval.
- Permission approval.
- Sequential multiple-choice and user-input questions.
- MCP elicitation forms, URLs, and choices.
- Theia’s restricted Discord messaging tool.
- Theia’s administrator-only Discord thread-creation tool.

Multiple-choice questions are asked one at a time, with the final structured payload returned only after all answers are collected.

## Tools and permissions

Codex provides the actual native tools. Theia mediates their permissions and Discord presentation.

Supported tool categories include:

- Command or shell execution.
- File changes and patch application.
- MCP tools.
- Web search.
- Image generation.
- Computer/tool calls.
- Theia’s Discord messaging tool.

Server administrators receive the configured Codex tool policy, normally workspace-write with approval requests. Normal users are restricted to read-only/safe operation and cannot perform writes, state-changing commands, credential access, external side effects, or unrestricted dynamic Discord actions.

Tool availability still depends on the selected model, account, provider capabilities, and Codex configuration. `modelProvider/capabilities/read` is implemented and cached, but does not currently have a dedicated Discord command.

## Web search

Theia uses Codex’s native web-search capability rather than implementing a separate search engine.

Supported configuration modes are:

```text
disabled
indexed
live
```

The default is indexed search. Live mode can be explicitly selected when current web information is required.

## Skills, memories, and personalities

### Skills

- `/skill` uses Discord autocomplete.
- The skill catalog comes from `skills/list`.
- It refreshes when `skills/changed` is received.
- Skills are discovered from Theia’s private directory, global Codex directories, Hermes directories, and project-local skill roots.
- Skills created by other Codex agents are discoverable when they are under the configured shared roots.

### Memories

- Memory roots are configurable.
- Theia can read private Theia memories, shared Hermes memories, global Codex memories, and project memory directories.
- Memory snapshots are bounded for safety.
- `Memory created` and `Memory updated` are displayed only after verified file-change/tool events affecting configured memory roots.

### Personalities

- `/personality file name` uploads and activates a Markdown/text personality.
- `/personality name` switches to an existing profile.
- `/personality name:none` clears it.
- `/personality` explains usage.
- Personality-name autocomplete is supported.
- Changing personality resets the Codex thread so the system instructions remain consistent.
- The base prior is identity-neutral and personality-independent.

## Voice mode

Text mode is the default.

Voice mode requires configured OpenAI-compatible services:

```text
STT_BASE_URL
STT_TOKEN
TTS_BASE_URL
TTS_TOKEN
```

Optional protocol, model, voice, and format settings are also supported.

When enabled:

- Theia listens in a Discord voice channel.
- Speech is transcribed and submitted to the same Codex session.
- Speakers are differentiated when the STT service supports it.
- Speech can interrupt and steer an active turn.
- Preambles, intermediates, and final responses can be spoken through TTS.
- Text is still posted in the associated Discord channel.
- `/mode voice` is unavailable unless the required APIs and voice dependencies are configured.

## Logging and presence

The `theia.codex` logger uses discord.py-style colored output.

It logs basic Codex lifecycle, requests, turns, tools, approvals, and failures without sensitive payloads. Debug logging can be enabled with:

```text
THEIA_CODEX_LOG_LEVEL=DEBUG
```

Presence behavior:

- Online after recent interaction.
- Idle after inactivity.
- DND only for genuinely long-running tool-backed tasks.
- Basic conversation and context compaction do not trigger DND.

## Important limitations

Theia is not a complete implementation of every current Codex app-server API. There is currently no Theia wrapper or Discord integration for several broader or newer surfaces, including:

- Configuration read/write RPCs.
- Logout or login cancellation.
- Collaboration-mode APIs.
- Plugin-management APIs.
- Remote-control APIs.
- Realtime APIs.
- Review APIs.
- Fuzzy file-search APIs.
- Direct command-execution APIs outside normal Codex turns.

Theia does implement the core conversation, account, model, skill, approval, history, lifecycle, rollback, retention, and interaction paths needed for Discord usage. Exact model and tool availability remains dependent on the installed Codex version, account, configuration, and provider.
