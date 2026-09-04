# Theia Agent source instructions

## Project boundaries

- Theia is a Discord-native conversational Codex agent. Keep Discord delivery,
  Codex protocol handling, persistence, voice, personalities, memories, and
  customization concerns in their existing modules.
- `main.py` is a compatibility launcher and re-export surface. Put new runtime
  behavior under `theia/` and update the re-exports only when callers need them.
- The local Codex App Server remains a child process. Do not replace its JSONL
  boundary with direct provider calls or expose raw protocol payloads to Discord.
- Keep the Python target at 3.10 syntax and preserve compatibility through 3.14.

## Behavioral invariants

- Keep authentication private to Theia's runtime home. Reuse a valid Theia
  cache, securely import another Codex cache when needed, and use the device-code
  flow only when no usable cache remains.
- Never log credentials, raw prompts, raw tool output, private paths, or raw
  protocol payloads. Preserve the existing sanitization and restricted-file
  permissions.
- Preserve session recovery, atomic state writes, corrupt-state quarantine,
  duplicate-message suppression, approval ownership checks, and safe path-root
  validation.
- Regular users remain on the safe/read-only Codex policy. Administrator access
  must be rechecked at permission-sensitive boundaries, including voice input.
- Keep configuration as the local `scripts/configure.py` setup script. It is not
  a Discord slash command.

## Commands and customization

Whenever a slash command is added or renamed, update all of the following:

1. The command registration and handler in `theia/bot.py`.
2. The command target list exposed by `FrontendCustomizationStore` in
   `theia/customization.py`.
3. The command-surface and customization tests in `tests/test_main.py`.
4. The concise command list and behavior summary in `README.md` and
   `THEIA_FEATURES.md` when user-visible behavior changes.

Whenever a new user-facing label is added that can be customized, add its stable
target to the customization store, render it through the frontend customization
helpers, and add a test proving both its default and customized values. The
customizable target list must never lag behind the commands or labels exposed by
the bot.

Keep customization server-scoped, administrator-only, placeholder-validated,
and separate from Codex prompts and persisted Codex session state.

## Tests and validation

- Add deterministic tests for new behavior and failure paths. Use the TypeScript
  fake App Server for real JSONL/process-boundary behavior rather than mocking
  the entire boundary.
- Keep `tests/fixtures/fake_app_server.ts` valid under both Prettier and strict
  TypeScript checking.
- Run the complete local contract before handoff:

  ```text
  uv run python scripts/run_ci.py
  ```

- For optional executable builds, use `uv run --with nuitka python
  scripts/build_nuitka.py`. The one-file executable still launches Codex as a
  child process and requires Node.js and a usable Codex CLI.

Document meaningful user-visible behavior in the relevant README or feature
inventory, but do not add boilerplate docstrings or documentation that merely
repeats a function name.
