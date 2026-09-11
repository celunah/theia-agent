# Conversational Attention and Natural Topic Transitions

## Goal

Add lightweight conversational attention management to Theia so one conversation can move naturally between multiple subjects while preserving earlier topics, unresolved ideas, and return points.

The feature should remain separate from work-task scheduling, mood, personality selection, persistent memory, and Discord routing.

## Architecture recommendation

Attach a dedicated attention state to each existing Theia session. Do not create separate Codex threads, Discord threads, worker sessions, or persistent memory records for conversational topics.

The existing session key already provides the relevant isolation:

```text
guild:<guild_id>:channel:<channel_id>:user:<user_id>
```

Account-installed operation should continue using its existing session-key behavior. Attention must follow the same effective session boundary and must not become process-global.

The classifier should be neutral and semantic. It should receive bounded topic context and recent messages, but it does not need to inherit the character's personality to classify relations. The main character agent will retain the active personality and generate natural transition wording.

The classifier must not be asked to determine which messages are relevant before it performs classification. Its input should contain the active context summary and recent exchanges, bounded metadata for parked contexts, and a small window of the latest global messages. The classifier then decides which context the new message belongs to.

`OFF_TOPIC` should not automatically create a durable context for every random aside. The harness should apply a retention threshold:

- a brief unrelated question remains a temporary aside;
- a substantial unrelated discussion becomes a parked context;
- an explicit “remember this thread” request or return cue always preserves the thread.

This threshold is a state-retention decision after semantic classification, not a replacement for semantic classification.

## Proposed data structures

Add focused dataclasses, likely in a new `theia/server/attention.py` or in shared core types if that is needed to avoid import cycles.

```text
ConversationContext:
    context_id
    title
    summary
    recent_messages
    open_loops
    parent_context_id
    last_active_at
    status                 # active, parked, closed
    transition_history

ConversationAttentionState:
    active_context_id
    contexts
    parked_context_ids
    last_transition_signature
    acknowledged_transition_signature
    version
```

All collections and strings must be bounded:

- limit the number of retained contexts;
- limit recent messages per context;
- limit unresolved ideas and open loops;
- limit transition history;
- limit topic title and summary lengths.

The harness creates context IDs. A classifier may refer only to context IDs already known to that session; unknown IDs are ignored.

## Transition flow

For each normal user turn:

1. `requests.py` obtains the existing session and its lock.
2. The attention manager loads the active and parked contexts for that session.
3. A low-effort, ephemeral Codex classifier receives:
   - the active topic;
   - bounded parked-topic metadata;
   - recent relevant messages from this session;
   - the current user message;
   - explicit return hints such as “back to” or “where were we?”.
4. The classifier returns strictly validated JSON.
5. The harness applies the result:
   - `CONTINUE`, `RELATED_EXTENSION`, and `CLARIFICATION` keep the active context;
   - a substantial `SIDETRACK` creates a child context and parks the parent;
   - `TOPIC_SHIFT` and `OFF_TOPIC` park the current context and create a new one;
   - `RETURN` restores the selected parked context;
   - `NESTED_RETURN` restores an earlier context while preserving the current sidetrack;
   - `END` closes the active context without creating a replacement.
6. The harness creates a structured transition event.
7. A compact attention instruction is added to the current turn prompt when acknowledgement is appropriate.
8. The normal character agent generates the acknowledgement and useful answer.
9. The bounded exchange is recorded in the active context for future classification.

The classifier runs before `turn/start` because a transition acknowledgement must affect the current response. It should have a short timeout and fail open: if classification fails or times out, Theia answers normally without a transition acknowledgement.

This introduces bounded latency to ordinary turns where classification is required. It should use low effort and, where safe, run concurrently with existing bounded reasoning assessment work.

## Supported relations

The classifier must return one of:

```text
CONTINUE
RELATED_EXTENSION
SIDETRACK
TOPIC_SHIFT
OFF_TOPIC
RETURN
NESTED_RETURN
CLARIFICATION
END
```

Classification must use semantic context, not only keyword matching. Phrases such as “by the way”, “speaking of”, “anyway”, “back to”, and “where were we?” are useful signals but must not be the sole mechanism.

An off-topic message remains valid. Theia must answer it and must never refuse, police, or require the user to return to the previous subject.

## Proposed classifier contract

```json
{
  "relation": "TOPIC_SHIFT",
  "confidence": 0.91,
  "acknowledge": true,
  "target_context_id": null,
  "topic_title": "Conversational attention",
  "topic_summary": "How Theia preserves and returns to multiple subjects.",
  "open_loops": [
    "Define how topic transitions are detected."
  ],
  "reason": "The user deliberately introduced a different subject."
}
```

For a return:

```json
{
  "relation": "RETURN",
  "confidence": 0.96,
  "acknowledge": true,
  "target_context_id": "known_context_id",
  "topic_title": null,
  "topic_summary": null,
  "open_loops": [],
  "reason": "The user returned to a parked topic."
}
```

The harness must:

- accept only the nine supported relation values;
- clamp confidence to `0.00` through `1.00`;
- bound every string and list;
- reject malformed or unsafe output;
- ensure `target_context_id` belongs to the current session;
- create new IDs itself;
- replace stale summaries and open loops rather than append indefinitely.

The reason is internal metadata only. It must be paraphrased and must not contain hidden chain-of-thought, raw tool calls, credentials, secrets, private paths, or unnecessarily copied user content.

## Attention transition event

The harness should produce runtime state equivalent to:

```json
{
  "type": "conversation_transition",
  "relation": "OFF_TOPIC",
  "previous_context_id": "context_a",
  "new_context_id": "context_b",
  "previous_topic": "Theia's conversational attention",
  "new_topic": "Rust memory allocation",
  "acknowledge": true,
  "return_available": true
}
```

The event is internal runtime state. Internal labels, context IDs, and implementation details must never be exposed to the user.

## Prompt integration

The active personality overlay remains in the normal system/developer instructions. The attention instruction belongs in the temporary turn-prompt builder, after personality-derived context and before the current user input.

Example:

```text
[Conversational attention]
The user has moved from "<previous topic>" into "<new topic>".
This is a separate conversational thread. Acknowledge the transition
naturally in one brief sentence, then answer the new subject directly.
Do not mention internal labels or context identifiers. Do not refuse,
police, or redirect the user.
[/Conversational attention]
```

The model chooses the actual wording. The instruction must require the acknowledgement to be followed by a useful answer.

For `CONTINUE` and ordinary `CLARIFICATION`, no attention instruction should be added.

Transition wording should be naturally varied through the character's existing personality. The harness should suppress repeated acknowledgements for the same transition signature while still allowing a new acknowledgement when the user genuinely changes or returns to a topic.

## Persistence and restoration

Serialize attention state inside each real session record under a versioned `attention` field.

Attention must not:

- create internal worker sessions;
- write to `MEMORY.md` or `USER.md`;
- create or update skills;
- modify personality files;
- enter nightly recaps;
- affect permanent usage history;
- create new Codex threads.

Reuse the existing atomic persistence and corrupt-state handling. During restoration, validate the attention structure and discard invalid attention data without invalidating the normal session.

Session restoration must preserve active, parked, and closed contexts deterministically. Existing retention and session cleanup must not be bypassed merely because attention contexts exist.

## Affected files

Expected implementation changes:

- `theia/server/attention.py` — attention state model, classifier worker, transition state machine, validation, and rendering helper.
- `theia/core.py` — shared attention dataclasses only if needed to avoid import cycles.
- `theia/server/requests.py` — invoke attention classification and apply the transition before `turn/start`.
- `theia/server/conversation.py` — inject the rendered attention context into the temporary turn prompt.
- `theia/server/prompts.py` — classifier schema and constrained internal instructions.
- `theia/server/state.py` — serialize, restore, validate, and migrate attention state.
- `theia/server/core.py` — register the new mixin or module if required by the current composition.
- `tests/test_attention.py` — focused state-machine and rendering tests.
- `tests/test_integration.py` — JSONL-boundary classification and normal-answer integration coverage.
- `tests/fixtures/fake_app_server.ts` — deterministic attention-classification scenario.
- `THEIA_FEATURES.md` and the concise user-facing README feature list — document the user-visible behavior if required by the existing feature inventory.

No changes should be needed to Discord routing, personality selection, mood, Rich Presence, voice, approval handling, or work-task scheduling.

## Failure, cancellation, and security behavior

The attention classifier must use the existing ephemeral internal-worker pattern:

- `approvalPolicy: "never"`;
- read-only sandbox;
- no tools or dynamic tools;
- low reasoning effort;
- no durable Codex session history;
- bounded timeout;
- cancellation and cleanup on turn cancellation or server shutdown.

Classifier failure must not fail the user turn. It should produce a normal response without transition metadata.

Attention state must be applied only under the existing session lock. A per-session state/version check should prevent a stale classifier result from overwriting a newer transition.

The classifier must receive only the bounded context for the current effective session. It must not receive unrelated guild, channel, or user context. User content and prior context must be treated as untrusted data.

The attention layer must not alter:

- approval or permission decisions;
- safe/read-only tool policy;
- safety rules or factual-accuracy requirements;
- adaptive reasoning;
- mood appraisal;
- Rich Presence;
- voice and TTS;
- online, idle, or DND status;
- account-installed versus guild-installed capability checks;
- self-improvement, memory, recap, or personality updates.

## Test strategy

Add focused tests covering:

1. Same-topic continuation without transition text.
2. Related topic extension.
3. Explicit and implicit unrelated topic shifts.
4. Explicit off-topic messages.
5. Temporary sidetracks.
6. Explicit and implicit returns.
7. Nested sidetracks and nested returns.
8. Clarifications that do not create new topics.
9. Ending a topic.
10. Multiple topics in one Discord channel.
11. Correct restoration of topic B after A → B → C → B.
12. User, channel, and guild isolation.
13. Persistence across session restoration.
14. Corrupt or invalid persisted attention data.
15. Bounded summaries, recent messages, open loops, and transition history.
16. Replacement of stale causes and summaries.
17. Suppression of repetitive transition wording.
18. Uncertain topic changes using a subtle bridge or no acknowledgement.
19. Transition acknowledgement followed by an actual answer.
20. Off-topic requests never being refused or policed.
21. Classifier timeout, malformed output, cancellation, and cleanup.
22. No tools, memory, recap, skill, personality, or self-improvement changes from classification.
23. Unchanged approval, permission, safety, mood, presence, voice, and session behavior.
24. Source-file size limits.

Use the TypeScript fake App Server for JSONL/process-boundary behavior rather than mocking the entire App Server. Run the complete local contract after implementation:

```text
uv run python scripts/run_ci.py
```

## Non-goals

This implementation should not:

- turn conversational topics into scheduled work tasks;
- create a separate Codex thread for every subject;
- permanently learn topics through memory or nightly recaps;
- let the model mutate active or parked contexts without harness validation;
- require explicit topic labels from the user;
- restrict or reject off-topic questions;
- redesign Discord thread management or session routing.
