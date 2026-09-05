import { createInterface } from "node:readline";

type JsonObject = Record<string, unknown>;
type RpcId = number | string;

interface RpcRequest {
  method: string;
  id?: RpcId;
  params: JsonObject;
}

interface ActiveTurn {
  threadId: string;
  turnId: string;
  prompt: string;
}

const scenario = (process.env.FAKE_APP_SERVER_SCENARIO || "").toLowerCase();
let nextThreadId = 1;
let nextTurnId = 1;
const activeTurns = new Map<string, ActiveTurn>();
const pendingApprovals = new Map<string, ActiveTurn>();
const pendingQuestions = new Map<string, ActiveTurn>();
const realtimeThreads = new Set<string>();
const realtimeResponses = new Set<string>();

function scenarioIs(...names: string[]): boolean {
  return names.includes(scenario);
}

function isRecord(value: unknown): value is JsonObject {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function isRpcId(value: unknown): value is RpcId {
  return typeof value === "number" || typeof value === "string";
}

function write(message: JsonObject): void {
  process.stdout.write(`${JSON.stringify(message)}\n`);
}

function respond(request: RpcRequest, result: JsonObject): void {
  if (request.id !== undefined) {
    write({ id: request.id, result });
  }
}

function respondToId(id: RpcId, result: JsonObject): void {
  write({ id, result });
}

function fail(request: RpcRequest, code: number, message: string): void {
  if (request.id !== undefined) {
    write({ id: request.id, error: { code, message } });
  }
}

function failInvalid(
  id: RpcId | null,
  code: -32600 | -32602,
  message: string,
): void {
  write({ id, error: { code, message } });
}

function notify(method: string, params: JsonObject): void {
  write({ method, params });
}

function realtimeAudioData(): string {
  return Buffer.alloc(3840).toString("base64");
}

function emitRealtimeResponse(threadId: string): void {
  if (realtimeResponses.has(threadId)) {
    return;
  }
  realtimeResponses.add(threadId);
  notify("thread/realtime/transcript/done", {
    threadId,
    role: "user",
    text: "hello from realtime",
  });
  notify("thread/realtime/transcript/done", {
    threadId,
    role: "assistant",
    text: "realtime response",
  });
  notify("thread/realtime/outputAudio/delta", {
    threadId,
    audio: {
      data: realtimeAudioData(),
      sampleRate: 48000,
      numChannels: 2,
    },
  });
}

function stringParam(params: JsonObject, name: string): string {
  const value = params[name];
  return typeof value === "string" ? value : "";
}

function inputPrompt(params: JsonObject): string {
  const value = params.input;
  if (!Array.isArray(value)) {
    return "";
  }
  return value
    .filter(isRecord)
    .filter((item) => item.type === "text")
    .map((item) => (typeof item.text === "string" ? item.text : ""))
    .join("\n");
}

function completeTurn(
  turn: ActiveTurn,
  status: string,
  text = "",
  error?: JsonObject,
): void {
  const item = text
    ? {
        type: "agentMessage",
        id: `message-${turn.turnId}`,
        phase: "final",
        text,
      }
    : undefined;
  const completed: JsonObject = {
    id: turn.turnId,
    status,
    items: item ? [item] : [],
  };
  if (error) {
    completed.error = error;
  }
  notify("turn/completed", {
    threadId: turn.threadId,
    turnId: turn.turnId,
    turn: completed,
  });
  activeTurns.delete(turn.turnId);
}

function emitCommentary(turn: ActiveTurn, text: string, label: string): void {
  const itemId = `${label}-${turn.turnId}`;
  notify("item/started", {
    threadId: turn.threadId,
    turnId: turn.turnId,
    item: { type: "agentMessage", id: itemId, phase: "commentary", text: "" },
  });
  notify("item/agentMessage/delta", {
    threadId: turn.threadId,
    turnId: turn.turnId,
    itemId,
    delta: text,
  });
  notify("item/completed", {
    threadId: turn.threadId,
    turnId: turn.turnId,
    item: { type: "agentMessage", id: itemId, phase: "commentary", text },
  });
}

function streamTurn(turn: ActiveTurn, text: string, malformed = false): void {
  if (malformed) {
    process.stdout.write("this is intentionally not JSON\n");
  }
  const itemId = `message-${turn.turnId}`;
  notify("item/started", {
    threadId: turn.threadId,
    turnId: turn.turnId,
    item: { type: "agentMessage", id: itemId, phase: "final", text: "" },
  });
  const split = Math.max(1, Math.floor(text.length / 2));
  for (const delta of [text.slice(0, split), text.slice(split)]) {
    if (delta) {
      notify("item/agentMessage/delta", {
        threadId: turn.threadId,
        turnId: turn.turnId,
        itemId,
        delta,
      });
    }
  }
  notify("item/completed", {
    threadId: turn.threadId,
    turnId: turn.turnId,
    item: { type: "agentMessage", id: itemId, phase: "final", text },
  });
  completeTurn(turn, "completed", text);
}

function responseText(turn: ActiveTurn): string {
  const lowerPrompt = turn.prompt.toLowerCase();
  if (scenarioIs("refusal")) {
    return "I can't help with that request.";
  }
  if (scenarioIs("prompt") || lowerPrompt.includes("echo")) {
    return `echo: ${turn.prompt}`;
  }
  if (scenarioIs("normal") || lowerPrompt.includes("normal")) {
    return "normal response";
  }
  return "streamed response";
}

function finishNormalTurn(turn: ActiveTurn): void {
  if (scenarioIs("preamble", "preamble-and-intermediate")) {
    emitCommentary(turn, "Here is a short preamble.", "preamble");
  }
  if (scenarioIs("intermediate", "preamble-and-intermediate")) {
    emitCommentary(turn, "Working through the request.", "intermediate");
  }
  streamTurn(turn, responseText(turn), scenarioIs("malformed-stream"));
}

function startTurn(request: RpcRequest): void {
  const threadId = stringParam(request.params, "threadId") || "thread-missing";
  const turn: ActiveTurn = {
    threadId,
    turnId: `turn-${nextTurnId++}`,
    prompt: inputPrompt(request.params),
  };
  activeTurns.set(turn.turnId, turn);
  respond(request, { turn: { id: turn.turnId } });

  if (scenarioIs("crash")) {
    setTimeout(() => process.exit(17), 25);
    return;
  }
  if (scenarioIs("eof")) {
    setTimeout(() => process.exit(0), 25);
    return;
  }
  if (scenarioIs("timeout")) {
    return;
  }
  if (scenarioIs("interrupt")) {
    return;
  }
  if (scenarioIs("outage")) {
    setTimeout(() => {
      notify("error", {
        threadId,
        turnId: turn.turnId,
        error: { code: 503, message: "OpenAI service unavailable" },
      });
      activeTurns.delete(turn.turnId);
    }, 25);
    return;
  }
  if (scenarioIs("approval")) {
    const approvalId = `approval-${turn.turnId}`;
    pendingApprovals.set(approvalId, turn);
    setTimeout(
      () =>
        write({
          method: "item/commandExecution/requestApproval",
          id: approvalId,
          params: {
            threadId,
            turnId: turn.turnId,
            itemId: `command-${turn.turnId}`,
            reason: "fake app-server approval request",
          },
        }),
      25,
    );
    return;
  }
  if (scenarioIs("multiple-choice")) {
    const questionId = `question-${turn.turnId}`;
    pendingQuestions.set(questionId, turn);
    setTimeout(
      () =>
        write({
          method: "item/tool/requestUserInput",
          id: questionId,
          params: {
            threadId,
            turnId: turn.turnId,
            questions: [
              {
                id: "color",
                header: "Color",
                question: "Choose a color",
                options: [
                  { label: "Blue", description: "A blue answer" },
                  { label: "Green", description: "A green answer" },
                ],
              },
            ],
          },
        }),
      25,
    );
    return;
  }
  if (scenarioIs("error")) {
    setTimeout(() => {
      notify("error", {
        threadId,
        turnId: turn.turnId,
        error: { message: "fake app-server failure" },
      });
      activeTurns.delete(turn.turnId);
    }, 25);
    return;
  }
  setTimeout(() => finishNormalTurn(turn), 25);
}

function handleApprovalResponse(id: RpcId, message: JsonObject): boolean {
  const turn = pendingApprovals.get(String(id));
  if (!turn) {
    return false;
  }
  pendingApprovals.delete(String(id));
  const result = isRecord(message.result) ? message.result : {};
  const decision = result.decision === "accept" ? "approved" : "denied";
  respondToId(id, {});
  streamTurn(turn, `${decision} response`);
  return true;
}

function handleQuestionResponse(id: RpcId, message: JsonObject): boolean {
  const turn = pendingQuestions.get(String(id));
  if (!turn) {
    return false;
  }
  pendingQuestions.delete(String(id));
  respondToId(id, {});
  streamTurn(turn, "choice accepted");
  return true;
}

function handleResponse(message: JsonObject): void {
  const id = message.id;
  if (!isRpcId(id)) {
    return;
  }
  handleApprovalResponse(id, message) || handleQuestionResponse(id, message);
}

function handleRequest(request: RpcRequest): void {
  const { method, params } = request;
  switch (method) {
    case "initialize":
      respond(request, {
        serverInfo: { name: "theia-test-server" },
        capabilities: {},
      });
      return;
    case "initialized":
      return;
    case "skills/extraRoots/set":
      respond(request, {});
      return;
    case "skills/list":
      respond(request, { data: [] });
      return;
    case "experimentalFeature/list":
      respond(request, {
        data: [
          {
            name: "realtime_conversation",
            enabled: !scenarioIs("realtime-disabled"),
            defaultEnabled: false,
            stage: "underDevelopment",
          },
        ],
      });
      return;
    case "account/read":
      respond(request, {
        account: scenarioIs("auth-failure") ? null : { type: "chatgpt" },
        requiresOpenaiAuth: scenarioIs("auth-failure"),
      });
      return;
    case "account/rateLimits/read":
      respond(request, {
        rateLimits: {
          primary: {
            usedPercent: scenarioIs("rate-limit") ? 100 : 12,
            windowDurationMins: 60,
            resetsAt: 4102444800,
          },
        },
      });
      return;
    case "account/usage/read":
      respond(request, {
        usage: {
          limit: 100,
          remaining: scenarioIs("usage-exhausted") ? 0 : 84,
        },
        exhausted: scenarioIs("usage-exhausted"),
      });
      return;
    case "thread/loaded/list":
      respond(request, { data: [] });
      return;
    case "model/list":
      respond(request, { data: [{ id: "fake-model", model: "fake-model" }] });
      return;
    case "thread/start": {
      const threadId = `thread-${nextThreadId++}`;
      respond(request, { thread: { id: threadId } });
      notify("thread/started", { thread: { id: threadId }, threadId });
      return;
    }
    case "thread/resume": {
      const threadId = stringParam(params, "threadId") || "thread-missing";
      respond(request, { thread: { id: threadId } });
      notify("thread/started", { thread: { id: threadId }, threadId });
      return;
    }
    case "thread/realtime/start": {
      const threadId = stringParam(params, "threadId") || "thread-missing";
      realtimeThreads.add(threadId);
      respond(request, {});
      setTimeout(
        () =>
          notify("thread/realtime/started", {
            threadId,
            version: "v3",
            realtimeSessionId: `realtime-${threadId}`,
          }),
        5,
      );
      return;
    }
    case "thread/realtime/appendAudio": {
      const threadId = stringParam(params, "threadId") || "thread-missing";
      respond(request, {});
      if (scenarioIs("realtime")) {
        emitRealtimeResponse(threadId);
      }
      return;
    }
    case "thread/realtime/appendSpeech": {
      const threadId = stringParam(params, "threadId") || "thread-missing";
      respond(request, {});
      if (scenarioIs("realtime")) {
        emitRealtimeResponse(threadId);
      }
      return;
    }
    case "thread/realtime/stop": {
      const threadId = stringParam(params, "threadId") || "thread-missing";
      realtimeThreads.delete(threadId);
      respond(request, {});
      notify("thread/realtime/closed", { threadId });
      return;
    }
    case "turn/start":
      if (scenarioIs("outage")) {
        fail(request, 503, "OpenAI service unavailable");
        return;
      }
      startTurn(request);
      return;
    case "turn/interrupt": {
      const turnId = stringParam(params, "turnId");
      respond(request, {});
      const turn = activeTurns.get(turnId);
      if (turn) {
        completeTurn(turn, "interrupted", "", { message: "fake interruption" });
      }
      return;
    }
    case "hang":
      return;
    case "die":
      setTimeout(() => process.exit(17), 10);
      return;
    case "close-stdin":
      respond(request, {});
      setTimeout(() => process.stdin.destroy(), 10);
      return;
    case "thread/delete":
    case "thread/archive":
    case "thread/unarchive":
    case "thread/name/set":
    case "thread/goal/set":
      respond(request, {});
      return;
    default:
      fail(request, -32601, `Invalid API request: ${method}`);
  }
}

const input = createInterface({ input: process.stdin, crlfDelay: Infinity });
process.stdin.on("error", () => undefined);
input.on("line", (line: string) => {
  if (!line.trim()) {
    return;
  }
  let parsed: unknown;
  try {
    parsed = JSON.parse(line);
  } catch {
    failInvalid(null, -32600, "Invalid JSON-RPC request");
    return;
  }

  if (!isRecord(parsed)) {
    failInvalid(null, -32600, "Invalid JSON-RPC request");
    return;
  }

  if (typeof parsed.method === "string") {
    if (
      (parsed.id !== undefined && !isRpcId(parsed.id)) ||
      (parsed.params !== undefined &&
        parsed.params !== null &&
        !isRecord(parsed.params))
    ) {
      const id = isRpcId(parsed.id) ? parsed.id : null;
      const code =
        parsed.params !== undefined &&
        parsed.params !== null &&
        !isRecord(parsed.params)
          ? -32602
          : -32600;
      const message =
        code === -32602 ? "Invalid params" : "Invalid JSON-RPC request";
      failInvalid(id, code, message);
      return;
    }

    const request: RpcRequest = {
      method: parsed.method,
      params:
        parsed.params === undefined || parsed.params === null
          ? {}
          : parsed.params,
    };
    if (isRpcId(parsed.id)) {
      request.id = parsed.id;
    }
    handleRequest(request);
    return;
  }

  if (
    Object.prototype.hasOwnProperty.call(parsed, "method") ||
    !isRpcId(parsed.id) ||
    (!Object.prototype.hasOwnProperty.call(parsed, "result") &&
      !Object.prototype.hasOwnProperty.call(parsed, "error"))
  ) {
    const id = isRpcId(parsed.id) ? parsed.id : null;
    failInvalid(id, -32600, "Invalid JSON-RPC request");
    return;
  }

  handleResponse(parsed);
});
