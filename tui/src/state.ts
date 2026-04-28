export type RpcEvent = {
  type: string
  run_id?: string | null
  run_dir?: string | null
  tool?: string | null
  status?: string | null
  data?: Record<string, unknown>
  input?: Record<string, unknown>
  output?: unknown
  timestamp?: string
}

export type AppState = {
  runId?: string
  runDir?: string
  transcript: string[]
  toolEvents: string[]
  artifacts: string[]
  pendingApprovals: Array<Record<string, unknown>>
  currentTool?: string
  status: string
}

export function createInitialState(): AppState {
  return {
    transcript: [],
    toolEvents: [],
    artifacts: [],
    pendingApprovals: [],
    status: "starting",
  }
}

export function applyEvent(state: AppState, event: RpcEvent): AppState {
  if (event.run_id) state.runId = event.run_id
  if (event.run_dir) state.runDir = event.run_dir
  state.status = event.status ?? event.type

  if (event.type === "assistant_chunk") {
    const content = String(event.data?.content ?? "")
    if (content) state.transcript.push(content)
  } else if (event.type === "tool_call") {
    state.currentTool = event.tool ?? undefined
    state.toolEvents.push(`CALL ${event.tool}: ${JSON.stringify(event.input ?? {})}`)
  } else if (event.type === "tool_output") {
    state.toolEvents.push(`${event.status?.toUpperCase() ?? "DONE"} ${event.tool}`)
    collectArtifacts(state, event.output)
  } else if (event.type === "approval_required") {
    const tools = event.data?.tools
    state.pendingApprovals = Array.isArray(tools) ? tools as Array<Record<string, unknown>> : []
  } else if (event.type === "error") {
    const message = String(event.data?.error ?? event.data?.message ?? "Unknown backend error")
    state.transcript.push(`\nError: ${message}\n`)
    state.toolEvents.push(`ERROR: ${message}`)
  } else if (event.type === "turn_complete") {
    state.currentTool = undefined
  }

  return state
}

function collectArtifacts(state: AppState, output: unknown) {
  if (!output || typeof output !== "object") return
  for (const [key, value] of Object.entries(output as Record<string, unknown>)) {
    if (key.endsWith("_path") && typeof value === "string") {
      state.artifacts.push(value)
    }
  }
}
