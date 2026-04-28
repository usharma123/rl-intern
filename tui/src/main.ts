import { Box, Text, createCliRenderer } from "@opentui/core"
import { Backend } from "./backend"
import { applyEvent, createInitialState } from "./state"

const renderer = await createCliRenderer({ exitOnCtrlC: false })
const state = createInitialState()
const backend = new Backend((event) => {
  applyEvent(state, event)
  render()
})

const args = Bun.argv.slice(2)
const modelIndex = args.indexOf("--model")
const model = modelIndex >= 0 ? args[modelIndex + 1] : undefined
backend.start(model)

let prompt = ""

function render() {
  try {
    renderer.root.remove("app")
  } catch {
    // First render has nothing to remove.
  }
  renderer.root.add(
    Box(
      {
        id: "app",
        flexDirection: "column",
        gap: 1,
        padding: 1,
      },
      Text({
        content: `rl-intern TUI | run=${state.runId ?? "starting"} | status=${state.status}`,
        fg: "#7dd3fc",
      }),
      Box(
        { flexDirection: "row", gap: 2 },
        Box(
          { flexDirection: "column", borderStyle: "single", padding: 1, width: "55%", height: 18 },
          Text({ content: "Transcript", fg: "#a7f3d0" }),
          Text({ content: state.transcript.slice(-20).join("") || "Waiting for input..." }),
        ),
        Box(
          { flexDirection: "column", borderStyle: "single", padding: 1, width: "45%", height: 18 },
          Text({ content: "Tools / Artifacts", fg: "#fde68a" }),
          Text({ content: state.toolEvents.slice(-10).join("\n") || "No tools yet." }),
          Text({ content: "\nArtifacts:\n" + (state.artifacts.slice(-6).join("\n") || "None") }),
        ),
      ),
      state.pendingApprovals.length
        ? Text({
            content: `Approval required: press a to approve or r to reject ${String(state.pendingApprovals[0].tool)}`,
            fg: "#fca5a5",
          })
        : Text({ content: "Type a prompt. Enter submits. Backspace edits. Ctrl+C exits.", fg: "#cbd5e1" }),
      Box(
        { borderStyle: "single", padding: 1, width: "100%" },
        Text({ content: prompt || "train PPO on CartPole-v1 for 1000 timesteps...", fg: prompt ? "#ffffff" : "#64748b" }),
      ),
    ),
  )
  renderer.root.requestRender()
}

renderer.keyInput.on("keypress", (event) => {
  if (event.ctrl && event.name === "c") {
    backend.shutdown()
    process.exit(0)
  }
  if (event.name === "return" || event.name === "enter") {
    const value = prompt.trim()
    if (value) backend.submit(value)
    prompt = ""
    render()
    return
  }
  if (event.name === "backspace") {
    prompt = prompt.slice(0, -1)
    render()
    return
  }
  if (event.name === "a" && state.pendingApprovals.length) {
    backend.approve(state.pendingApprovals[0].tool_call_id, true)
    state.pendingApprovals = []
    render()
  }
  if (event.name === "r" && state.pendingApprovals.length) {
    backend.approve(state.pendingApprovals[0].tool_call_id, false)
    state.pendingApprovals = []
    render()
  }
  if (event.name === "v" && state.runId) {
    state.toolEvents.push(`Viewer: http://127.0.0.1:8765/runs/${state.runId}/viewer`)
    render()
  }
  if (!event.ctrl && !event.meta && event.sequence && event.sequence.length === 1) {
    prompt += event.sequence
    render()
  }
})

render()
