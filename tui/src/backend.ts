import { spawn, type Subprocess } from "bun"
import type { RpcEvent } from "./state"

export class Backend {
  private proc?: Subprocess<"pipe", "pipe", "inherit">
  private buffer = ""

  constructor(private onEvent: (event: RpcEvent) => void) {}

  start(model?: string) {
    const root = process.env.RL_INTERN_PROJECT_ROOT ?? process.cwd()
    this.proc = spawn({
      cmd: ["uv", "run", "python", "-m", "rl_intern.rpc"],
      cwd: root,
      stdin: "pipe",
      stdout: "pipe",
      stderr: "inherit",
    })
    this.readLoop()
    this.send({
      type: "start_run",
      id: "start",
      model,
    })
  }

  send(payload: Record<string, unknown>) {
    this.proc?.stdin.write(JSON.stringify(payload) + "\n")
  }

  submit(text: string) {
    this.send({ type: "user_input", id: crypto.randomUUID(), text })
  }

  approve(toolCallId: unknown, approved: boolean) {
    this.send({
      type: "exec_approval",
      id: crypto.randomUUID(),
      approvals: [{ tool_call_id: toolCallId, approved, feedback: null }],
    })
  }

  interrupt() {
    this.send({ type: "interrupt", id: crypto.randomUUID() })
  }

  shutdown() {
    this.send({ type: "shutdown", id: "shutdown" })
  }

  private async readLoop() {
    if (!this.proc?.stdout) return
    const reader = this.proc.stdout.getReader()
    const decoder = new TextDecoder()
    while (true) {
      const { done, value } = await reader.read()
      if (done) break
      this.buffer += decoder.decode(value)
      let index = this.buffer.indexOf("\n")
      while (index >= 0) {
        const line = this.buffer.slice(0, index)
        this.buffer = this.buffer.slice(index + 1)
        if (line.trim()) this.onEvent(JSON.parse(line))
        index = this.buffer.indexOf("\n")
      }
    }
  }
}
