import { useCallback, useEffect, useRef, useState } from 'react';
import type {
  ArtifactDashboardState,
  ChatMessage,
  JobDashboardState,
  PlanDashboardState,
  RpcEvent,
  ToolBlock,
  ToolStatus,
} from '@/types/events';
import { useLayoutStore } from '@/store/layoutStore';
import { apiFetch, apiWsUrl } from '@/utils/api';

export type ConnectionState = 'connecting' | 'open' | 'closed' | 'error';

export interface UseAgentChat {
  messages: ChatMessage[];
  status: string;
  connection: ConnectionState;
  bridgeOpen: boolean;
  agentReady: boolean;
  pendingApprovals: Array<Record<string, unknown>>;
  plan: PlanDashboardState;
  jobs: JobDashboardState[];
  artifacts: ArtifactDashboardState;
  sendMessage: (text: string) => void;
  approve: (toolCallId: unknown, approved: boolean) => void;
  interrupt: () => void;
}

let counter = 0;
const nextId = () => `m${++counter}`;

function cryptoRandom(): string {
  if (typeof crypto !== 'undefined' && 'randomUUID' in crypto) {
    return (crypto as Crypto).randomUUID();
  }
  return Math.random().toString(36).slice(2);
}

export function useAgentChat(sessionId: string): UseAgentChat {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [status, setStatus] = useState<string>('connecting');
  const [connection, setConnection] = useState<ConnectionState>('connecting');
  const [bridgeOpen, setBridgeOpen] = useState(false);
  const [agentReady, setAgentReady] = useState(false);
  const [pendingApprovals, setPendingApprovals] = useState<Array<Record<string, unknown>>>([]);
  const [plan, setPlan] = useState<PlanDashboardState>({});
  const [jobs, setJobs] = useState<JobDashboardState[]>([]);
  const [artifacts, setArtifacts] = useState<ArtifactDashboardState>({});
  const wsRef = useRef<WebSocket | null>(null);
  const model = useLayoutStore.getState().model;

  // Open one WS per session id. Stays open for the session's lifetime.
  useEffect(() => {
    let cancelled = false;
    async function loadHistory() {
      setMessages([]);
      setPendingApprovals([]);
      setPlan({});
      setJobs([]);
      setArtifacts({});
      try {
        const res = await apiFetch(`/runs/${sessionId}/events.jsonl`);
        if (!res.ok) return;
        const text = await res.text();
        if (cancelled) return;
        const events = text
          .split(/\r?\n/)
          .map((line) => line.trim())
          .filter(Boolean)
          .map((line) => JSON.parse(line) as RpcEvent);
        for (const event of events) {
          reduce(event, setMessages, setPendingApprovals, setPlan, setJobs, setArtifacts);
        }
      } catch {
        /* History is best-effort; a brand-new run won't have events yet. */
      }
    }
    loadHistory();

    const url = apiWsUrl('/api/ws/chat');
    url.searchParams.set('session_id', sessionId);
    if (model) url.searchParams.set('model', model);

    const ws = new WebSocket(url.toString());
    wsRef.current = ws;
    queueMicrotask(() => {
      if (wsRef.current !== ws) return;
      setConnection('connecting');
      setStatus('connecting');
    });

    ws.addEventListener('open', () => setConnection('open'));
    ws.addEventListener('close', () => {
      setConnection('closed');
      setBridgeOpen(false);
      setAgentReady(false);
    });
    ws.addEventListener('error', () => setConnection('error'));
    ws.addEventListener('message', (event) => {
      let parsed: RpcEvent;
      try {
        parsed = JSON.parse(event.data);
      } catch {
        return;
      }

      // Bridge-side events from the FastAPI WS shim
      if (parsed.type === 'bridge_open') {
        setBridgeOpen(true);
        return;
      }
      if (parsed.type === 'bridge_log') {
        const level = (parsed.data?.level as 'info' | 'warn' | 'error') ?? 'warn';
        const source = (parsed.data?.source as string | undefined) ?? 'bridge';
        const line = String(parsed.data?.line ?? '');
        setMessages((prev) => [
          ...prev,
          { kind: 'system', id: nextId(), text: line, ts: Date.now(), level, source },
        ]);
        return;
      }
      if (parsed.type === 'bridge_exit') {
        const code = parsed.data?.code as number | undefined;
        const text = `agent process exited (code ${code ?? '?'})`;
        setMessages((prev) => [
          ...prev,
          {
            kind: 'system',
            id: nextId(),
            text,
            ts: Date.now(),
            level: code === 0 ? 'info' : 'error',
            source: 'exit',
          },
        ]);
        setBridgeOpen(false);
        setAgentReady(false);
        return;
      }

      // First sign of life from the agent itself
      if (parsed.type === 'ready') setAgentReady(true);

      if (parsed.status) setStatus(parsed.status);
      else if (parsed.type) setStatus(parsed.type);
      reduce(parsed, setMessages, setPendingApprovals, setPlan, setJobs, setArtifacts);
    });

    return () => {
      cancelled = true;
      try {
        ws.send(JSON.stringify({ type: 'shutdown', id: 'shutdown' }));
      } catch {
        /* ignore */
      }
      ws.close();
      wsRef.current = null;
    };
  }, [sessionId, model]);

  const sendMessage = useCallback((text: string) => {
    const trimmed = text.trim();
    if (!trimmed) return;
    setMessages((prev) => [
      ...prev,
      { kind: 'user', id: nextId(), text: trimmed, ts: Date.now() },
    ]);
    const ws = wsRef.current;
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ type: 'user_input', id: cryptoRandom(), text: trimmed }));
    }
  }, []);

  const approve = useCallback((toolCallId: unknown, approved: boolean) => {
    const ws = wsRef.current;
    if (!ws || ws.readyState !== WebSocket.OPEN) return;
    ws.send(
      JSON.stringify({
        type: 'exec_approval',
        id: cryptoRandom(),
        approvals: [{ tool_call_id: toolCallId, approved, feedback: null }],
      }),
    );
    setPendingApprovals([]);
  }, []);

  const interrupt = useCallback(() => {
    const ws = wsRef.current;
    if (!ws || ws.readyState !== WebSocket.OPEN) return;
    ws.send(JSON.stringify({ type: 'interrupt', id: cryptoRandom() }));
  }, []);

  return {
    messages,
    status,
    connection,
    bridgeOpen,
    agentReady,
    pendingApprovals,
    plan,
    jobs,
    artifacts,
    sendMessage,
    approve,
    interrupt,
  };
}

// ---------- reducer ----------

function reduce(
  event: RpcEvent,
  setMessages: React.Dispatch<React.SetStateAction<ChatMessage[]>>,
  setApprovals: React.Dispatch<React.SetStateAction<Array<Record<string, unknown>>>>,
  setPlan: React.Dispatch<React.SetStateAction<PlanDashboardState>>,
  setJobs: React.Dispatch<React.SetStateAction<JobDashboardState[]>>,
  setArtifacts: React.Dispatch<React.SetStateAction<ArtifactDashboardState>>,
) {
  switch (event.type) {
    case 'user_input': {
      const text = String(event.content ?? event.data?.text ?? '');
      if (!text) return;
      setMessages((prev) => appendUserText(prev, text, event.timestamp));
      return;
    }
    case 'assistant_chunk':
    case 'assistant_message': {
      const content = String(event.data?.content ?? '');
      if (!content) return;
      setMessages((prev) => appendAssistantText(prev, content));
      return;
    }
    case 'tool_call': {
      const tool: ToolBlock = {
        id: nextId(),
        toolCallId: event.tool_call_id ?? undefined,
        name: String(event.tool ?? event.data?.tool ?? 'tool'),
        input: (event.input ?? event.data?.arguments ?? {}) as Record<string, unknown>,
        status: 'running',
        artifacts: [],
        startedAt: Date.now(),
      };
      setMessages((prev) => appendToolToAssistant(prev, tool));
      return;
    }
    case 'tool_state_change': {
      const next = String(event.status ?? event.data?.state ?? '').toLowerCase() as ToolStatus;
      if (next !== 'success' && next !== 'error' && next !== 'running') return;
      setMessages((prev) => updateTool(prev, event.tool_call_id, (t) => ({ ...t, status: next })));
      return;
    }
    case 'tool_output': {
      const ok = String(event.status ?? '').toLowerCase() === 'success';
      const summary = summarizeOutput(event.output);
      const artifacts = collectArtifacts(event.output);
      setMessages((prev) =>
        updateTool(prev, event.tool_call_id, (t) => ({
          ...t,
          status: ok ? 'success' : 'error',
          summary,
          artifacts: [...t.artifacts, ...artifacts],
          durationMs: Date.now() - t.startedAt,
        })),
      );
      return;
    }
    case 'approval_required': {
      const tools = event.data?.tools;
      setApprovals(Array.isArray(tools) ? (tools as Array<Record<string, unknown>>) : []);
      return;
    }
    case 'plan_update': {
      const data = event.data ?? {};
      setPlan((prev) => ({
        ...prev,
        plan: (event.plan ?? data.plan ?? prev.plan) as Record<string, unknown> | undefined,
        stage: (event.stage ?? data.stage ?? prev.stage) as string | undefined,
        status: String(event.status ?? data.status ?? prev.status ?? 'updated'),
        result: data.result ?? prev.result,
      }));
      return;
    }
    case 'job_update': {
      const data = event.data ?? {};
      const job: JobDashboardState = {
        jobId: String(event.job_id ?? data.job_id ?? data.jobId ?? ''),
        backendId: String(event.backend_id ?? data.backend_id ?? ''),
        stage: typeof data.stage === 'string' ? data.stage : undefined,
        status: String(event.status ?? data.status ?? 'updated'),
        hardware: typeof data.hardware === 'string' ? data.hardware : undefined,
        error: typeof data.error === 'string' ? data.error : undefined,
      };
      setJobs((prev) => {
        const key = job.jobId || job.backendId;
        if (!key) return [job, ...prev].slice(0, 8);
        const idx = prev.findIndex((j) => (j.jobId || j.backendId) === key);
        if (idx === -1) return [job, ...prev].slice(0, 8);
        return prev.map((j, i) => (i === idx ? { ...j, ...job } : j));
      });
      return;
    }
    case 'artifact_manifest': {
      const manifest = (event.manifest ?? event.data?.manifest) as Record<string, unknown>;
      setArtifacts({ manifest });
      return;
    }
    case 'error': {
      const message = String(event.data?.error ?? event.error ?? 'Unknown backend error');
      setMessages((prev) => [
        ...prev,
        { kind: 'error', id: nextId(), text: message, ts: Date.now() },
      ]);
      return;
    }
  }
}

function appendUserText(messages: ChatMessage[], text: string, timestamp?: string): ChatMessage[] {
  const last = messages[messages.length - 1];
  if (last?.kind === 'user' && last.text === text) return messages;
  return [
    ...messages,
    {
      kind: 'user',
      id: nextId(),
      text,
      ts: timestamp ? new Date(timestamp).getTime() : Date.now(),
    },
  ];
}

function appendAssistantText(messages: ChatMessage[], chunk: string): ChatMessage[] {
  const last = messages[messages.length - 1];
  if (last && last.kind === 'assistant' && last.tools.length === 0) {
    return [...messages.slice(0, -1), { ...last, text: last.text + chunk }];
  }
  return [
    ...messages,
    { kind: 'assistant', id: nextId(), text: chunk, ts: Date.now(), tools: [] },
  ];
}

function appendToolToAssistant(messages: ChatMessage[], tool: ToolBlock): ChatMessage[] {
  const last = messages[messages.length - 1];
  if (last && last.kind === 'assistant') {
    return [...messages.slice(0, -1), { ...last, tools: [...last.tools, tool] }];
  }
  return [
    ...messages,
    { kind: 'assistant', id: nextId(), text: '', ts: Date.now(), tools: [tool] },
  ];
}

function updateTool(
  messages: ChatMessage[],
  toolCallId: string | null | undefined,
  patch: (t: ToolBlock) => ToolBlock,
): ChatMessage[] {
  for (let i = messages.length - 1; i >= 0; i -= 1) {
    const m = messages[i];
    if (m.kind !== 'assistant') continue;
    const idx = findToolIndex(m.tools, toolCallId);
    if (idx === -1) continue;
    const tools = m.tools.map((t, j) => (j === idx ? patch(t) : t));
    return [...messages.slice(0, i), { ...m, tools }, ...messages.slice(i + 1)];
  }
  return messages;
}

function findToolIndex(tools: ToolBlock[], toolCallId: string | null | undefined): number {
  if (toolCallId) {
    for (let i = tools.length - 1; i >= 0; i -= 1) {
      if (tools[i].toolCallId === toolCallId) return i;
    }
  }
  for (let i = tools.length - 1; i >= 0; i -= 1) {
    if (tools[i].status === 'running') return i;
  }
  return -1;
}

function collectArtifacts(output: unknown): string[] {
  if (!output || typeof output !== 'object') return [];
  const out: string[] = [];
  for (const [key, value] of Object.entries(output as Record<string, unknown>)) {
    if (key.endsWith('_path') && typeof value === 'string') out.push(value);
    if (key === 'artifacts' && Array.isArray(value)) {
      for (const v of value) if (typeof v === 'string') out.push(v);
    }
  }
  return out;
}

function summarizeOutput(output: unknown): string | undefined {
  if (output == null) return undefined;
  if (typeof output === 'string') {
    const first = output.trim().split(/\r?\n/)[0];
    return first.length > 140 ? first.slice(0, 139) + '…' : first;
  }
  if (typeof output === 'object') {
    const obj = output as Record<string, unknown>;
    if (typeof obj.message === 'string') return summarizeOutput(obj.message);
    if (typeof obj.summary === 'string') return summarizeOutput(obj.summary);
    if (typeof obj.mean_reward === 'number') {
      const std =
        typeof obj.std_reward === 'number' ? ` ± ${(obj.std_reward as number).toFixed(1)}` : '';
      return `mean reward ${(obj.mean_reward as number).toFixed(1)}${std}`;
    }
  }
  return undefined;
}
