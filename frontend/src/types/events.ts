// Wire format coming off the FastAPI WebSocket bridge — same shape that
// rl_intern.events.normalize_event produces.

export interface RpcEvent {
  type: string;
  run_id?: string | null;
  run_dir?: string | null;
  turn_id?: string | null;
  tool?: string | null;
  tool_call_id?: string | null;
  status?: string | null;
  data?: Record<string, unknown>;
  input?: Record<string, unknown>;
  output?: unknown;
  content?: unknown;
  plan?: unknown;
  stage?: string | null;
  job_id?: string | null;
  backend_id?: string | null;
  manifest?: unknown;
  timestamp?: string;
  error?: string;
}

export type ToolStatus = 'running' | 'success' | 'error';

export interface ToolBlock {
  id: string;
  toolCallId?: string;
  name: string;
  input: Record<string, unknown>;
  status: ToolStatus;
  artifacts: string[];
  summary?: string;
  startedAt: number;
  durationMs?: number;
}

export type SystemLevel = 'info' | 'warn' | 'error';

export interface PlanDashboardState {
  plan?: Record<string, unknown>;
  stage?: string;
  status?: string;
  result?: unknown;
}

export interface JobDashboardState {
  jobId?: string;
  backendId?: string;
  stage?: string;
  status?: string;
  hardware?: string;
  error?: string;
}

export interface ArtifactDashboardState {
  manifest?: Record<string, unknown>;
}

export type ChatMessage =
  | { kind: 'user'; id: string; text: string; ts: number }
  | { kind: 'assistant'; id: string; text: string; ts: number; tools: ToolBlock[] }
  | { kind: 'error'; id: string; text: string; ts: number }
  | { kind: 'system'; id: string; text: string; ts: number; level: SystemLevel; source?: string };
