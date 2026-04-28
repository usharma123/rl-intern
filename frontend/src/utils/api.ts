const API_BASE_URL =
  import.meta.env.VITE_RL_INTERN_API_URL?.replace(/\/$/, '') ?? 'http://127.0.0.1:8765';

export function apiUrl(path: string): string {
  return new URL(path, `${API_BASE_URL}/`).toString();
}

export function apiWsUrl(path: string): URL {
  const url = new URL(apiUrl(path));
  url.protocol = url.protocol === 'https:' ? 'wss:' : 'ws:';
  return url;
}

/** Thin wrapper so HTTP calls use the local run server directly under Bun. */
export async function apiFetch(path: string, init?: RequestInit): Promise<Response> {
  return fetch(apiUrl(path), {
    headers: { 'Content-Type': 'application/json' },
    ...init,
  });
}
