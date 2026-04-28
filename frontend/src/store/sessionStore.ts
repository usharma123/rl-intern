import { create } from 'zustand';
import { persist } from 'zustand/middleware';

export interface SessionMeta {
  id: string;
  title: string;
  createdAt: string;
  needsAttention?: boolean;
}

interface SessionStore {
  sessions: SessionMeta[];
  activeSessionId: string | null;

  createSession: (id: string) => void;
  deleteSession: (id: string) => void;
  switchSession: (id: string) => void;
  updateSessionTitle: (id: string, title: string) => void;
  setNeedsAttention: (id: string, needs: boolean) => void;
}

export const useSessionStore = create<SessionStore>()(
  persist(
    (set, get) => ({
      sessions: [],
      activeSessionId: null,

      createSession: (id) => {
        const meta: SessionMeta = {
          id,
          title: `Session ${get().sessions.length + 1}`,
          createdAt: new Date().toISOString(),
          needsAttention: false,
        };
        set((s) => ({ sessions: [...s.sessions, meta], activeSessionId: id }));
      },

      deleteSession: (id) => {
        set((s) => {
          const remaining = s.sessions.filter((x) => x.id !== id);
          const nextActive =
            s.activeSessionId === id ? remaining[remaining.length - 1]?.id ?? null : s.activeSessionId;
          return { sessions: remaining, activeSessionId: nextActive };
        });
      },

      switchSession: (id) => {
        set((s) => ({
          activeSessionId: id,
          sessions: s.sessions.map((x) => (x.id === id ? { ...x, needsAttention: false } : x)),
        }));
      },

      updateSessionTitle: (id, title) => {
        set((s) => ({ sessions: s.sessions.map((x) => (x.id === id ? { ...x, title } : x)) }));
      },

      setNeedsAttention: (id, needs) => {
        set((s) => ({
          sessions: s.sessions.map((x) => (x.id === id ? { ...x, needsAttention: needs } : x)),
        }));
      },
    }),
    {
      name: 'rl-intern-sessions',
      partialize: (s) => ({ sessions: s.sessions, activeSessionId: s.activeSessionId }),
    },
  ),
);
