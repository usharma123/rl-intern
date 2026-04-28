import { Box } from '@mui/material';
import { useEffect, useState } from 'react';
import AppLayout from '@/components/Layout/AppLayout';
import WelcomeScreen from '@/components/WelcomeScreen/WelcomeScreen';
import { useSessionStore } from '@/store/sessionStore';
import { apiFetch } from '@/utils/api';

interface RunMetadata {
  run_id: string;
  created_at?: string;
  prompt?: string | null;
}

export default function App() {
  const sessions = useSessionStore((s) => s.sessions);
  const hydrateSessions = useSessionStore((s) => s.hydrateSessions);
  const [hydrated, setHydrated] = useState(false);
  const hasSessions = sessions.length > 0;

  useEffect(() => {
    let cancelled = false;
    async function hydrate() {
      try {
        const res = await apiFetch('/runs');
        if (!res.ok) return;
        const runs = (await res.json()) as RunMetadata[];
        if (cancelled) return;
        hydrateSessions(
          runs
            .filter((run) => run.run_id)
            .map((run, idx) => ({
              id: run.run_id,
              title: run.prompt || `Session ${idx + 1}`,
              createdAt: run.created_at ?? new Date().toISOString(),
              needsAttention: false,
            })),
        );
      } finally {
        if (!cancelled) setHydrated(true);
      }
    }
    hydrate();
    return () => {
      cancelled = true;
    };
  }, [hydrateSessions]);

  return (
    <Box sx={{ height: '100vh', display: 'flex' }}>
      {hasSessions || !hydrated ? <AppLayout /> : <WelcomeScreen />}
    </Box>
  );
}
