import { useCallback, useEffect, useMemo, useState } from 'react';
import { Alert, Box, Button, CircularProgress, TextField, Typography } from '@mui/material';
import ArrowForwardIcon from '@mui/icons-material/ArrowForward';
import { useSessionStore } from '@/store/sessionStore';
import { useLayoutStore } from '@/store/layoutStore';
import { apiFetch } from '@/utils/api';
import StatusChip from '@/components/Common/StatusChip';

const FEATURES = [
  ['ENV', 'gymnasium · classic-control · box2d'],
  ['ALGO', 'PPO · A2C · SAC · TD3 · custom'],
  ['EVAL', 'mean reward · rollouts · reports'],
  ['RUNNER', 'local · modal'],
] as const;

type SetupStatus = {
  openrouter: boolean;
  huggingface: boolean;
  modalInstalled: boolean;
  modalAuthenticated: boolean;
  modalAppsDeployed: boolean;
};

export default function WelcomeScreen() {
  const createSession = useSessionStore((s) => s.createSession);
  const model = useLayoutStore((s) => s.model);
  const setModel = useLayoutStore((s) => s.setModel);
  const [isCreating, setIsCreating] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [setupStatus, setSetupStatus] = useState<SetupStatus | null>(null);

  useEffect(() => {
    let cancelled = false;
    void apiFetch('/api/setup/status')
      .then(async (res) => {
        if (!res.ok) return;
        const data = (await res.json()) as SetupStatus;
        if (!cancelled) setSetupStatus(data);
      })
      .catch(() => {});
    return () => {
      cancelled = true;
    };
  }, []);

  const missingSetup = useMemo(() => {
    if (!setupStatus) return [];
    const missing: string[] = [];
    if (!setupStatus.openrouter) missing.push('OpenRouter key');
    if (!setupStatus.huggingface) missing.push('Hugging Face token');
    if (!setupStatus.modalInstalled) missing.push('Modal dependency');
    if (!setupStatus.modalAuthenticated) missing.push('Modal auth');
    if (!setupStatus.modalAppsDeployed) missing.push('Modal apps');
    return missing;
  }, [setupStatus]);

  const handleStartSession = useCallback(async () => {
    if (isCreating) return;
    setIsCreating(true);
    setError(null);
    try {
      const res = await apiFetch('/api/session', { method: 'POST' });
      if (!res.ok) {
        setError('Could not reach the run server. Is it running on :8765?');
        return;
      }
      const data = (await res.json()) as { session_id: string };
      createSession(data.session_id);
    } catch {
      setError('Could not reach the run server. Is it running on :8765?');
    } finally {
      setIsCreating(false);
    }
  }, [isCreating, createSession]);

  return (
    <Box
      sx={{
        width: '100%',
        height: '100%',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        overflowY: 'auto',
        p: { xs: 3, md: 6 },
      }}
    >
      <Box
        sx={{
          width: '100%',
          maxWidth: 560,
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'flex-start',
        }}
      >
        {/* Status row */}
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 4 }}>
          <StatusChip label="rl-intern · v0.1" tone="ok" />
          <Typography
            sx={{
              fontFamily: 'var(--font-mono)',
              fontSize: '0.7rem',
              color: 'var(--text-faint)',
              letterSpacing: '0.08em',
            }}
          >
            127.0.0.1:8765
          </Typography>
        </Box>

        {/* Wordmark */}
        <Box sx={{ display: 'flex', alignItems: 'baseline', gap: 0, mb: 1.5 }}>
          <Typography
            component="span"
            sx={{
              fontFamily: 'var(--font-mono)',
              fontSize: { xs: '2rem', md: '2.6rem' },
              fontWeight: 600,
              color: 'var(--text)',
              letterSpacing: '-0.04em',
              lineHeight: 1,
            }}
          >
            rl
          </Typography>
          <Typography
            component="span"
            sx={{
              fontFamily: 'var(--font-mono)',
              fontSize: { xs: '2rem', md: '2.6rem' },
              fontWeight: 600,
              color: 'var(--accent)',
              letterSpacing: '-0.04em',
              lineHeight: 1,
            }}
          >
            /
          </Typography>
          <Typography
            component="span"
            sx={{
              fontFamily: 'var(--font-mono)',
              fontSize: { xs: '2rem', md: '2.6rem' },
              fontWeight: 600,
              color: 'var(--text)',
              letterSpacing: '-0.04em',
              lineHeight: 1,
            }}
          >
            intern
          </Typography>
        </Box>

        <Typography
          sx={{
            fontFamily: 'var(--font-mono)',
            fontSize: '0.72rem',
            color: 'var(--accent)',
            letterSpacing: '0.18em',
            textTransform: 'uppercase',
            mb: 2.5,
          }}
        >
          the agent that trains agents
        </Typography>

        <Typography
          sx={{
            fontSize: '0.95rem',
            lineHeight: 1.65,
            color: 'var(--text-muted)',
            maxWidth: 480,
            mb: 4,
            '& strong': { color: 'var(--text)', fontWeight: 600 },
          }}
        >
          Spin up an <strong>RL agent</strong> that picks an algorithm, sets up the environment,
          trains a policy, evaluates it, records a rollout, and writes up the result. Instructions
          in. Trained model out.
        </Typography>

        {/* Feature grid */}
        <Box
          sx={{
            display: 'grid',
            gridTemplateColumns: { xs: '1fr', sm: '1fr 1fr' },
            columnGap: 3,
            rowGap: 1.25,
            width: '100%',
            mb: 4,
            pb: 3,
            borderBottom: '1px solid var(--border)',
          }}
        >
          {FEATURES.map(([k, v]) => (
            <Box key={k} sx={{ display: 'flex', gap: 1.25, alignItems: 'baseline' }}>
              <Typography
                sx={{
                  fontFamily: 'var(--font-mono)',
                  fontSize: '0.66rem',
                  color: 'var(--accent)',
                  letterSpacing: '0.12em',
                  fontWeight: 600,
                  width: 56,
                  flexShrink: 0,
                }}
              >
                {k}
              </Typography>
              <Typography
                sx={{
                  fontFamily: 'var(--font-mono)',
                  fontSize: '0.78rem',
                  color: 'var(--text-muted)',
                }}
              >
                {v}
              </Typography>
            </Box>
          ))}
        </Box>

        {/* Model field */}
        <Box sx={{ width: '100%', mb: 2 }}>
          <Typography
            sx={{
              fontFamily: 'var(--font-mono)',
              fontSize: '0.66rem',
              color: 'var(--text-muted)',
              letterSpacing: '0.12em',
              fontWeight: 600,
              mb: 0.75,
            }}
          >
            MODEL
          </Typography>
          <TextField
            fullWidth
            size="small"
            value={model}
            onChange={(e) => setModel(e.target.value)}
            placeholder="openrouter/openai/gpt-oss-120b:free"
            spellCheck={false}
            slotProps={{
              htmlInput: { autoCorrect: 'off', autoCapitalize: 'off' },
            }}
            sx={{
              '& .MuiInputBase-root': {
                fontFamily: 'var(--font-mono)',
                fontSize: '0.82rem',
              },
            }}
          />
          <Typography
            sx={{
              mt: 0.75,
              fontFamily: 'var(--font-mono)',
              fontSize: '0.7rem',
              color: 'var(--text-faint)',
              letterSpacing: '-0.005em',
            }}
          >
            litellm model string · openrouter keys need an{' '}
            <Box
              component="span"
              sx={{
                color: 'var(--accent)',
                bgcolor: 'var(--accent-soft)',
                px: 0.5,
                borderRadius: '3px',
              }}
            >
              openrouter/
            </Box>{' '}
            prefix
          </Typography>
        </Box>

        {/* CTA */}
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1.5, mt: 1, width: '100%' }}>
          <Button
            variant="contained"
            disabled={isCreating}
            onClick={handleStartSession}
            endIcon={
              isCreating ? (
                <CircularProgress size={14} thickness={5} sx={{ color: 'inherit' }} />
              ) : (
                <ArrowForwardIcon sx={{ fontSize: 16 }} />
              )
            }
            sx={{
              fontFamily: 'var(--font-mono)',
              fontSize: '0.78rem',
              letterSpacing: '0.04em',
              fontWeight: 600,
              px: 2.5,
              py: 1.1,
              textTransform: 'uppercase',
            }}
          >
            {isCreating ? 'starting' : 'start session'}
          </Button>
          <Typography
            sx={{
              fontFamily: 'var(--font-mono)',
              fontSize: '0.7rem',
              color: 'var(--text-faint)',
              letterSpacing: '0.06em',
            }}
          >
            ↵ to confirm
          </Typography>
        </Box>

        {missingSetup.length > 0 && (
          <Alert
            severity="warning"
            variant="outlined"
            sx={{
              mt: 3,
              width: '100%',
              borderColor: 'var(--warn-border)',
              color: 'var(--text)',
              fontSize: '0.82rem',
              fontFamily: 'var(--font-mono)',
              alignItems: 'center',
              '& .MuiAlert-icon': { color: 'var(--warn)' },
            }}
          >
            Missing setup: {missingSetup.join(', ')}. Run `cd frontend && bun run setup`.
          </Alert>
        )}

        {error && (
          <Alert
            severity="error"
            variant="outlined"
            onClose={() => setError(null)}
            sx={{
              mt: 3,
              width: '100%',
              borderColor: 'var(--error-border)',
              color: 'var(--text)',
              fontSize: '0.82rem',
              fontFamily: 'var(--font-mono)',
              alignItems: 'center',
              '& .MuiAlert-icon': { color: 'var(--error)' },
            }}
          >
            {error}
          </Alert>
        )}

        <Typography
          sx={{
            mt: 5,
            fontFamily: 'var(--font-mono)',
            fontSize: '0.66rem',
            color: 'var(--text-faint)',
            letterSpacing: '0.08em',
          }}
        >
          // sessions are stored locally in your browser
        </Typography>
      </Box>
    </Box>
  );
}
