import { Box, Button, IconButton, TextField, Tooltip, Typography } from '@mui/material';
import AddIcon from '@mui/icons-material/Add';
import DeleteOutlineIcon from '@mui/icons-material/DeleteOutlineOutlined';
import { useState } from 'react';
import { useSessionStore } from '@/store/sessionStore';
import { useLayoutStore } from '@/store/layoutStore';
import { apiFetch } from '@/utils/api';

export default function Sidebar() {
  const { sessions, activeSessionId, switchSession, deleteSession, createSession } =
    useSessionStore();
  const model = useLayoutStore((s) => s.model);
  const setModel = useLayoutStore((s) => s.setModel);
  const [creating, setCreating] = useState(false);
  const [deleting, setDeleting] = useState<string | null>(null);
  const [draftModel, setDraftModel] = useState(model);

  const handleNewSession = async () => {
    if (creating) return;
    setCreating(true);
    try {
      const res = await apiFetch('/api/session', { method: 'POST' });
      if (!res.ok) return;
      const data = (await res.json()) as { session_id: string };
      createSession(data.session_id);
    } finally {
      setCreating(false);
    }
  };

  const handleDeleteSession = async (id: string) => {
    if (deleting) return;
    setDeleting(id);
    try {
      const res = await apiFetch(`/runs/${id}`, { method: 'DELETE' });
      if (res.ok || res.status === 404) deleteSession(id);
    } finally {
      setDeleting(null);
    }
  };

  const reversed = [...sessions].reverse();

  return (
    <Box sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
      {/* Brand */}
      <Box
        sx={{
          display: 'flex',
          alignItems: 'center',
          gap: 1,
          height: 48,
          px: 2,
          borderBottom: '1px solid var(--border)',
        }}
      >
        <Box
          sx={{
            width: 18,
            height: 18,
            borderRadius: '4px',
            bgcolor: 'var(--accent)',
            color: '#0a0b0c',
            fontFamily: 'var(--font-mono)',
            fontWeight: 700,
            fontSize: '0.6rem',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            letterSpacing: '-0.04em',
          }}
        >
          r/i
        </Box>
        <Typography
          sx={{
            fontFamily: 'var(--font-mono)',
            fontSize: '0.78rem',
            fontWeight: 600,
            color: 'var(--text)',
            letterSpacing: '-0.01em',
          }}
        >
          rl-intern
        </Typography>
      </Box>

      {/* New session */}
      <Box sx={{ p: 1.5 }}>
        <Button
          fullWidth
          startIcon={<AddIcon sx={{ fontSize: 16 }} />}
          onClick={handleNewSession}
          disabled={creating}
          variant="outlined"
          sx={{
            justifyContent: 'flex-start',
            fontFamily: 'var(--font-mono)',
            fontSize: '0.76rem',
            fontWeight: 500,
            letterSpacing: '0.02em',
            py: 0.875,
            color: 'var(--text)',
            borderColor: 'var(--border)',
            background: 'var(--surface)',
            textTransform: 'lowercase',
            '&:hover': {
              borderColor: 'var(--accent-border)',
              background: 'var(--accent-soft)',
              color: 'var(--accent)',
            },
            '&.Mui-disabled': {
              color: 'var(--text-faint)',
              borderColor: 'var(--border)',
            },
          }}
        >
          {creating ? 'starting…' : 'new session'}
        </Button>
      </Box>

      {/* Section label */}
      <Typography
        sx={{
          mx: 1.5,
          mt: 0.5,
          mb: 1,
          fontFamily: 'var(--font-mono)',
          fontSize: '0.62rem',
          color: 'var(--text-faint)',
          letterSpacing: '0.16em',
          fontWeight: 600,
        }}
      >
        SESSIONS · {sessions.length.toString().padStart(2, '0')}
      </Typography>

      <Box sx={{ flex: 1, overflowY: 'auto', px: 1, pb: 1 }}>
        {reversed.length === 0 ? (
          <Typography
            sx={{
              p: 1.5,
              fontFamily: 'var(--font-mono)',
              fontSize: '0.72rem',
              color: 'var(--text-faint)',
            }}
          >
            // none yet
          </Typography>
        ) : (
          reversed.map((s) => {
            const isActive = s.id === activeSessionId;
            return (
              <Box
                key={s.id}
                onClick={() => switchSession(s.id)}
                sx={{
                  position: 'relative',
                  display: 'flex',
                  alignItems: 'center',
                  gap: 1,
                  px: 1.25,
                  py: 0.875,
                  borderRadius: 'var(--radius-sm)',
                  cursor: 'pointer',
                  borderLeft: '2px solid',
                  borderLeftColor: isActive ? 'var(--accent)' : 'transparent',
                  background: isActive ? 'var(--accent-soft)' : 'transparent',
                  transition: 'background 80ms ease, border-color 80ms ease',
                  mb: 0.25,
                  '&:hover': {
                    background: isActive ? 'var(--accent-soft-2)' : 'var(--surface-hover)',
                    '& .delete-btn': { opacity: 0.85 },
                  },
                }}
              >
                <Box sx={{ flex: 1, minWidth: 0 }}>
                  <Typography
                    sx={{
                      fontSize: '0.82rem',
                      fontWeight: isActive ? 600 : 500,
                      color: isActive ? 'var(--text)' : 'var(--text-muted)',
                      overflow: 'hidden',
                      textOverflow: 'ellipsis',
                      whiteSpace: 'nowrap',
                      letterSpacing: '-0.005em',
                    }}
                  >
                    {s.title}
                  </Typography>
                  <Typography
                    sx={{
                      mt: 0.125,
                      fontFamily: 'var(--font-mono)',
                      fontSize: '0.68rem',
                      color: isActive ? 'var(--accent)' : 'var(--text-faint)',
                      overflow: 'hidden',
                      textOverflow: 'ellipsis',
                      whiteSpace: 'nowrap',
                      letterSpacing: 0,
                    }}
                  >
                    {s.id}
                  </Typography>
                </Box>
                {s.needsAttention && !isActive && (
                  <Box
                    sx={{
                      width: 6,
                      height: 6,
                      borderRadius: '50%',
                      bgcolor: 'var(--warn)',
                      boxShadow: '0 0 6px var(--warn)',
                      flexShrink: 0,
                    }}
                  />
                )}
                <Tooltip title="Delete">
                  <IconButton
                    className="delete-btn"
                    size="small"
                    onClick={(e) => {
                      e.stopPropagation();
                      handleDeleteSession(s.id);
                    }}
                    disabled={deleting === s.id}
                    sx={{
                      opacity: 0,
                      width: 22,
                      height: 22,
                      borderRadius: 'var(--radius-sm)',
                      transition: 'opacity 100ms ease, color 100ms ease',
                      color: 'var(--text-faint)',
                      '&:hover': { color: 'var(--error)', background: 'var(--error-soft)' },
                    }}
                  >
                    <DeleteOutlineIcon sx={{ fontSize: 14 }} />
                  </IconButton>
                </Tooltip>
              </Box>
            );
          })
        )}
      </Box>

      {/* Model footer */}
      <Box sx={{ p: 1.5, borderTop: '1px solid var(--border)' }}>
        <Typography
          sx={{
            fontFamily: 'var(--font-mono)',
            fontSize: '0.62rem',
            color: 'var(--text-faint)',
            letterSpacing: '0.16em',
            fontWeight: 600,
            mb: 0.75,
          }}
        >
          MODEL
        </Typography>
        <TextField
          fullWidth
          size="small"
          value={draftModel}
          onChange={(e) => setDraftModel(e.target.value)}
          onBlur={() => setModel(draftModel)}
          onKeyDown={(e) => {
            if (e.key === 'Enter') (e.target as HTMLInputElement).blur();
          }}
          spellCheck={false}
          slotProps={{
            htmlInput: { autoCorrect: 'off', autoCapitalize: 'off' },
          }}
          sx={{
            '& .MuiInputBase-root': {
              fontFamily: 'var(--font-mono)',
              fontSize: '0.74rem',
            },
          }}
        />
        <Typography
          sx={{
            mt: 0.75,
            fontFamily: 'var(--font-mono)',
            fontSize: '0.62rem',
            color: 'var(--text-faint)',
            letterSpacing: '0.04em',
            lineHeight: 1.5,
          }}
        >
          // applies to new sessions
        </Typography>
      </Box>
    </Box>
  );
}
