import { useState } from 'react';
import { Box, CircularProgress, Collapse, Typography } from '@mui/material';
import KeyboardArrowDownIcon from '@mui/icons-material/KeyboardArrowDown';
import KeyboardArrowRightIcon from '@mui/icons-material/KeyboardArrowRight';
import type { ToolBlock } from '@/types/events';
import StatusChip, { type StatusTone } from '@/components/Common/StatusChip';

interface Props {
  tool: ToolBlock;
}

function truncate(s: string, max: number) {
  return s.length > max ? s.slice(0, Math.max(0, max - 1)) + '…' : s;
}

function formatValue(v: unknown, max = 56): string {
  if (v == null) return '∅';
  if (typeof v === 'string') return truncate(v, max);
  if (typeof v === 'number' || typeof v === 'boolean') return String(v);
  if (Array.isArray(v)) return truncate(JSON.stringify(v), max);
  if (typeof v === 'object') {
    try {
      return truncate(JSON.stringify(v), max);
    } catch {
      return '{…}';
    }
  }
  return truncate(String(v), max);
}

function formatDuration(ms?: number): string {
  if (ms == null || ms <= 0) return '';
  if (ms < 1000) return `${ms}ms`;
  const s = ms / 1000;
  if (s < 60) return `${s.toFixed(s < 10 ? 1 : 0)}s`;
  const m = Math.floor(s / 60);
  return `${m}m${Math.round(s - m * 60)}s`;
}

function statusInfo(status: ToolBlock['status']): { label: string; tone: StatusTone } {
  if (status === 'success') return { label: 'OK', tone: 'ok' };
  if (status === 'error') return { label: 'ERR', tone: 'err' };
  return { label: 'RUN', tone: 'warn' };
}

export default function ToolCallGroup({ tool }: Props) {
  const [open, setOpen] = useState(false);
  const args = Object.entries(tool.input ?? {});
  const duration = formatDuration(tool.durationMs);
  const status = statusInfo(tool.status);

  return (
    <Box
      sx={{
        my: 1,
        background: 'var(--surface)',
        border: '1px solid var(--border)',
        borderRadius: 'var(--radius-md)',
        overflow: 'hidden',
        transition: 'border-color 100ms ease, background 100ms ease',
        '&:hover': { borderColor: 'var(--border-strong)' },
      }}
    >
      <Box
        onClick={() => setOpen((o) => !o)}
        sx={{
          display: 'flex',
          alignItems: 'center',
          gap: 1.25,
          px: 1.25,
          py: 0.875,
          cursor: 'pointer',
          userSelect: 'none',
          '&:hover': { background: 'var(--surface-hover)' },
        }}
      >
        {open ? (
          <KeyboardArrowDownIcon sx={{ fontSize: 16, color: 'var(--text-faint)' }} />
        ) : (
          <KeyboardArrowRightIcon sx={{ fontSize: 16, color: 'var(--text-faint)' }} />
        )}

        {tool.status === 'running' ? (
          <Box sx={{ width: 56, display: 'flex', justifyContent: 'flex-start' }}>
            <Box
              sx={{
                display: 'inline-flex',
                alignItems: 'center',
                gap: 0.75,
                px: 0.875,
                height: 20,
                fontFamily: 'var(--font-mono)',
                fontSize: '0.66rem',
                fontWeight: 600,
                letterSpacing: '0.12em',
                textTransform: 'uppercase',
                color: 'var(--warn)',
                bgcolor: 'var(--warn-soft)',
                border: '1px solid var(--warn-border)',
                borderRadius: 2,
              }}
            >
              <CircularProgress size={8} thickness={6} sx={{ color: 'var(--warn)' }} />
              RUN
            </Box>
          </Box>
        ) : (
          <Box sx={{ width: 56 }}>
            <StatusChip label={status.label} tone={status.tone} dot />
          </Box>
        )}

        <Typography
          sx={{
            fontFamily: 'var(--font-mono)',
            fontWeight: 600,
            fontSize: '0.78rem',
            color: 'var(--text)',
            letterSpacing: '-0.005em',
            flexShrink: 0,
          }}
        >
          {tool.name}
        </Typography>

        {!open && args.length > 0 && (
          <Typography
            sx={{
              fontFamily: 'var(--font-mono)',
              fontSize: '0.74rem',
              color: 'var(--text-muted)',
              flex: 1,
              overflow: 'hidden',
              textOverflow: 'ellipsis',
              whiteSpace: 'nowrap',
              minWidth: 0,
            }}
          >
            {args
              .map(
                ([k, v]) => `${k}=${typeof v === 'string' ? `"${formatValue(v, 24)}"` : formatValue(v, 24)}`,
              )
              .join('  ')}
          </Typography>
        )}

        {duration && (
          <Typography
            sx={{
              ml: 'auto',
              fontFamily: 'var(--font-mono)',
              fontSize: '0.7rem',
              color: 'var(--text-faint)',
              flexShrink: 0,
            }}
          >
            {duration}
          </Typography>
        )}
      </Box>

      <Collapse in={open} unmountOnExit>
        <Box sx={{ px: 1.5, py: 1.25, borderTop: '1px solid var(--border)' }}>
          {args.length > 0 && (
            <Box sx={{ mb: tool.summary || tool.artifacts.length ? 1.5 : 0 }}>
              <Typography
                sx={{
                  fontFamily: 'var(--font-mono)',
                  fontSize: '0.62rem',
                  fontWeight: 600,
                  letterSpacing: '0.16em',
                  color: 'var(--text-faint)',
                  mb: 0.5,
                }}
              >
                ARGS
              </Typography>
              <Box
                component="pre"
                sx={{
                  m: 0,
                  fontFamily: 'var(--font-mono)',
                  fontSize: '0.74rem',
                  color: 'var(--text)',
                  background: 'var(--code-panel-bg)',
                  border: '1px solid var(--border)',
                  borderRadius: 'var(--radius-sm)',
                  p: 1.25,
                  overflow: 'auto',
                  maxHeight: 240,
                  lineHeight: 1.55,
                }}
              >
                {JSON.stringify(tool.input, null, 2)}
              </Box>
            </Box>
          )}

          {tool.summary && (
            <Box sx={{ mb: tool.artifacts.length ? 1.5 : 0 }}>
              <Typography
                sx={{
                  fontFamily: 'var(--font-mono)',
                  fontSize: '0.62rem',
                  fontWeight: 600,
                  letterSpacing: '0.16em',
                  color: 'var(--text-faint)',
                  mb: 0.5,
                }}
              >
                RESULT
              </Typography>
              <Typography
                sx={{
                  fontFamily: 'var(--font-mono)',
                  fontSize: '0.78rem',
                  color: 'var(--text)',
                }}
              >
                {tool.summary}
              </Typography>
            </Box>
          )}

          {tool.artifacts.length > 0 && (
            <Box>
              <Typography
                sx={{
                  fontFamily: 'var(--font-mono)',
                  fontSize: '0.62rem',
                  fontWeight: 600,
                  letterSpacing: '0.16em',
                  color: 'var(--text-faint)',
                  mb: 0.5,
                }}
              >
                ARTIFACTS · {tool.artifacts.length}
              </Typography>
              <Box sx={{ display: 'flex', flexDirection: 'column', gap: 0.25 }}>
                {tool.artifacts.map((a) => (
                  <Box
                    key={a}
                    sx={{
                      display: 'flex',
                      alignItems: 'center',
                      gap: 1,
                      fontFamily: 'var(--font-mono)',
                      fontSize: '0.74rem',
                      color: 'var(--text)',
                    }}
                  >
                    <Box component="span" sx={{ color: 'var(--accent)' }}>
                      ◇
                    </Box>
                    <Box component="span" sx={{ wordBreak: 'break-all' }}>
                      {a}
                    </Box>
                  </Box>
                ))}
              </Box>
            </Box>
          )}
        </Box>
      </Collapse>
    </Box>
  );
}
