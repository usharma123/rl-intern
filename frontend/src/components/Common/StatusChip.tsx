import { Box } from '@mui/material';

export type StatusTone = 'idle' | 'ok' | 'warn' | 'err' | 'mute';

interface Props {
  label: string;
  tone?: StatusTone;
  dot?: boolean;
}

const PALETTE: Record<StatusTone, { fg: string; border: string; bg: string }> = {
  idle: {
    fg: 'var(--text-muted)',
    border: 'var(--border-strong)',
    bg: 'transparent',
  },
  ok: {
    fg: 'var(--accent)',
    border: 'var(--accent-border)',
    bg: 'var(--accent-soft)',
  },
  warn: {
    fg: 'var(--warn)',
    border: 'var(--warn-border)',
    bg: 'var(--warn-soft)',
  },
  err: {
    fg: 'var(--error)',
    border: 'var(--error-border)',
    bg: 'var(--error-soft)',
  },
  mute: {
    fg: 'var(--text-faint)',
    border: 'var(--border)',
    bg: 'transparent',
  },
};

export default function StatusChip({ label, tone = 'idle', dot = true }: Props) {
  const c = PALETTE[tone];
  return (
    <Box
      component="span"
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
        color: c.fg,
        bgcolor: c.bg,
        border: `1px solid ${c.border}`,
        borderRadius: 2,
        whiteSpace: 'nowrap',
      }}
    >
      {dot && (
        <Box
          component="span"
          sx={{
            width: 5,
            height: 5,
            borderRadius: '50%',
            bgcolor: 'currentColor',
            display: 'inline-block',
            boxShadow: tone === 'ok' ? '0 0 6px currentColor' : 'none',
          }}
        />
      )}
      <Box component="span">{label}</Box>
    </Box>
  );
}
