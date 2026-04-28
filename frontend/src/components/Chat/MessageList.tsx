import { useEffect, useRef } from 'react';
import { Box, Typography } from '@mui/material';
import UserMessage from './UserMessage';
import AssistantMessage from './AssistantMessage';
import type { ChatMessage, SystemLevel } from '@/types/events';

const SYS_PALETTE: Record<SystemLevel, { fg: string; border: string; bg: string }> = {
  info: {
    fg: 'var(--text-muted)',
    border: 'var(--border-strong)',
    bg: 'transparent',
  },
  warn: {
    fg: 'var(--warn)',
    border: 'var(--warn-border)',
    bg: 'var(--warn-soft)',
  },
  error: {
    fg: 'var(--error)',
    border: 'var(--error-border)',
    bg: 'var(--error-soft)',
  },
};

interface Props {
  messages: ChatMessage[];
}

export default function MessageList({ messages }: Props) {
  const ref = useRef<HTMLDivElement>(null);
  const lastLength = useRef(0);

  useEffect(() => {
    const container = ref.current;
    if (!container) return;
    const last = messages[messages.length - 1];
    const justGrew = messages.length > lastLength.current;
    const lastIsAssistant = last?.kind === 'assistant';
    if (justGrew || lastIsAssistant) {
      container.scrollTo({ top: container.scrollHeight, behavior: 'smooth' });
    }
    lastLength.current = messages.length;
  }, [messages]);

  return (
    <Box
      ref={ref}
      sx={{
        flex: 1,
        overflowY: 'auto',
        px: { xs: 1.5, md: 2.5 },
        pt: 3,
        pb: 1.5,
        display: 'flex',
        flexDirection: 'column',
      }}
    >
      <Box sx={{ width: '100%', maxWidth: 880, mx: 'auto', flex: 1 }}>
        {messages.map((m) => {
          if (m.kind === 'user') return <UserMessage key={m.id} text={m.text} />;
          if (m.kind === 'assistant')
            return <AssistantMessage key={m.id} text={m.text} tools={m.tools} />;
          if (m.kind === 'error') {
            return (
              <Box
                key={m.id}
                sx={{
                  display: 'flex',
                  gap: 1.25,
                  alignItems: 'flex-start',
                  px: 1.5,
                  py: 1.25,
                  my: 1.25,
                  borderRadius: 'var(--radius-md)',
                  background: 'var(--error-soft)',
                  border: '1px solid var(--error-border)',
                }}
              >
                <Typography
                  sx={{
                    fontFamily: 'var(--font-mono)',
                    fontSize: '0.62rem',
                    fontWeight: 600,
                    letterSpacing: '0.16em',
                    color: 'var(--error)',
                    pt: '2px',
                    flexShrink: 0,
                  }}
                >
                  ERROR
                </Typography>
                <Typography
                  sx={{
                    fontFamily: 'var(--font-mono)',
                    fontSize: '0.82rem',
                    color: 'var(--text)',
                    lineHeight: 1.55,
                    wordBreak: 'break-word',
                  }}
                >
                  {m.text}
                </Typography>
              </Box>
            );
          }
          // system / bridge log
          const c = SYS_PALETTE[m.level];
          const tag = `${m.source ?? 'bridge'}/${m.level}`.toUpperCase();
          return (
            <Box
              key={m.id}
              sx={{
                display: 'flex',
                gap: 1.25,
                alignItems: 'flex-start',
                px: 1.25,
                py: 0.75,
                my: 0.75,
                borderRadius: 'var(--radius-sm)',
                background: c.bg,
                border: `1px solid ${c.border}`,
                fontFamily: 'var(--font-mono)',
              }}
            >
              <Typography
                sx={{
                  fontFamily: 'var(--font-mono)',
                  fontSize: '0.6rem',
                  fontWeight: 600,
                  letterSpacing: '0.14em',
                  color: c.fg,
                  pt: '2px',
                  flexShrink: 0,
                  width: 88,
                }}
              >
                [{tag}]
              </Typography>
              <Typography
                sx={{
                  fontFamily: 'var(--font-mono)',
                  fontSize: '0.78rem',
                  color: 'var(--text)',
                  lineHeight: 1.55,
                  wordBreak: 'break-word',
                  flex: 1,
                  minWidth: 0,
                }}
              >
                {m.text}
              </Typography>
            </Box>
          );
        })}
      </Box>
    </Box>
  );
}
