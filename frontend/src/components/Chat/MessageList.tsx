import { useEffect, useMemo, useRef } from 'react';
import { Box, Typography } from '@mui/material';
import UserMessage from './UserMessage';
import AssistantMessage from './AssistantMessage';
import LogGroup from './LogGroup';
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

type SystemMessage = Extract<ChatMessage, { kind: 'system' }>;
type Renderable =
  | { kind: 'msg'; message: ChatMessage }
  | { kind: 'sysGroup'; messages: SystemMessage[] };

// Collapse consecutive `system` messages from the same source into one group so
// streaming stderr (Python tracebacks especially) renders as a single block
// rather than a stack of single-line cards.
function buildRenderable(messages: ChatMessage[]): Renderable[] {
  const out: Renderable[] = [];
  let current: SystemMessage[] | null = null;
  let currentSource: string | undefined;

  const flush = () => {
    if (!current) return;
    if (current.length === 1) {
      out.push({ kind: 'msg', message: current[0] });
    } else {
      out.push({ kind: 'sysGroup', messages: current });
    }
    current = null;
    currentSource = undefined;
  };

  for (const m of messages) {
    if (m.kind === 'system') {
      const src = m.source ?? 'bridge';
      if (current && currentSource === src) {
        current.push(m);
      } else {
        flush();
        current = [m];
        currentSource = src;
      }
    } else {
      flush();
      out.push({ kind: 'msg', message: m });
    }
  }
  flush();
  return out;
}

interface Props {
  messages: ChatMessage[];
}

export default function MessageList({ messages }: Props) {
  const ref = useRef<HTMLDivElement>(null);
  const lastLength = useRef(0);

  const renderable = useMemo(() => buildRenderable(messages), [messages]);

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
      <Box sx={{ width: '100%', maxWidth: 880, mx: 'auto', mt: 'auto' }}>
        {renderable.map((entry) => {
          if (entry.kind === 'sysGroup') {
            return <LogGroup key={entry.messages[0].id} messages={entry.messages} />;
          }
          const m = entry.message;
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
          // single system message — keep the original compact inline style
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
