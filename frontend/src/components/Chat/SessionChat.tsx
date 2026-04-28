import { useEffect } from 'react';
import { Box, Button, Typography } from '@mui/material';
import CheckCircleOutlineIcon from '@mui/icons-material/CheckCircleOutlineOutlined';
import CancelOutlinedIcon from '@mui/icons-material/CancelOutlined';
import ArrowForwardIcon from '@mui/icons-material/ArrowForward';
import { useAgentChat } from '@/hooks/useAgentChat';
import { useSessionStore } from '@/store/sessionStore';
import MessageList from './MessageList';
import ChatInput from './ChatInput';
import StatusChip, { type StatusTone } from '@/components/Common/StatusChip';

const SUGGESTIONS = [
  'train PPO on CartPole-v1 for 1000 timesteps',
  'evaluate the trained policy for 5 episodes and record a rollout',
  'compare PPO vs A2C on LunarLander-v3 with a short report',
];

interface Props {
  sessionId: string;
  isActive: boolean;
}

interface ChipState {
  label: string;
  tone: StatusTone;
}

function statusToChip(
  status: string,
  connection: string,
  bridgeOpen: boolean,
  agentReady: boolean,
): ChipState {
  const s = status.toLowerCase();
  if (connection === 'connecting') return { label: 'WS CONNECTING', tone: 'mute' };
  if (connection === 'closed' || connection === 'error')
    return { label: 'OFFLINE', tone: 'err' };
  if (!bridgeOpen) return { label: 'BRIDGE STARTING', tone: 'mute' };
  if (!agentReady) return { label: 'AGENT BOOTING', tone: 'warn' };
  if (s.includes('error')) return { label: 'ERROR', tone: 'err' };
  if (s.includes('approval')) return { label: 'AWAITING APPROVAL', tone: 'warn' };
  if (s.includes('complete')) return { label: 'IDLE', tone: 'ok' };
  if (s.includes('ready')) return { label: 'READY', tone: 'ok' };
  if (
    s.includes('chunk') ||
    s.includes('running') ||
    s.includes('tool') ||
    s.includes('started')
  )
    return { label: 'RUNNING', tone: 'warn' };
  return { label: 'IDLE', tone: 'ok' };
}

export default function SessionChat({ sessionId, isActive }: Props) {
  const {
    messages,
    status,
    connection,
    bridgeOpen,
    agentReady,
    pendingApprovals,
    sendMessage,
    approve,
    interrupt,
  } = useAgentChat(sessionId);
  const { updateSessionTitle, setNeedsAttention, sessions } = useSessionStore();
  const session = sessions.find((s) => s.id === sessionId);

  // Auto-title from first user message
  useEffect(() => {
    if (!session) return;
    const first = messages.find((m) => m.kind === 'user');
    if (!first || first.kind !== 'user') return;
    if (!session.title.startsWith('Session ')) return;
    const t = first.text.trim();
    updateSessionTitle(sessionId, t.length > 48 ? t.slice(0, 47) + '…' : t);
  }, [messages, session, sessionId, updateSessionTitle]);

  // Approval bell while session is in background
  useEffect(() => {
    if (!isActive && pendingApprovals.length > 0) {
      setNeedsAttention(sessionId, true);
    }
  }, [pendingApprovals, isActive, sessionId, setNeedsAttention]);

  if (!isActive) return null;

  const approval = pendingApprovals[0];
  const isStreaming =
    status.toLowerCase().includes('chunk') ||
    status.toLowerCase().includes('running') ||
    status.toLowerCase().includes('tool') ||
    status.toLowerCase().includes('started');
  const disabled = connection !== 'open' || !bridgeOpen;
  const chip = statusToChip(status, connection, bridgeOpen, agentReady);

  // Empty state — Modal-flavored hero
  if (messages.length === 0) {
    const showRail = connection !== 'open' || !bridgeOpen || !agentReady;
    return (
      <Box sx={{ flex: 1, display: 'flex', flexDirection: 'column', minHeight: 0 }}>
        {showRail && <ConnectingRail tone={chip.tone} label={chip.label} />}
        <Box
          sx={{
            flex: 1,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            flexDirection: 'column',
            gap: 2,
            px: 3,
          }}
        >
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1 }}>
            <StatusChip label={chip.label} tone={chip.tone} />
            <Typography
              sx={{
                fontFamily: 'var(--font-mono)',
                fontSize: '0.68rem',
                color: 'var(--text-faint)',
                letterSpacing: '0.08em',
              }}
            >
              {sessionId}
            </Typography>
          </Box>

          <Typography
            sx={{
              fontFamily: 'var(--font-mono)',
              fontSize: { xs: '1.4rem', md: '1.7rem' },
              fontWeight: 600,
              color: 'var(--text)',
              letterSpacing: '-0.025em',
              textAlign: 'center',
            }}
          >
            what should we{' '}
            <Box component="span" sx={{ color: 'var(--accent)' }}>
              train
            </Box>{' '}
            today?
          </Typography>
          <Typography
            sx={{
              color: 'var(--text-muted)',
              textAlign: 'center',
              maxWidth: 540,
              fontSize: '0.9rem',
              lineHeight: 1.6,
            }}
          >
            rl-intern can set up environments, train policies, evaluate, record rollouts, and
            write up the result. Pick a starting prompt or ask anything.
          </Typography>
          <Box
            sx={{
              display: 'flex',
              flexDirection: 'column',
              gap: 0.875,
              mt: 1.5,
              maxWidth: 580,
              width: '100%',
            }}
          >
            {SUGGESTIONS.map((s) => (
              <Button
                key={s}
                onClick={() => sendMessage(s)}
                disabled={disabled}
                endIcon={<ArrowForwardIcon sx={{ fontSize: 14 }} />}
                sx={{
                  justifyContent: 'space-between',
                  textAlign: 'left',
                  px: 1.75,
                  py: 1.125,
                  fontFamily: 'var(--font-mono)',
                  fontSize: '0.78rem',
                  fontWeight: 500,
                  letterSpacing: '-0.005em',
                  color: 'var(--text-muted)',
                  background: 'var(--surface)',
                  border: '1px solid var(--border)',
                  borderRadius: 'var(--radius-md)',
                  textTransform: 'none',
                  transition: 'border-color 100ms, color 100ms, background 100ms',
                  '&:hover': {
                    borderColor: 'var(--accent-border)',
                    color: 'var(--text)',
                    background: 'var(--accent-soft)',
                    '& .MuiSvgIcon-root': { color: 'var(--accent)' },
                  },
                  '& .MuiSvgIcon-root': { color: 'var(--text-faint)' },
                }}
              >
                <span>{s}</span>
              </Button>
            ))}
          </Box>
        </Box>
        <ChatInput
          disabled={disabled}
          isStreaming={false}
          onSend={sendMessage}
          onInterrupt={interrupt}
        />
      </Box>
    );
  }

  // Active conversation
  const turns = messages.filter((m) => m.kind === 'user').length;
  const showConnectingRail =
    connection !== 'open' || !bridgeOpen || !agentReady;

  return (
    <Box sx={{ flex: 1, display: 'flex', flexDirection: 'column', minHeight: 0 }}>
      {/* Status bar pinned above the message list */}
      <Box
        sx={{
          display: 'flex',
          alignItems: 'center',
          gap: 1.5,
          px: { xs: 1.5, md: 2.5 },
          height: 32,
          borderBottom: '1px solid var(--border)',
          flexShrink: 0,
        }}
      >
        <StatusChip label={chip.label} tone={chip.tone} />
        <Typography
          sx={{
            fontFamily: 'var(--font-mono)',
            fontSize: '0.68rem',
            color: 'var(--text-faint)',
            letterSpacing: '0.04em',
          }}
        >
          {turns} {turns === 1 ? 'TURN' : 'TURNS'}
        </Typography>
      </Box>

      {showConnectingRail && <ConnectingRail tone={chip.tone} label={chip.label} />}

      <MessageList messages={messages} />

      {approval && (
        <Box
          sx={{
            mx: 'auto',
            mb: 1,
            maxWidth: 880,
            width: '100%',
            display: 'flex',
            alignItems: 'center',
            gap: 1.5,
            px: 1.75,
            py: 1.25,
            borderRadius: 'var(--radius-md)',
            border: '1px solid var(--warn-border)',
            background: 'var(--warn-soft)',
          }}
        >
          <StatusChip label="APPROVAL" tone="warn" />
          <Typography
            sx={{
              flex: 1,
              fontFamily: 'var(--font-mono)',
              fontSize: '0.82rem',
              color: 'var(--text)',
            }}
          >
            <Box component="span" sx={{ color: 'var(--accent)', fontWeight: 600 }}>
              {String(approval.tool)}
            </Box>{' '}
            needs approval to run
          </Typography>
          <Button
            size="small"
            startIcon={<CheckCircleOutlineIcon fontSize="small" />}
            onClick={() => approve(approval.tool_call_id, true)}
            sx={{
              fontFamily: 'var(--font-mono)',
              fontSize: '0.74rem',
              color: 'var(--accent)',
              border: '1px solid var(--accent-border)',
              px: 1.25,
              '&:hover': { background: 'var(--accent-soft)' },
            }}
          >
            approve
          </Button>
          <Button
            size="small"
            startIcon={<CancelOutlinedIcon fontSize="small" />}
            onClick={() => approve(approval.tool_call_id, false)}
            sx={{
              fontFamily: 'var(--font-mono)',
              fontSize: '0.74rem',
              color: 'var(--error)',
              border: '1px solid var(--error-border)',
              px: 1.25,
              '&:hover': { background: 'var(--error-soft)' },
            }}
          >
            reject
          </Button>
        </Box>
      )}

      <ChatInput
        disabled={disabled}
        isStreaming={isStreaming}
        onSend={sendMessage}
        onInterrupt={interrupt}
      />
    </Box>
  );
}

function ConnectingRail({ label, tone }: { label: string; tone: StatusTone }) {
  const color =
    tone === 'err'
      ? 'var(--error)'
      : tone === 'warn'
        ? 'var(--warn)'
        : 'var(--text-muted)';
  return (
    <Box
      sx={{
        display: 'flex',
        alignItems: 'center',
        gap: 1.25,
        px: { xs: 1.5, md: 2.5 },
        py: 1,
        borderBottom: '1px solid var(--border)',
        background: 'rgba(255,255,255,0.015)',
      }}
    >
      <Box sx={{ display: 'inline-flex', gap: 0.5 }}>
        {[0, 1, 2].map((i) => (
          <Box
            key={i}
            sx={{
              width: 5,
              height: 5,
              borderRadius: '50%',
              bgcolor: color,
              animation: 'rli-pulse 1.1s ease-in-out infinite',
              animationDelay: `${i * 0.16}s`,
              '@keyframes rli-pulse': {
                '0%, 80%, 100%': { opacity: 0.25 },
                '40%': { opacity: 1 },
              },
            }}
          />
        ))}
      </Box>
      <Typography
        sx={{
          fontFamily: 'var(--font-mono)',
          fontSize: '0.72rem',
          color: 'var(--text-muted)',
          letterSpacing: '0.04em',
        }}
      >
        {label.toLowerCase()} · check the bridge log below for details
      </Typography>
    </Box>
  );
}
