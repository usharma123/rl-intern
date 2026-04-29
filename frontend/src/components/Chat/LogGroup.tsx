import { useMemo, useState } from 'react';
import { Box, IconButton, Tooltip, Typography, Collapse } from '@mui/material';
import ContentCopyRoundedIcon from '@mui/icons-material/ContentCopyRounded';
import KeyboardArrowDownRoundedIcon from '@mui/icons-material/KeyboardArrowDownRounded';
import ReportGmailerrorredRoundedIcon from '@mui/icons-material/ReportGmailerrorredRounded';
import TerminalRoundedIcon from '@mui/icons-material/TerminalRounded';
import type { ChatMessage, SystemLevel } from '@/types/events';

// ─── parser ────────────────────────────────────────────────────────────────

interface Frame {
  file: string;
  line: number;
  func: string;
  code?: string;
}

interface TracebackInfo {
  kind: 'traceback';
  loggerLevel?: string;
  loggerName?: string;
  loggerMessage?: string;
  preamble: string[]; // any lines before the traceback that are not the logger header
  frames: Frame[];
  exception: { type: string; message: string };
  raw: string;
}

interface ConsoleInfo {
  kind: 'console';
  lines: string[];
  raw: string;
}

type Parsed = TracebackInfo | ConsoleInfo;

const FRAME_RE = /^\s*File "([^"]+)", line (\d+), in (.+)$/;
const LOGGER_RE = /^(ERROR|WARNING|INFO|DEBUG|CRITICAL|WARN):([^:]+):(.*)$/;
const EXC_RE = /^([\w.]+(?:Error|Exception|Warning|Failure|Interrupt|Signal|Exit))(?::\s*(.*))?$/;

function parseTraceback(lines: string[]): Parsed {
  const tbStart = lines.findIndex((l) => /Traceback \(most recent call last\):/.test(l));
  if (tbStart === -1) {
    return { kind: 'console', lines, raw: lines.join('\n') };
  }

  // Walk backwards from end to find the exception line. It is usually the last
  // non-empty line. If no match, fall back to console.
  let excIdx = lines.length - 1;
  while (excIdx > tbStart && !lines[excIdx].trim()) excIdx -= 1;
  const excLine = lines[excIdx] ?? '';
  const excMatch = excLine.match(EXC_RE);
  if (!excMatch) {
    return { kind: 'console', lines, raw: lines.join('\n') };
  }

  // Anything before the traceback that looks like a logger preface becomes
  // the title; anything else above is preamble.
  let loggerLevel: string | undefined;
  let loggerName: string | undefined;
  let loggerMessage: string | undefined;
  const preamble: string[] = [];
  for (let i = 0; i < tbStart; i += 1) {
    const m = lines[i].match(LOGGER_RE);
    if (m && !loggerMessage) {
      loggerLevel = m[1];
      loggerName = m[2];
      loggerMessage = m[3].trim();
    } else if (lines[i].trim()) {
      preamble.push(lines[i]);
    }
  }

  const frames: Frame[] = [];
  let i = tbStart + 1;
  while (i < excIdx) {
    const m = lines[i].match(FRAME_RE);
    if (!m) {
      i += 1;
      continue;
    }
    const frame: Frame = { file: m[1], line: parseInt(m[2], 10), func: m[3] };
    let j = i + 1;
    const codeChunks: string[] = [];
    while (j < excIdx && !FRAME_RE.test(lines[j])) {
      codeChunks.push(lines[j].trim());
      j += 1;
    }
    const real = codeChunks.find((c) => c && !/^[\^~ ]+$/.test(c) && !/^\.\.\.<\d+ lines>\.\.\.$/.test(c));
    if (real) frame.code = real;
    frames.push(frame);
    i = j;
  }

  return {
    kind: 'traceback',
    loggerLevel,
    loggerName,
    loggerMessage,
    preamble,
    frames,
    exception: {
      type: excMatch[1],
      message: (excMatch[2] ?? '').trim(),
    },
    raw: lines.join('\n'),
  };
}

// ─── styling helpers ───────────────────────────────────────────────────────

const sysFor = (level: SystemLevel) => {
  if (level === 'error') return { fg: 'var(--error)', soft: 'var(--error-soft)', border: 'var(--error-border)' };
  if (level === 'warn') return { fg: 'var(--warn)', soft: 'var(--warn-soft)', border: 'var(--warn-border)' };
  return { fg: 'var(--text-muted)', soft: 'transparent', border: 'var(--border-strong)' };
};

const basenameOf = (path: string) => {
  const parts = path.split('/');
  return parts[parts.length - 1] || path;
};

const dirnameOf = (path: string) => {
  const parts = path.split('/');
  parts.pop();
  // collapse leading repo-root noise so we focus on the meaningful tail
  const idx = parts.findIndex((p) => p === 'rl-intern');
  const tail = idx >= 0 ? parts.slice(idx + 1) : parts.slice(-3);
  return tail.join('/');
};

// ─── traceback card ───────────────────────────────────────────────────────

function TracebackCard({ tb, level }: { tb: TracebackInfo; level: SystemLevel }) {
  const [expanded, setExpanded] = useState(false);
  const [copied, setCopied] = useState(false);
  const palette = sysFor(level === 'info' ? 'error' : level);
  const deepest = tb.frames[tb.frames.length - 1];

  const copy = async () => {
    try {
      await navigator.clipboard.writeText(tb.raw);
      setCopied(true);
      setTimeout(() => setCopied(false), 1400);
    } catch {
      /* noop */
    }
  };

  return (
    <Box
      sx={{
        my: 1.5,
        position: 'relative',
        display: 'flex',
        alignItems: 'stretch',
        borderRadius: 'var(--radius-md)',
        background:
          'linear-gradient(180deg, rgba(255,107,107,0.06) 0%, rgba(255,107,107,0.02) 100%), var(--surface)',
        border: `1px solid ${palette.border}`,
        boxShadow:
          '0 0 0 1px rgba(255,107,107,0.04), 0 8px 24px -16px rgba(255,107,107,0.45), inset 0 0 0 1px rgba(255,255,255,0.01)',
        overflow: 'hidden',
        // dotted texture in the corner
        '&::after': {
          content: '""',
          position: 'absolute',
          top: 0,
          right: 0,
          width: 140,
          height: 80,
          background:
            'radial-gradient(circle at 1px 1px, rgba(255,107,107,0.18) 1px, transparent 0) 0 0 / 8px 8px',
          opacity: 0.35,
          pointerEvents: 'none',
          maskImage: 'linear-gradient(225deg, black, transparent 70%)',
          WebkitMaskImage: 'linear-gradient(225deg, black, transparent 70%)',
        },
      }}
    >
      {/* error rail */}
      <Box
        sx={{
          width: 4,
          background: `linear-gradient(180deg, ${palette.fg} 0%, transparent 100%)`,
          flexShrink: 0,
        }}
      />

      <Box sx={{ flex: 1, minWidth: 0, p: 1.75 }}>
        {/* header strip */}
        <Box
          sx={{
            display: 'flex',
            alignItems: 'center',
            gap: 1.25,
            mb: 1,
          }}
        >
          <Box
            sx={{
              display: 'flex',
              alignItems: 'center',
              gap: 0.75,
              px: 0.85,
              py: 0.4,
              borderRadius: 'var(--radius-sm)',
              background: 'rgba(255,107,107,0.12)',
              border: `1px solid ${palette.border}`,
            }}
          >
            <ReportGmailerrorredRoundedIcon sx={{ fontSize: 13, color: palette.fg }} />
            <Typography
              sx={{
                fontFamily: 'var(--font-mono)',
                fontSize: '0.6rem',
                fontWeight: 700,
                letterSpacing: '0.18em',
                color: palette.fg,
              }}
            >
              EXCEPTION
            </Typography>
          </Box>
          {tb.loggerName && (
            <Typography
              sx={{
                fontFamily: 'var(--font-mono)',
                fontSize: '0.65rem',
                color: 'var(--text-faint)',
                letterSpacing: '0.04em',
                overflow: 'hidden',
                textOverflow: 'ellipsis',
                whiteSpace: 'nowrap',
                minWidth: 0,
                flex: 1,
              }}
            >
              {tb.loggerName}
            </Typography>
          )}
          {!tb.loggerName && <Box sx={{ flex: 1 }} />}
          <Tooltip title={copied ? 'copied' : 'copy traceback'} placement="top">
            <IconButton
              size="small"
              onClick={copy}
              sx={{
                width: 24,
                height: 24,
                color: copied ? 'var(--accent)' : 'var(--text-muted)',
              }}
            >
              <ContentCopyRoundedIcon sx={{ fontSize: 13 }} />
            </IconButton>
          </Tooltip>
        </Box>

        {/* exception headline */}
        <Box sx={{ display: 'flex', alignItems: 'baseline', gap: 1.25, flexWrap: 'wrap' }}>
          <Typography
            sx={{
              fontFamily: 'var(--font-mono)',
              fontSize: '1.05rem',
              fontWeight: 700,
              color: palette.fg,
              letterSpacing: '-0.01em',
            }}
          >
            {tb.exception.type}
          </Typography>
          {tb.exception.message && (
            <Typography
              sx={{
                fontFamily: 'var(--font-mono)',
                fontSize: '0.86rem',
                color: 'var(--text)',
                lineHeight: 1.5,
                wordBreak: 'break-word',
                minWidth: 0,
              }}
            >
              {tb.exception.message}
            </Typography>
          )}
        </Box>

        {/* logger pre-message (e.g. "Tool execution failed") */}
        {tb.loggerMessage && (
          <Typography
            sx={{
              mt: 0.5,
              fontFamily: 'var(--font-sans)',
              fontSize: '0.78rem',
              color: 'var(--text-muted)',
              fontStyle: 'italic',
              letterSpacing: '-0.003em',
            }}
          >
            {tb.loggerMessage}
          </Typography>
        )}

        {/* origin pill + frame count */}
        {deepest && (
          <Box
            sx={{
              mt: 1.5,
              display: 'flex',
              alignItems: 'center',
              gap: 1,
              flexWrap: 'wrap',
            }}
          >
            <Box
              sx={{
                display: 'flex',
                alignItems: 'center',
                gap: 0.85,
                px: 1,
                py: 0.6,
                borderRadius: 'var(--radius-sm)',
                background: 'var(--code-panel-bg)',
                border: '1px solid var(--border-strong)',
                fontFamily: 'var(--font-mono)',
                fontSize: '0.72rem',
                minWidth: 0,
                flex: '1 1 auto',
                maxWidth: '100%',
              }}
            >
              <Box
                component="span"
                sx={{
                  color: 'var(--text-faint)',
                  fontSize: '0.6rem',
                  letterSpacing: '0.16em',
                  fontWeight: 600,
                }}
              >
                ORIGIN
              </Box>
              <Box
                component="span"
                sx={{
                  color: 'var(--text-faint)',
                  whiteSpace: 'nowrap',
                  overflow: 'hidden',
                  textOverflow: 'ellipsis',
                  minWidth: 0,
                }}
              >
                {dirnameOf(deepest.file)}/
              </Box>
              <Box component="span" sx={{ color: 'var(--accent)', fontWeight: 600 }}>
                {basenameOf(deepest.file)}
              </Box>
              <Box component="span" sx={{ color: 'var(--text-faint)' }}>:</Box>
              <Box component="span" sx={{ color: 'var(--text)', fontWeight: 600 }}>
                {deepest.line}
              </Box>
              <Box component="span" sx={{ color: 'var(--text-faint)' }}>·</Box>
              <Box component="span" sx={{ color: 'var(--text-muted)', fontStyle: 'italic' }}>
                {deepest.func}()
              </Box>
            </Box>

            <Box
              role="button"
              onClick={() => setExpanded((e) => !e)}
              sx={{
                cursor: 'pointer',
                userSelect: 'none',
                display: 'flex',
                alignItems: 'center',
                gap: 0.5,
                px: 1,
                py: 0.6,
                borderRadius: 'var(--radius-sm)',
                background: expanded ? 'var(--accent-soft)' : 'transparent',
                border: `1px solid ${expanded ? 'var(--accent-border)' : 'var(--border-strong)'}`,
                color: expanded ? 'var(--accent)' : 'var(--text-muted)',
                fontFamily: 'var(--font-mono)',
                fontSize: '0.68rem',
                letterSpacing: '0.06em',
                fontWeight: 600,
                transition: 'all 0.12s ease',
                '&:hover': {
                  borderColor: 'var(--accent-border)',
                  color: 'var(--accent)',
                },
              }}
            >
              {tb.frames.length} {tb.frames.length === 1 ? 'frame' : 'frames'}
              <KeyboardArrowDownRoundedIcon
                sx={{
                  fontSize: 14,
                  transition: 'transform 0.18s ease',
                  transform: expanded ? 'rotate(180deg)' : 'rotate(0deg)',
                }}
              />
            </Box>
          </Box>
        )}

        {/* expanded stack */}
        <Collapse in={expanded} timeout={200} unmountOnExit>
          <Box
            sx={{
              mt: 1.5,
              borderRadius: 'var(--radius-sm)',
              background: 'var(--code-panel-bg)',
              border: '1px solid var(--border)',
              overflow: 'hidden',
            }}
          >
            <Box
              sx={{
                px: 1.25,
                py: 0.6,
                borderBottom: '1px solid var(--border)',
                display: 'flex',
                alignItems: 'center',
                gap: 0.75,
                background: 'rgba(255,255,255,0.015)',
              }}
            >
              <Typography
                sx={{
                  fontFamily: 'var(--font-mono)',
                  fontSize: '0.6rem',
                  letterSpacing: '0.16em',
                  fontWeight: 600,
                  color: 'var(--text-faint)',
                }}
              >
                CALL STACK · OUTERMOST FIRST
              </Typography>
            </Box>
            <Box>
              {tb.frames.map((f, idx) => {
                const isDeepest = idx === tb.frames.length - 1;
                return (
                  <Box
                    key={`${f.file}:${f.line}:${idx}`}
                    sx={{
                      display: 'grid',
                      gridTemplateColumns: '32px 1fr',
                      px: 1.25,
                      py: 1,
                      borderTop: idx === 0 ? 'none' : '1px solid var(--border)',
                      background: isDeepest ? 'rgba(255,107,107,0.04)' : 'transparent',
                      position: 'relative',
                    }}
                  >
                    {isDeepest && (
                      <Box
                        sx={{
                          position: 'absolute',
                          left: 0,
                          top: 0,
                          bottom: 0,
                          width: 2,
                          background: palette.fg,
                        }}
                      />
                    )}
                    <Typography
                      sx={{
                        fontFamily: 'var(--font-mono)',
                        fontSize: '0.65rem',
                        color: isDeepest ? palette.fg : 'var(--text-faint)',
                        fontWeight: 600,
                        letterSpacing: '0.05em',
                      }}
                    >
                      {String(idx + 1).padStart(2, '0')}
                    </Typography>
                    <Box sx={{ minWidth: 0 }}>
                      <Box
                        sx={{
                          display: 'flex',
                          alignItems: 'baseline',
                          gap: 0.75,
                          flexWrap: 'wrap',
                          fontFamily: 'var(--font-mono)',
                          fontSize: '0.74rem',
                        }}
                      >
                        <Box
                          component="span"
                          sx={{
                            color: 'var(--text-faint)',
                            wordBreak: 'break-all',
                          }}
                        >
                          {dirnameOf(f.file)}/
                        </Box>
                        <Box
                          component="span"
                          sx={{ color: isDeepest ? palette.fg : 'var(--accent)', fontWeight: 600 }}
                        >
                          {basenameOf(f.file)}
                        </Box>
                        <Box component="span" sx={{ color: 'var(--text-faint)' }}>:</Box>
                        <Box component="span" sx={{ color: 'var(--text)', fontWeight: 600 }}>
                          {f.line}
                        </Box>
                        <Box component="span" sx={{ color: 'var(--text-muted)', fontStyle: 'italic' }}>
                          in {f.func}
                        </Box>
                      </Box>
                      {f.code && (
                        <Typography
                          sx={{
                            mt: 0.45,
                            fontFamily: 'var(--font-mono)',
                            fontSize: '0.74rem',
                            color: isDeepest ? 'var(--text)' : 'var(--text-muted)',
                            lineHeight: 1.45,
                            wordBreak: 'break-word',
                            paddingLeft: '0.85em',
                            borderLeft: `1px solid ${isDeepest ? palette.border : 'var(--border)'}`,
                          }}
                        >
                          {f.code}
                        </Typography>
                      )}
                    </Box>
                  </Box>
                );
              })}
            </Box>
          </Box>
        </Collapse>
      </Box>
    </Box>
  );
}

// ─── multi-line console block ─────────────────────────────────────────────

function ConsoleBlock({
  lines,
  level,
  source,
  raw,
}: {
  lines: string[];
  level: SystemLevel;
  source: string;
  raw: string;
}) {
  const [expanded, setExpanded] = useState(false);
  const [copied, setCopied] = useState(false);
  const palette = sysFor(level);
  const COLLAPSE_THRESHOLD = 6;
  const overflowing = lines.length > COLLAPSE_THRESHOLD;
  const visible = expanded || !overflowing ? lines : lines.slice(0, COLLAPSE_THRESHOLD);
  const hidden = lines.length - visible.length;

  const copy = async () => {
    try {
      await navigator.clipboard.writeText(raw);
      setCopied(true);
      setTimeout(() => setCopied(false), 1400);
    } catch {
      /* noop */
    }
  };

  return (
    <Box
      sx={{
        my: 1.25,
        borderRadius: 'var(--radius-md)',
        border: `1px solid ${palette.border}`,
        background: palette.soft === 'transparent' ? 'var(--surface)' : palette.soft,
        overflow: 'hidden',
      }}
    >
      <Box
        sx={{
          display: 'flex',
          alignItems: 'center',
          gap: 1,
          px: 1.25,
          py: 0.7,
          borderBottom: `1px solid ${palette.border}`,
          background: 'rgba(0,0,0,0.18)',
        }}
      >
        <TerminalRoundedIcon sx={{ fontSize: 13, color: palette.fg }} />
        <Typography
          sx={{
            fontFamily: 'var(--font-mono)',
            fontSize: '0.6rem',
            letterSpacing: '0.18em',
            fontWeight: 700,
            color: palette.fg,
          }}
        >
          {source.toUpperCase()} · {level.toUpperCase()}
        </Typography>
        <Box sx={{ width: 1, height: 12, background: 'var(--border-strong)' }} />
        <Typography
          sx={{
            fontFamily: 'var(--font-mono)',
            fontSize: '0.62rem',
            color: 'var(--text-faint)',
            letterSpacing: '0.06em',
          }}
        >
          {lines.length} {lines.length === 1 ? 'line' : 'lines'}
        </Typography>
        <Box sx={{ flex: 1 }} />
        <Tooltip title={copied ? 'copied' : 'copy'} placement="top">
          <IconButton
            size="small"
            onClick={copy}
            sx={{ width: 22, height: 22, color: copied ? 'var(--accent)' : 'var(--text-muted)' }}
          >
            <ContentCopyRoundedIcon sx={{ fontSize: 12 }} />
          </IconButton>
        </Tooltip>
      </Box>

      <Box
        sx={{
          fontFamily: 'var(--font-mono)',
          fontSize: '0.76rem',
          py: 0.75,
          maxHeight: expanded ? 'none' : 360,
          overflowY: 'auto',
        }}
      >
        {visible.map((ln, idx) => (
          <Box
            key={idx}
            sx={{
              display: 'grid',
              gridTemplateColumns: '40px 1fr',
              alignItems: 'baseline',
              px: 1.25,
              py: 0.15,
              '&:hover': { background: 'rgba(255,255,255,0.025)' },
            }}
          >
            <Typography
              sx={{
                fontFamily: 'var(--font-mono)',
                fontSize: '0.66rem',
                color: 'var(--text-faint)',
                textAlign: 'right',
                pr: 1,
                userSelect: 'none',
              }}
            >
              {String(idx + 1).padStart(2, '0')}
            </Typography>
            <Typography
              sx={{
                fontFamily: 'var(--font-mono)',
                fontSize: '0.76rem',
                color: 'var(--text)',
                lineHeight: 1.55,
                whiteSpace: 'pre-wrap',
                wordBreak: 'break-word',
              }}
            >
              {ln || ' '}
            </Typography>
          </Box>
        ))}
      </Box>

      {overflowing && (
        <Box
          role="button"
          onClick={() => setExpanded((e) => !e)}
          sx={{
            cursor: 'pointer',
            userSelect: 'none',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            gap: 0.5,
            py: 0.65,
            borderTop: '1px solid var(--border)',
            color: 'var(--text-muted)',
            fontFamily: 'var(--font-mono)',
            fontSize: '0.66rem',
            letterSpacing: '0.12em',
            fontWeight: 600,
            background: 'rgba(0,0,0,0.18)',
            transition: 'color 0.12s ease, background 0.12s ease',
            '&:hover': { color: 'var(--accent)', background: 'rgba(127,238,100,0.04)' },
          }}
        >
          {expanded ? 'COLLAPSE' : `SHOW ${hidden} MORE`}
          <KeyboardArrowDownRoundedIcon
            sx={{
              fontSize: 14,
              transition: 'transform 0.18s ease',
              transform: expanded ? 'rotate(180deg)' : 'rotate(0deg)',
            }}
          />
        </Box>
      )}
    </Box>
  );
}

// ─── public component ─────────────────────────────────────────────────────

type SystemMessage = Extract<ChatMessage, { kind: 'system' }>;

interface Props {
  messages: SystemMessage[];
}

export default function LogGroup({ messages }: Props) {
  const lines = useMemo(() => messages.map((m) => m.text), [messages]);
  const parsed = useMemo(() => parseTraceback(lines), [lines]);
  const peakLevel = useMemo<SystemLevel>(() => {
    if (messages.some((m) => m.level === 'error')) return 'error';
    if (messages.some((m) => m.level === 'warn')) return 'warn';
    return 'info';
  }, [messages]);
  const source = messages[0]?.source ?? 'bridge';

  if (parsed.kind === 'traceback') {
    return (
      <Box>
        {parsed.preamble.length > 0 && (
          <ConsoleBlock
            lines={parsed.preamble}
            level={peakLevel}
            source={source}
            raw={parsed.preamble.join('\n')}
          />
        )}
        <TracebackCard tb={parsed} level={peakLevel} />
      </Box>
    );
  }

  return <ConsoleBlock lines={parsed.lines} level={peakLevel} source={source} raw={parsed.raw} />;
}
