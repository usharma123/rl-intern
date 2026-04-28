import { KeyboardEvent, useState } from 'react';
import { Box, IconButton, InputBase, Tooltip, Typography } from '@mui/material';
import ArrowUpwardIcon from '@mui/icons-material/ArrowUpward';
import StopRoundedIcon from '@mui/icons-material/StopRounded';

interface Props {
  disabled?: boolean;
  isStreaming?: boolean;
  onSend: (text: string) => void;
  onInterrupt?: () => void;
  placeholder?: string;
}

export default function ChatInput({
  disabled,
  isStreaming,
  onSend,
  onInterrupt,
  placeholder,
}: Props) {
  const [value, setValue] = useState('');
  const [focused, setFocused] = useState(false);
  const isEmpty = value.length === 0;

  const handleSubmit = () => {
    const text = value.trim();
    if (!text) return;
    onSend(text);
    setValue('');
  };

  const handleKey = (e: KeyboardEvent<HTMLInputElement | HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  return (
    <Box
      sx={{
        width: '100%',
        maxWidth: 880,
        mx: 'auto',
        px: { xs: 1.5, md: 2 },
        pt: 1,
        pb: 2,
      }}
    >
      <Box
        sx={{
          display: 'flex',
          alignItems: 'flex-end',
          gap: 1,
          p: 1.25,
          background: 'var(--surface)',
          border: '1px solid',
          borderColor: focused ? 'var(--accent)' : 'var(--border)',
          borderRadius: 'var(--radius-md)',
          transition: 'border-color 100ms ease, box-shadow 100ms ease',
          boxShadow: focused ? '0 0 0 1px var(--accent), 0 0 24px rgba(127,238,100,0.10)' : 'none',
        }}
      >
        {/* Prompt prefix */}
        <Typography
          component="span"
          sx={{
            fontFamily: 'var(--font-mono)',
            fontWeight: 600,
            fontSize: '0.95rem',
            color: focused || !isEmpty ? 'var(--accent)' : 'var(--text-muted)',
            transition: 'color 100ms ease',
            mt: '6px',
            ml: 0.25,
            userSelect: 'none',
          }}
        >
          &gt;
        </Typography>

        <Box sx={{ flex: 1, position: 'relative', minWidth: 0 }}>
          <InputBase
            multiline
            maxRows={10}
            fullWidth
            value={value}
            onChange={(e) => setValue(e.target.value)}
            onKeyDown={handleKey}
            onFocus={() => setFocused(true)}
            onBlur={() => setFocused(false)}
            disabled={disabled}
            placeholder={
              isEmpty
                ? '' // we render a custom placeholder w/ caret below
                : ''
            }
            inputProps={{
              spellCheck: true,
              autoCorrect: 'off',
              autoCapitalize: 'off',
            }}
            sx={{
              fontSize: '0.92rem',
              fontFamily: 'var(--font-sans)',
              color: 'var(--text)',
              lineHeight: 1.55,
              p: 0,
              mt: '4px',
              '& textarea': { resize: 'none' },
            }}
          />
          {/* Custom placeholder with blinking caret */}
          {isEmpty && (
            <Box
              sx={{
                position: 'absolute',
                top: '4px',
                left: 0,
                pointerEvents: 'none',
                fontFamily: 'var(--font-sans)',
                fontSize: '0.92rem',
                lineHeight: 1.55,
                color: 'var(--text-faint)',
                whiteSpace: 'nowrap',
                overflow: 'hidden',
                textOverflow: 'ellipsis',
                maxWidth: '100%',
              }}
            >
              {placeholder ??
                (disabled
                  ? 'connecting to agent…'
                  : 'ask anything · e.g. train PPO on CartPole-v1 for 1000 timesteps')}
              {!disabled && <span className="rli-caret">▍</span>}
            </Box>
          )}
        </Box>

        {isStreaming ? (
          <Tooltip title="Interrupt">
            <span>
              <IconButton
                onClick={onInterrupt}
                sx={{
                  width: 30,
                  height: 30,
                  borderRadius: 'var(--radius-sm)',
                  border: '1px solid var(--error-border)',
                  background: 'var(--error-soft)',
                  color: 'var(--error)',
                  '&:hover': { background: 'rgba(255,107,107,0.18)', color: 'var(--error)' },
                }}
              >
                <StopRoundedIcon sx={{ fontSize: 16 }} />
              </IconButton>
            </span>
          </Tooltip>
        ) : (
          <Tooltip title="Send · ↵">
            <span>
              <IconButton
                onClick={handleSubmit}
                disabled={disabled || !value.trim()}
                sx={{
                  width: 30,
                  height: 30,
                  borderRadius: 'var(--radius-sm)',
                  background: 'var(--accent)',
                  color: '#0a0b0c',
                  border: '1px solid transparent',
                  '&:hover': { background: 'var(--accent-strong)', color: '#0a0b0c' },
                  '&.Mui-disabled': {
                    background: 'transparent',
                    border: '1px solid var(--border)',
                    color: 'var(--text-faint)',
                  },
                }}
              >
                <ArrowUpwardIcon sx={{ fontSize: 16 }} />
              </IconButton>
            </span>
          </Tooltip>
        )}
      </Box>

      {/* Hint row */}
      <Box
        sx={{
          mt: 0.75,
          display: 'flex',
          gap: 1.5,
          fontFamily: 'var(--font-mono)',
          fontSize: '0.66rem',
          color: 'var(--text-faint)',
          letterSpacing: '0.06em',
          px: 0.5,
        }}
      >
        <Box>↵ send</Box>
        <Box>⇧↵ newline</Box>
        {isStreaming && <Box sx={{ color: 'var(--warn)' }}>esc · interrupt running</Box>}
      </Box>
    </Box>
  );
}
