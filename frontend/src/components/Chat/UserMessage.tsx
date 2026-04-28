import { Box, Typography } from '@mui/material';

interface Props {
  text: string;
}

export default function UserMessage({ text }: Props) {
  return (
    <Box sx={{ display: 'flex', justifyContent: 'flex-end', mb: 3 }}>
      <Box
        sx={{
          maxWidth: { xs: '94%', md: '76%' },
          background: 'var(--accent-soft)',
          border: '1px solid var(--accent-border)',
          borderRadius: 'var(--radius-md)',
          px: 1.75,
          py: 1.25,
          position: 'relative',
        }}
      >
        <Typography
          sx={{
            position: 'absolute',
            top: -9,
            right: 12,
            px: 0.75,
            background: 'var(--bg)',
            fontFamily: 'var(--font-mono)',
            fontSize: '0.6rem',
            fontWeight: 600,
            letterSpacing: '0.16em',
            color: 'var(--accent)',
          }}
        >
          YOU
        </Typography>
        <Typography
          sx={{
            fontSize: '0.92rem',
            lineHeight: 1.55,
            color: 'var(--text)',
            whiteSpace: 'pre-wrap',
            wordBreak: 'break-word',
          }}
        >
          {text}
        </Typography>
      </Box>
    </Box>
  );
}
