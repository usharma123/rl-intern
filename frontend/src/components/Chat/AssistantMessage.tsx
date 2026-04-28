import { Box, Typography } from '@mui/material';
import MarkdownContent from './MarkdownContent';
import ToolCallGroup from './ToolCallGroup';
import type { ToolBlock } from '@/types/events';

interface Props {
  text: string;
  tools: ToolBlock[];
}

export default function AssistantMessage({ text, tools }: Props) {
  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', mb: 3 }}>
      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 0.875 }}>
        <Box
          sx={{
            width: 6,
            height: 6,
            borderRadius: '50%',
            background: 'var(--accent)',
            boxShadow: '0 0 8px var(--accent)',
          }}
        />
        <Typography
          sx={{
            fontFamily: 'var(--font-mono)',
            fontSize: '0.66rem',
            fontWeight: 600,
            letterSpacing: '0.16em',
            color: 'var(--accent)',
          }}
        >
          AGENT
        </Typography>
      </Box>

      <Box sx={{ pl: 1.75, borderLeft: '1px solid var(--border)' }}>
        {text && <MarkdownContent content={text} />}
        {tools.length > 0 && (
          <Box sx={{ mt: text ? 1.5 : 0 }}>
            {tools.map((tool) => (
              <ToolCallGroup key={tool.id} tool={tool} />
            ))}
          </Box>
        )}
      </Box>
    </Box>
  );
}
