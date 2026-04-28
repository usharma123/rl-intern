import { memo } from 'react';
import { Box } from '@mui/material';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { oneDark } from 'react-syntax-highlighter/dist/esm/styles/prism';

interface Props {
  content: string;
}

export default memo(function MarkdownContent({ content }: Props) {
  return (
    <Box
      sx={{
        color: 'var(--text)',
        fontSize: '0.93rem',
        lineHeight: 1.65,
        letterSpacing: '-0.003em',
        '& p': { my: 1 },
        '& p:first-of-type': { mt: 0 },
        '& p:last-of-type': { mb: 0 },
        '& a': {
          color: 'var(--accent)',
          textDecoration: 'underline',
          textUnderlineOffset: 2,
          textDecorationColor: 'rgba(127,238,100,0.4)',
          '&:hover': { textDecorationColor: 'var(--accent)' },
        },
        '& ul, & ol': { pl: 3, my: 1 },
        '& li': { my: 0.25 },
        '& code': {
          fontFamily: 'var(--font-mono)',
          fontSize: '0.85em',
          background: 'var(--code-bg)',
          padding: '1px 6px',
          borderRadius: '3px',
          color: 'var(--accent-strong)',
        },
        '& pre': {
          background: 'var(--code-panel-bg)',
          border: '1px solid var(--border)',
          borderRadius: 'var(--radius-md)',
          padding: '12px 14px',
          overflowX: 'auto',
          my: 1.5,
        },
        '& pre code': {
          background: 'transparent',
          padding: 0,
          fontSize: '0.82rem',
          color: 'var(--text)',
        },
        '& blockquote': {
          borderLeft: '2px solid var(--accent)',
          pl: 2,
          color: 'var(--text-muted)',
          my: 1.5,
          ml: 0,
        },
        '& table': {
          borderCollapse: 'collapse',
          width: '100%',
          my: 1.5,
          fontSize: '0.85rem',
          fontFamily: 'var(--font-mono)',
        },
        '& th, & td': {
          border: '1px solid var(--border)',
          padding: '6px 10px',
          textAlign: 'left',
        },
        '& th': {
          bgcolor: 'var(--surface)',
          fontWeight: 600,
          color: 'var(--text-muted)',
          textTransform: 'uppercase',
          fontSize: '0.7rem',
          letterSpacing: '0.08em',
        },
        '& hr': { border: 0, borderTop: '1px solid var(--border)', my: 2 },
        '& h1, & h2, & h3, & h4': {
          fontWeight: 700,
          mt: 1.5,
          mb: 0.75,
          color: 'var(--text)',
          letterSpacing: '-0.015em',
        },
        '& h1': { fontSize: '1.2rem' },
        '& h2': { fontSize: '1.05rem' },
        '& h3': { fontSize: '0.98rem' },
        '& strong': { color: 'var(--text)', fontWeight: 600 },
      }}
    >
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={{
          code({ className, children, ...rest }) {
            const inline = !className;
            const match = /language-(\w+)/.exec(className || '');
            if (inline || !match) {
              return (
                <code className={className} {...rest}>
                  {children}
                </code>
              );
            }
            return (
              <SyntaxHighlighter
                language={match[1]}
                style={oneDark as { [key: string]: React.CSSProperties }}
                customStyle={{
                  margin: 0,
                  background: 'transparent',
                  fontSize: '0.82rem',
                  padding: 0,
                }}
                PreTag="div"
              >
                {String(children).replace(/\n$/, '')}
              </SyntaxHighlighter>
            );
          },
        }}
      >
        {content}
      </ReactMarkdown>
    </Box>
  );
});
