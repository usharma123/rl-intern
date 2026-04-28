import { createTheme, type ThemeOptions } from '@mui/material/styles';

// ────────────────────────────────────────────────────────────────────────────
// Modal-flavored design tokens. Near-black canvas, lime-green accent, sharp
// 8 px corners, JetBrains Mono surfacing every technical detail.
// ────────────────────────────────────────────────────────────────────────────

const tokens = {
  '--bg': '#0a0b0c',
  '--bg-alt': '#08090a',
  '--panel': '#0c0d0f',
  '--surface': '#101214',
  '--surface-hover': '#15181a',

  '--text': '#e6e8eb',
  '--text-muted': '#8a8f96',
  '--text-faint': '#52575d',

  '--accent': '#7FEE64',
  '--accent-strong': '#a9f696',
  '--accent-soft': 'rgba(127, 238, 100, 0.08)',
  '--accent-soft-2': 'rgba(127, 238, 100, 0.14)',
  '--accent-border': 'rgba(127, 238, 100, 0.45)',

  '--warn': '#ffaa3a',
  '--warn-soft': 'rgba(255, 170, 58, 0.10)',
  '--warn-border': 'rgba(255, 170, 58, 0.40)',

  '--error': '#ff6b6b',
  '--error-soft': 'rgba(255, 107, 107, 0.10)',
  '--error-border': 'rgba(255, 107, 107, 0.40)',

  '--border': 'rgba(255, 255, 255, 0.06)',
  '--border-strong': 'rgba(255, 255, 255, 0.12)',
  '--border-hover': 'rgba(127, 238, 100, 0.45)',

  '--code-bg': 'rgba(0, 0, 0, 0.4)',
  '--code-panel-bg': '#06070a',

  '--scrollbar-thumb': '#1c2024',

  '--shadow-1': '0 0 0 1px rgba(255, 255, 255, 0.04), 0 4px 14px rgba(0, 0, 0, 0.5)',
  '--shadow-glow': '0 0 0 1px rgba(127, 238, 100, 0.45), 0 0 24px rgba(127, 238, 100, 0.10)',

  '--radius-sm': '4px',
  '--radius-md': '8px',
  '--radius-lg': '10px',

  '--font-sans':
    '"Inter", system-ui, -apple-system, "Segoe UI", Roboto, Arial, sans-serif',
  '--font-mono':
    '"JetBrains Mono", ui-monospace, SFMono-Regular, Menlo, Monaco, "Roboto Mono", monospace',
} as const;

const typography: ThemeOptions['typography'] = {
  fontFamily: tokens['--font-sans'],
  fontSize: 14,
  htmlFontSize: 16,
  body1: { letterSpacing: '-0.005em' },
  body2: { letterSpacing: '-0.003em' },
  h1: { fontWeight: 700, letterSpacing: '-0.025em' },
  h2: { fontWeight: 700, letterSpacing: '-0.022em' },
  h3: { fontWeight: 700, letterSpacing: '-0.018em' },
  h4: { fontWeight: 700, letterSpacing: '-0.015em' },
  h5: { fontWeight: 700, letterSpacing: '-0.012em' },
  h6: { fontWeight: 700, letterSpacing: '-0.01em' },
  button: {
    fontFamily: tokens['--font-sans'],
    textTransform: 'none' as const,
    fontWeight: 600,
    letterSpacing: '-0.005em',
  },
  caption: { letterSpacing: '0.01em' },
  overline: {
    fontFamily: tokens['--font-mono'],
    letterSpacing: '0.12em',
    fontWeight: 600,
  },
};

const cssBaseline: ThemeOptions['components'] = {
  MuiCssBaseline: {
    styleOverrides: {
      ':root': tokens,
      '*, *::before, *::after': { boxSizing: 'border-box' },
      html: { colorScheme: 'dark' },
      body: {
        margin: 0,
        background:
          // base + faint top-center lime glow + ultra-subtle dot grid
          'radial-gradient(ellipse 80% 38% at 50% -10%, rgba(127, 238, 100, 0.045), transparent 70%),' +
          'radial-gradient(circle, rgba(255,255,255,0.025) 1px, transparent 1px) 0 0 / 22px 22px,' +
          'linear-gradient(180deg, #0b0c0e 0%, #08090a 100%)',
        color: 'var(--text)',
        fontFamily: 'var(--font-sans)',
        WebkitFontSmoothing: 'antialiased',
        MozOsxFontSmoothing: 'grayscale',
        scrollbarWidth: 'thin' as const,
        scrollbarColor: 'var(--scrollbar-thumb) transparent',
        '&::-webkit-scrollbar': { width: '8px', height: '8px' },
        '&::-webkit-scrollbar-thumb': {
          backgroundColor: 'var(--scrollbar-thumb)',
          borderRadius: '0',
        },
        '&::-webkit-scrollbar-track': { backgroundColor: 'transparent' },
      },
      '::selection': {
        background: 'rgba(127, 238, 100, 0.22)',
        color: '#fff',
      },
      'code, kbd, pre, samp': { fontFamily: 'var(--font-mono)' },
      // Composer caret blink (used as a class)
      '@keyframes rli-blink': { '0%, 49%': { opacity: 1 }, '50%, 100%': { opacity: 0 } },
      '.rli-caret': {
        display: 'inline-block',
        width: '0.55ch',
        marginLeft: '0.15ch',
        background: 'var(--accent)',
        color: 'transparent',
        height: '1.05em',
        verticalAlign: 'text-bottom',
        animation: 'rli-blink 1.05s steps(1) infinite',
      },
    },
  },
};

const components: ThemeOptions['components'] = {
  ...cssBaseline,
  MuiPaper: { styleOverrides: { root: { backgroundImage: 'none' } } },
  MuiButton: {
    defaultProps: { disableElevation: true, disableRipple: true },
    styleOverrides: {
      root: {
        borderRadius: 'var(--radius-md)',
        fontWeight: 600,
        letterSpacing: '-0.005em',
        transition: 'background 0.12s ease, border-color 0.12s ease, color 0.12s ease',
        textTransform: 'none',
      },
      outlined: {
        borderColor: 'var(--border)',
        color: 'var(--text)',
        '&:hover': { borderColor: 'var(--border-hover)', background: 'var(--accent-soft)' },
      },
      text: { color: 'var(--text)', '&:hover': { background: 'var(--surface-hover)' } },
      contained: {
        background: 'var(--accent)',
        color: '#0a0b0c',
        '&:hover': { background: 'var(--accent-strong)' },
        '&.Mui-disabled': {
          background: 'var(--surface)',
          color: 'var(--text-faint)',
        },
      },
    },
  },
  MuiIconButton: {
    defaultProps: { disableRipple: true },
    styleOverrides: {
      root: {
        borderRadius: 'var(--radius-sm)',
        color: 'var(--text-muted)',
        transition: 'background 0.1s ease, color 0.1s ease, border-color 0.1s ease',
        '&:hover': { color: 'var(--text)', background: 'var(--surface-hover)' },
      },
    },
  },
  MuiTextField: {
    defaultProps: { variant: 'outlined' },
    styleOverrides: {
      root: {
        '& .MuiOutlinedInput-root': {
          borderRadius: 'var(--radius-md)',
          background: 'var(--surface)',
          color: 'var(--text)',
          fontSize: '0.88rem',
          '& fieldset': { borderColor: 'var(--border)', transition: 'border-color 0.12s' },
          '&:hover fieldset': { borderColor: 'var(--border-strong)' },
          '&.Mui-focused fieldset': {
            borderColor: 'var(--accent)',
            borderWidth: 1,
          },
        },
        '& .MuiInputBase-input::placeholder': { color: 'var(--text-faint)', opacity: 1 },
      },
    },
  },
  MuiDrawer: {
    styleOverrides: {
      paper: {
        backgroundColor: 'var(--panel)',
        backgroundImage: 'none',
        borderRight: '1px solid var(--border)',
      },
    },
  },
  MuiTooltip: {
    styleOverrides: {
      tooltip: {
        background: '#191c1f',
        color: 'var(--text)',
        fontFamily: 'var(--font-mono)',
        fontSize: '0.7rem',
        padding: '4px 8px',
        borderRadius: 4,
        border: '1px solid var(--border-strong)',
      },
      arrow: { color: '#191c1f' },
    },
  },
  MuiCircularProgress: { styleOverrides: { root: { color: 'var(--accent)' } } },
  MuiDivider: { styleOverrides: { root: { borderColor: 'var(--border)' } } },
};

export const darkTheme = createTheme({
  palette: {
    mode: 'dark',
    primary: { main: '#7FEE64', contrastText: '#0a0b0c' },
    secondary: { main: '#7FEE64' },
    background: { default: '#0a0b0c', paper: '#0c0d0f' },
    text: { primary: '#e6e8eb', secondary: '#8a8f96' },
    divider: 'rgba(255,255,255,0.06)',
    success: { main: '#7FEE64' },
    warning: { main: '#ffaa3a' },
    error: { main: '#ff6b6b' },
    info: { main: '#69b6ff' },
  },
  typography,
  components,
  shape: { borderRadius: 8 },
});

export default darkTheme;
