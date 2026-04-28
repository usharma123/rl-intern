import {
  Box,
  Drawer,
  IconButton,
  Tooltip,
  Typography,
  useMediaQuery,
  useTheme,
} from '@mui/material';
import MenuIcon from '@mui/icons-material/Menu';
import ChevronLeftIcon from '@mui/icons-material/ChevronLeft';

import { useLayoutStore } from '@/store/layoutStore';
import { useSessionStore } from '@/store/sessionStore';
import Sidebar from '@/components/Sidebar/Sidebar';
import SessionChat from '@/components/Chat/SessionChat';

const DRAWER_WIDTH = 252;

function BrandMark({ size = 22 }: { size?: number }) {
  return (
    <Box
      sx={{
        width: size,
        height: size,
        borderRadius: '4px',
        bgcolor: 'var(--accent)',
        color: '#0a0b0c',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        fontFamily: 'var(--font-mono)',
        fontWeight: 700,
        fontSize: size * 0.5,
        letterSpacing: '-0.02em',
        flexShrink: 0,
      }}
    >
      r/i
    </Box>
  );
}

export default function AppLayout() {
  const { isLeftSidebarOpen, toggleLeftSidebar, setLeftSidebarOpen } = useLayoutStore();
  const { sessions, activeSessionId } = useSessionStore();
  const theme = useTheme();
  const isMobile = useMediaQuery(theme.breakpoints.down('md'));
  const activeSession = sessions.find((s) => s.id === activeSessionId);

  const sidebarDrawer = (
    <Drawer
      variant={isMobile ? 'temporary' : 'persistent'}
      anchor="left"
      open={isLeftSidebarOpen}
      onClose={() => setLeftSidebarOpen(false)}
      ModalProps={{ keepMounted: true }}
      sx={{
        '& .MuiDrawer-paper': {
          boxSizing: 'border-box',
          width: DRAWER_WIDTH,
          top: 0,
          height: '100%',
        },
      }}
    >
      <Sidebar />
    </Drawer>
  );

  return (
    <Box sx={{ display: 'flex', width: '100%', height: '100%' }}>
      {isMobile ? (
        sidebarDrawer
      ) : (
        <Box
          component="nav"
          sx={{
            width: isLeftSidebarOpen ? DRAWER_WIDTH : 0,
            flexShrink: 0,
            transition: 'width 180ms cubic-bezier(0.4, 0, 0.2, 1)',
            overflow: 'hidden',
          }}
        >
          {sidebarDrawer}
        </Box>
      )}

      <Box
        sx={{
          flexGrow: 1,
          height: '100%',
          display: 'flex',
          flexDirection: 'column',
          overflow: 'hidden',
          minWidth: 0,
        }}
      >
        {/* Slim header: brand + breadcrumb + sidebar toggle */}
        <Box
          sx={{
            height: 48,
            px: 1.5,
            display: 'flex',
            alignItems: 'center',
            gap: 1.25,
            borderBottom: '1px solid var(--border)',
            bgcolor: 'rgba(10, 11, 12, 0.7)',
            backdropFilter: 'saturate(140%) blur(8px)',
            flexShrink: 0,
          }}
        >
          <Tooltip title={isLeftSidebarOpen ? 'Hide sessions' : 'Show sessions'}>
            <IconButton onClick={toggleLeftSidebar} size="small" sx={{ width: 28, height: 28 }}>
              {isLeftSidebarOpen && !isMobile ? (
                <ChevronLeftIcon sx={{ fontSize: 18 }} />
              ) : (
                <MenuIcon sx={{ fontSize: 18 }} />
              )}
            </IconButton>
          </Tooltip>

          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, minWidth: 0, flex: 1 }}>
            <BrandMark size={18} />
            <Typography
              sx={{
                fontFamily: 'var(--font-mono)',
                fontSize: '0.78rem',
                fontWeight: 600,
                color: 'var(--text)',
                letterSpacing: '-0.01em',
              }}
            >
              rl-intern
            </Typography>
            {activeSession && (
              <>
                <Typography
                  sx={{
                    fontFamily: 'var(--font-mono)',
                    fontSize: '0.78rem',
                    color: 'var(--text-faint)',
                  }}
                >
                  /
                </Typography>
                <Typography
                  sx={{
                    fontFamily: 'var(--font-mono)',
                    fontSize: '0.78rem',
                    color: 'var(--text-muted)',
                    overflow: 'hidden',
                    textOverflow: 'ellipsis',
                    whiteSpace: 'nowrap',
                  }}
                >
                  {activeSession.id}
                </Typography>
              </>
            )}
          </Box>
        </Box>

        <Box
          component="main"
          sx={{
            flexGrow: 1,
            display: 'flex',
            flexDirection: 'column',
            overflow: 'hidden',
            minWidth: 0,
          }}
        >
          {activeSessionId ? (
            sessions.map((s) => (
              <SessionChat key={s.id} sessionId={s.id} isActive={s.id === activeSessionId} />
            ))
          ) : (
            <Box
              sx={{
                flex: 1,
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                color: 'var(--text-faint)',
                fontFamily: 'var(--font-mono)',
                fontSize: '0.8rem',
                letterSpacing: '0.06em',
                textTransform: 'uppercase',
              }}
            >
              [ no session selected ]
            </Box>
          )}
        </Box>
      </Box>
    </Box>
  );
}
