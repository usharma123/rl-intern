import { Box } from '@mui/material';
import AppLayout from '@/components/Layout/AppLayout';
import WelcomeScreen from '@/components/WelcomeScreen/WelcomeScreen';
import { useSessionStore } from '@/store/sessionStore';

export default function App() {
  const sessions = useSessionStore((s) => s.sessions);
  const hasSessions = sessions.length > 0;

  return (
    <Box sx={{ height: '100vh', display: 'flex' }}>
      {hasSessions ? <AppLayout /> : <WelcomeScreen />}
    </Box>
  );
}
