import { Box, Typography } from '@mui/material';
import type {
  ArtifactDashboardState,
  JobDashboardState,
  PlanDashboardState,
} from '@/types/events';
import StatusChip, { type StatusTone } from '@/components/Common/StatusChip';

interface Props {
  plan: PlanDashboardState;
  jobs: JobDashboardState[];
  artifacts: ArtifactDashboardState;
}

function tone(status?: string): StatusTone {
  const s = (status ?? '').toLowerCase();
  if (s.includes('fail') || s.includes('error') || s.includes('cancel')) return 'err';
  if (s.includes('running') || s.includes('pending') || s.includes('updated')) return 'warn';
  return 'ok';
}

function artifactCount(manifest?: Record<string, unknown>): number {
  if (!manifest) return 0;
  return Object.values(manifest).reduce<number>((acc, value) => {
    if (Array.isArray(value)) return acc + value.length;
    return acc;
  }, 0);
}

export default function RunDashboard({ plan, jobs, artifacts }: Props) {
  const hasPlan = Boolean(plan.plan || plan.stage);
  const hasJobs = jobs.length > 0;
  const count = artifactCount(artifacts.manifest);
  if (!hasPlan && !hasJobs && count === 0) return null;

  const domain = String(plan.plan?.domain ?? 'no-domain');
  const objective = String(plan.plan?.objective ?? 'experiment plan pending');
  const stage = plan.stage ?? 'no active stage';
  const status = plan.status ?? 'updated';

  return (
    <Box
      sx={{
        mx: { xs: 1.5, md: 2.5 },
        mt: 1.25,
        p: 1.25,
        border: '1px solid var(--border)',
        borderRadius: 'var(--radius-md)',
        background: 'var(--surface)',
        display: 'grid',
        gridTemplateColumns: { xs: '1fr', md: '1.5fr 1fr 0.7fr' },
        gap: 1.25,
      }}
    >
      <Box sx={{ minWidth: 0 }}>
        <Typography
          sx={{
            fontFamily: 'var(--font-mono)',
            fontSize: '0.62rem',
            color: 'var(--text-faint)',
            letterSpacing: '0.16em',
            mb: 0.5,
          }}
        >
          PLAN
        </Typography>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 0.5 }}>
          <StatusChip label={status.toUpperCase()} tone={tone(status)} />
          <Typography
            sx={{
              fontFamily: 'var(--font-mono)',
              fontSize: '0.75rem',
              color: 'var(--accent)',
            }}
          >
            {domain}
          </Typography>
        </Box>
        <Typography
          sx={{
            fontSize: '0.82rem',
            color: 'var(--text)',
            whiteSpace: 'nowrap',
            overflow: 'hidden',
            textOverflow: 'ellipsis',
          }}
        >
          {objective}
        </Typography>
        <Typography sx={{ fontFamily: 'var(--font-mono)', fontSize: '0.72rem', color: 'var(--text-muted)' }}>
          stage: {stage}
        </Typography>
      </Box>

      <Box sx={{ minWidth: 0 }}>
        <Typography
          sx={{
            fontFamily: 'var(--font-mono)',
            fontSize: '0.62rem',
            color: 'var(--text-faint)',
            letterSpacing: '0.16em',
            mb: 0.5,
          }}
        >
          JOBS
        </Typography>
        {jobs.length === 0 ? (
          <Typography sx={{ fontFamily: 'var(--font-mono)', fontSize: '0.74rem', color: 'var(--text-muted)' }}>
            no remote jobs
          </Typography>
        ) : (
          jobs.slice(0, 3).map((job, idx) => (
            <Box key={`${job.jobId}-${job.backendId}-${idx}`} sx={{ display: 'flex', gap: 1, alignItems: 'center', mb: 0.35 }}>
              <StatusChip label={(job.status ?? 'JOB').toUpperCase()} tone={tone(job.status)} />
              <Typography
                sx={{
                  fontFamily: 'var(--font-mono)',
                  fontSize: '0.72rem',
                  color: job.error ? 'var(--error)' : 'var(--text)',
                  overflow: 'hidden',
                  textOverflow: 'ellipsis',
                  whiteSpace: 'nowrap',
                }}
              >
                {job.stage ?? job.hardware ?? job.jobId ?? job.backendId}
              </Typography>
            </Box>
          ))
        )}
      </Box>

      <Box>
        <Typography
          sx={{
            fontFamily: 'var(--font-mono)',
            fontSize: '0.62rem',
            color: 'var(--text-faint)',
            letterSpacing: '0.16em',
            mb: 0.5,
          }}
        >
          ARTIFACTS
        </Typography>
        <Typography sx={{ fontFamily: 'var(--font-mono)', fontSize: '1.4rem', color: 'var(--accent)', lineHeight: 1 }}>
          {count}
        </Typography>
      </Box>
    </Box>
  );
}
