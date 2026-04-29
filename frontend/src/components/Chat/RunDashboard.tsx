import { Box, IconButton, Tooltip, Typography } from '@mui/material';
import CloseIcon from '@mui/icons-material/Close';
import InsightsOutlinedIcon from '@mui/icons-material/InsightsOutlined';
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
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

export function hasRunDashboardContent(
  plan: PlanDashboardState,
  jobs: JobDashboardState[],
  artifacts: ArtifactDashboardState,
): boolean {
  const hasPlan = Boolean(plan.plan || plan.stage);
  const hasJobs = jobs.length > 0;
  const count = artifactCount(artifacts.manifest);
  return hasPlan || hasJobs || count > 0;
}

function tone(status?: string): StatusTone {
  const s = (status ?? '').toLowerCase();
  if (s.includes('fail') || s.includes('error') || s.includes('cancel')) return 'err';
  if (s.includes('running') || s.includes('pending') || s.includes('updated')) return 'warn';
  return 'ok';
}

function verdictTone(verdict?: string): StatusTone {
  const v = (verdict ?? '').toLowerCase();
  if (v === 'regressed') return 'err';
  if (v === 'inconclusive' || v === 'smoke_only') return 'warn';
  return 'ok';
}

const ARTIFACT_BUCKETS = [
  'checkpoints',
  'adapters',
  'configs',
  'metrics',
  'logs',
  'videos',
  'reports',
  'samples',
  'errors',
] as const;

export function artifactCount(manifest?: Record<string, unknown>): number {
  if (!manifest) return 0;
  const bucketed = ARTIFACT_BUCKETS.reduce<number>((acc, bucket) => {
    const value = manifest[bucket];
    return Array.isArray(value) ? acc + value.length : acc;
  }, 0);
  if (bucketed > 0) return bucketed;
  return Array.isArray(manifest.artifacts) ? manifest.artifacts.length : 0;
}

function artifactBuckets(manifest?: Record<string, unknown>) {
  if (!manifest) return [];
  return ARTIFACT_BUCKETS.map((bucket) => {
    const items = manifest[bucket];
    if (!Array.isArray(items) || items.length === 0) return null;
    const names = items
      .slice(-2)
      .map((item) => {
        if (item && typeof item === 'object' && 'name' in item)
          return String((item as { name?: unknown }).name);
        if (item && typeof item === 'object' && 'path' in item)
          return String((item as { path?: unknown }).path).split('/').pop();
        return String(item);
      })
      .filter(Boolean);
    return { bucket, count: items.length, names };
  }).filter(Boolean) as Array<{ bucket: string; count: number; names: string[] }>;
}

function evidenceSummary(manifest?: Record<string, unknown>) {
  if (!manifest) return null;
  const metrics = manifest.metrics;
  if (!Array.isArray(metrics)) return null;
  const summaries = metrics
    .map((item) => {
      if (!item || typeof item !== 'object') return null;
      const objectItem = item as { name?: unknown; kind?: unknown; path?: unknown; metadata?: unknown };
      const metadata = objectItem.metadata;
      if (!metadata || typeof metadata !== 'object') return null;
      const meta = metadata as { verdict?: unknown; run_class?: unknown; reason?: unknown };
      if (!meta.verdict && !meta.run_class) return null;
      return {
        name: String(objectItem.name ?? objectItem.kind ?? objectItem.path ?? 'evaluation'),
        verdict: String(meta.run_class === 'smoke_only' ? 'smoke_only' : meta.verdict ?? 'inconclusive'),
        reason: meta.reason ? String(meta.reason) : '',
      };
    })
    .filter(Boolean) as Array<{ name: string; verdict: string; reason: string }>;
  return summaries.length ? summaries[summaries.length - 1] : null;
}

function SectionLabel({ children }: { children: React.ReactNode }) {
  return (
    <Typography
      sx={{
        fontFamily: 'var(--font-mono)',
        fontSize: '0.6rem',
        color: 'var(--text-faint)',
        letterSpacing: '0.18em',
        mb: 0.875,
        textTransform: 'uppercase',
      }}
    >
      {children}
    </Typography>
  );
}

export default function RunDashboard({ plan, jobs, artifacts, open, onOpenChange }: Props) {
  if (!hasRunDashboardContent(plan, jobs, artifacts)) return null;

  const count = artifactCount(artifacts.manifest);
  const domain = String(plan.plan?.domain ?? 'no-domain');
  const objective = String(plan.plan?.objective ?? 'experiment plan pending');
  const stage = plan.stage ?? 'no active stage';
  const status = plan.status ?? 'updated';
  const buckets = artifactBuckets(artifacts.manifest);
  const evidence = evidenceSummary(artifacts.manifest);

  // Collapsed: small floating tab to reopen
  if (!open) {
    return (
      <Tooltip title="show run monitor" placement="left">
        <IconButton
          onClick={() => onOpenChange(true)}
          aria-label="show run monitor"
          sx={{
            position: 'absolute',
            top: 16,
            right: 16,
            zIndex: 12,
            width: 36,
            height: 36,
            borderRadius: 'var(--radius-md)',
            border: '1px solid var(--border-strong)',
            background: 'rgba(12, 13, 15, 0.85)',
            backdropFilter: 'saturate(140%) blur(8px)',
            color: 'var(--accent)',
            transition: 'border-color 120ms, background 120ms, box-shadow 120ms',
            '&:hover': {
              borderColor: 'var(--accent-border)',
              background: 'var(--accent-soft)',
              boxShadow: '0 0 0 1px var(--accent-border), 0 0 20px rgba(127, 238, 100, 0.12)',
            },
          }}
        >
          <InsightsOutlinedIcon sx={{ fontSize: 18 }} />
        </IconButton>
      </Tooltip>
    );
  }

  return (
    <Box
      sx={{
        position: 'absolute',
        top: 16,
        right: 16,
        bottom: 84,
        width: { xs: 'calc(100% - 24px)', sm: 320 },
        maxHeight: 'calc(100% - 100px)',
        zIndex: 12,
        display: 'flex',
        flexDirection: 'column',
        borderRadius: 'var(--radius-lg)',
        border: '1px solid var(--border-strong)',
        background:
          'linear-gradient(180deg, rgba(16, 18, 20, 0.92) 0%, rgba(10, 11, 12, 0.92) 100%)',
        backdropFilter: 'saturate(160%) blur(14px)',
        boxShadow:
          '0 0 0 1px rgba(127, 238, 100, 0.06), 0 18px 48px rgba(0, 0, 0, 0.55), 0 2px 6px rgba(0, 0, 0, 0.4)',
        overflow: 'hidden',
        animation: 'rli-panel-in 220ms cubic-bezier(0.2, 0.8, 0.2, 1)',
        '@keyframes rli-panel-in': {
          '0%': { opacity: 0, transform: 'translateX(8px) scale(0.985)' },
          '100%': { opacity: 1, transform: 'translateX(0) scale(1)' },
        },
        '&::before': {
          content: '""',
          position: 'absolute',
          inset: 0,
          pointerEvents: 'none',
          borderRadius: 'inherit',
          background:
            'radial-gradient(120% 60% at 100% 0%, rgba(127, 238, 100, 0.06), transparent 60%)',
        },
      }}
    >
      {/* Header bar */}
      <Box
        sx={{
          display: 'flex',
          alignItems: 'center',
          gap: 1,
          px: 1.5,
          py: 1.125,
          borderBottom: '1px solid var(--border)',
          flexShrink: 0,
        }}
      >
        <Box
          sx={{
            width: 6,
            height: 6,
            borderRadius: '50%',
            bgcolor: 'var(--accent)',
            boxShadow: '0 0 8px rgba(127, 238, 100, 0.6)',
          }}
        />
        <Typography
          sx={{
            fontFamily: 'var(--font-mono)',
            fontSize: '0.68rem',
            color: 'var(--text)',
            letterSpacing: '0.18em',
            textTransform: 'uppercase',
            fontWeight: 600,
            flex: 1,
          }}
        >
          run monitor
        </Typography>
        <Tooltip title="hide">
          <IconButton
            size="small"
            onClick={() => onOpenChange(false)}
            aria-label="hide run monitor"
            sx={{ width: 22, height: 22 }}
          >
            <CloseIcon sx={{ fontSize: 14 }} />
          </IconButton>
        </Tooltip>
      </Box>

      {/* Body */}
      <Box
        sx={{
          flex: 1,
          overflow: 'auto',
          px: 1.75,
          py: 1.5,
          display: 'flex',
          flexDirection: 'column',
          gap: 1.75,
          position: 'relative',
        }}
      >
        {/* PLAN */}
        <Box>
          <SectionLabel>plan</SectionLabel>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.875, mb: 0.875 }}>
            <StatusChip label={status.toUpperCase()} tone={tone(status)} />
            <Typography
              sx={{
                fontFamily: 'var(--font-mono)',
                fontSize: '0.74rem',
                color: 'var(--accent)',
                letterSpacing: '-0.005em',
              }}
            >
              {domain}
            </Typography>
          </Box>
          <Typography
            sx={{
              fontSize: '0.82rem',
              color: 'var(--text)',
              lineHeight: 1.45,
              letterSpacing: '-0.005em',
              mb: 0.625,
            }}
          >
            {objective}
          </Typography>
          <Box
            sx={{
              display: 'inline-flex',
              alignItems: 'center',
              gap: 0.625,
              fontFamily: 'var(--font-mono)',
              fontSize: '0.7rem',
              color: 'var(--text-muted)',
              px: 0.875,
              py: 0.375,
              borderRadius: 'var(--radius-sm)',
              background: 'var(--code-bg)',
              border: '1px solid var(--border)',
            }}
          >
            <Box component="span" sx={{ color: 'var(--text-faint)' }}>
              stage
            </Box>
            <Box component="span" sx={{ color: 'var(--text)' }}>
              {stage}
            </Box>
          </Box>
        </Box>

        <Box sx={{ height: '1px', flexShrink: 0, background: 'var(--border)' }} />

        {/* EVIDENCE */}
        {evidence && (
          <>
            <Box>
              <SectionLabel>evidence</SectionLabel>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.875, mb: 0.75 }}>
                <StatusChip label={evidence.verdict.toUpperCase()} tone={verdictTone(evidence.verdict)} />
                <Typography
                  sx={{
                    fontFamily: 'var(--font-mono)',
                    fontSize: '0.7rem',
                    color: 'var(--text-muted)',
                    overflow: 'hidden',
                    textOverflow: 'ellipsis',
                    whiteSpace: 'nowrap',
                    minWidth: 0,
                  }}
                >
                  {evidence.name}
                </Typography>
              </Box>
              {evidence.reason && (
                <Typography
                  sx={{
                    fontSize: '0.74rem',
                    color: 'var(--text-muted)',
                    lineHeight: 1.4,
                    letterSpacing: 0,
                  }}
                >
                  {evidence.reason}
                </Typography>
              )}
            </Box>

            <Box sx={{ height: '1px', flexShrink: 0, background: 'var(--border)' }} />
          </>
        )}

        {/* JOBS */}
        <Box>
          <SectionLabel>jobs · {jobs.length}</SectionLabel>
          {jobs.length === 0 ? (
            <Typography
              sx={{
                fontFamily: 'var(--font-mono)',
                fontSize: '0.74rem',
                color: 'var(--text-faint)',
              }}
            >
              no remote jobs
            </Typography>
          ) : (
            <Box sx={{ display: 'flex', flexDirection: 'column', gap: 0.625 }}>
              {jobs.slice(0, 6).map((job, idx) => (
                <Box
                  key={`${job.jobId}-${job.backendId}-${idx}`}
                  sx={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: 0.875,
                    px: 0.875,
                    py: 0.625,
                    borderRadius: 'var(--radius-sm)',
                    border: '1px solid var(--border)',
                    background: 'rgba(255, 255, 255, 0.012)',
                  }}
                >
                  <StatusChip label={(job.status ?? 'JOB').toUpperCase()} tone={tone(job.status)} />
                  <Typography
                    sx={{
                      fontFamily: 'var(--font-mono)',
                      fontSize: '0.72rem',
                      color: job.error ? 'var(--error)' : 'var(--text)',
                      overflow: 'hidden',
                      textOverflow: 'ellipsis',
                      whiteSpace: 'nowrap',
                      flex: 1,
                      minWidth: 0,
                    }}
                  >
                    {job.stage ?? job.hardware ?? job.jobId ?? job.backendId}
                  </Typography>
                </Box>
              ))}
            </Box>
          )}
        </Box>

        <Box sx={{ height: '1px', flexShrink: 0, background: 'var(--border)' }} />

        {/* ARTIFACTS */}
        <Box>
          <SectionLabel>artifacts</SectionLabel>
          <Box sx={{ display: 'flex', alignItems: 'baseline', gap: 1, mb: 0.875 }}>
            <Typography
              sx={{
                fontFamily: 'var(--font-mono)',
                fontSize: '1.75rem',
                color: 'var(--accent)',
                lineHeight: 1,
                letterSpacing: '-0.02em',
                fontWeight: 500,
              }}
            >
              {count}
            </Typography>
            <Typography
              sx={{
                fontFamily: 'var(--font-mono)',
                fontSize: '0.66rem',
                color: 'var(--text-faint)',
                letterSpacing: '0.14em',
                textTransform: 'uppercase',
              }}
            >
              {count === 1 ? 'item' : 'items'}
            </Typography>
          </Box>
          {buckets.length > 0 && (
            <Box sx={{ display: 'flex', flexDirection: 'column', gap: 0.375 }}>
              {buckets.slice(0, 6).map((bucket) => (
                <Box
                  key={bucket.bucket}
                  sx={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: 0.75,
                    fontFamily: 'var(--font-mono)',
                    fontSize: '0.7rem',
                  }}
                >
                  <Box
                    sx={{
                      width: 22,
                      textAlign: 'right',
                      color:
                        bucket.bucket === 'errors' ? 'var(--error)' : 'var(--accent-strong)',
                      flexShrink: 0,
                    }}
                  >
                    {bucket.count}
                  </Box>
                  <Box
                    sx={{
                      color:
                        bucket.bucket === 'errors' ? 'var(--error)' : 'var(--text)',
                      flexShrink: 0,
                    }}
                  >
                    {bucket.bucket}
                  </Box>
                  {bucket.names.length > 0 && (
                    <Box
                      sx={{
                        color: 'var(--text-faint)',
                        overflow: 'hidden',
                        textOverflow: 'ellipsis',
                        whiteSpace: 'nowrap',
                        minWidth: 0,
                        flex: 1,
                      }}
                    >
                      · {bucket.names.join(', ')}
                    </Box>
                  )}
                </Box>
              ))}
            </Box>
          )}
        </Box>
      </Box>
    </Box>
  );
}
