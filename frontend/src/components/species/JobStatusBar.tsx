import { AlertCircle, CheckCircle2, Loader2 } from 'lucide-react'

import type { JobOut } from '@/api/types'
import { Progress } from '@/components/ui/progress'
import { formatRelative } from '@/lib/utils'

const STAGE_LABEL: Record<string, string> = {
  raw_occurrences: 'Fetching occurrences',
  features: 'Extracting bioclim features',
  model: 'Training MaxEnt (spatial block CV)',
  suitability_raster: 'Projecting suitability + MESS',
  bulletin: 'Rendering bulletin',
  scenarios: 'Projecting climate scenarios',
  validation: 'Evaluating reference species',
  scenario_raster: 'Projecting climate scenarios',
  done: 'Finishing',
}

/** Live progress for the latest job (polled via TanStack Query refetchInterval). */
export function JobStatusBar({ job }: { job: JobOut | undefined }) {
  if (!job) return null
  const active = job.status === 'queued' || job.status === 'running'
  const recentlyDone = !active && Date.now() - new Date(job.updated_ts).getTime() < 5 * 60_000
  if (!active && !recentlyDone) return null

  const stage = job.stage ? (STAGE_LABEL[job.stage] ?? job.stage) : job.status === 'queued' ? 'Queued' : 'Starting'
  return (
    <div className="rounded-xl border bg-card px-4 py-3">
      <div className="mb-2 flex items-center gap-2 text-sm">
        {active ? (
          <Loader2 className="size-4 animate-spin text-primary" />
        ) : job.status === 'succeeded' ? (
          <CheckCircle2 className="size-4 text-success" />
        ) : (
          <AlertCircle className="size-4 text-destructive" />
        )}
        <span className="font-medium">
          {job.kind === 'bulletin' ? 'Bulletin' : job.kind === 'scenarios'
              ? 'Scenario projection'
              : job.kind === 'validation'
                ? 'Validation suite'
                : 'Pipeline run'} ·{' '}
          {active ? stage : job.status === 'succeeded' ? 'completed' : 'failed'}
        </span>
        <span className="ml-auto text-xs text-muted-foreground">
          {active ? `${Math.round(job.progress * 100)}%` : formatRelative(job.updated_ts)}
        </span>
      </div>
      {active && <Progress value={job.progress * 100} />}
      {job.message && (
        <p className={`mt-2 text-xs ${job.status === 'failed' ? 'text-destructive' : 'text-muted-foreground'}`}>
          {job.message}
          {job.result && typeof job.result.decision === 'string' && ` — ${job.result.decision.replaceAll('_', ' ')}`}
        </p>
      )}
    </div>
  )
}
