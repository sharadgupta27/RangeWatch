import { Dna, Loader2, Play } from 'lucide-react'

import { useBenchmarkJob, useBenchmarkReport, usePermissions, useRunBenchmark } from '@/api/queries'
import type { BenchmarkRequest } from '@/api/types'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table'
import { formatDate } from '@/lib/utils'

import { JobStatusBar } from './JobStatusBar'

const DEFAULT_REQUEST: BenchmarkRequest = {
  n_presences: 300,
  bias: 'synthetic',
  bias_strength: 3,
  variants: ['no_correction', 'target_group', 'bias_covariate', 'density_weights'],
  seed: 42,
}
const num = (v: number | null | undefined) => (v == null ? '–' : v.toFixed(2))

/** §4.6: which sampling-bias correction recovers a known (virtual) niche. */
export function BenchmarkCard() {
  const report = useBenchmarkReport()
  const job = useBenchmarkJob()
  const run = useRunBenchmark()
  const { run: canRun } = usePermissions()
  const r = report.data
  const active = job.data?.status === 'queued' || job.data?.status === 'running'
  return (
    <Card>
      <CardHeader className="flex flex-row flex-wrap items-start justify-between gap-3">
        <div className="space-y-1.5">
          <CardTitle className="flex items-center gap-2">
            <Dna className="size-4" /> Virtual-species bias benchmark
          </CardTitle>
          <CardDescription className="max-w-3xl">
            A simulated species with a known niche is sampled with realistic observer bias, then
            each correction is fitted and compared with the truth. Spatial CV on the biased data
            is shown next to it: a correction can look better under CV without being better.
          </CardDescription>
        </div>
        <Button size="sm" onClick={() => run.mutate(DEFAULT_REQUEST)} disabled={!canRun || active || run.isPending}>
          {active || run.isPending ? <Loader2 className="animate-spin" /> : <Play />} Run benchmark
        </Button>
      </CardHeader>
      <CardContent className="space-y-3 text-xs">
        {job.data && <JobStatusBar job={job.data} />}
        {run.isError && <p className="text-destructive">{run.error.message}</p>}
        {!r ? (
          <p className="text-muted-foreground">Not run yet.</p>
        ) : (
          <>
            <p>
              <strong>{r.interpretation}</strong>{' '}
              <span className="text-muted-foreground">
                {r.n_presences} biased presences · {r.n_evaluation_cells.toLocaleString()} cells ·
                bias {String(r.bias.source)} (strength {String(r.bias.strength)}) ·{' '}
                {formatDate(r.created_ts, true)}
              </span>
            </p>
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Correction</TableHead>
                  <TableHead className="text-right">Spearman vs truth</TableHead>
                  <TableHead className="text-right">Schoener&apos;s D</TableHead>
                  <TableHead className="text-right">AUC vs true occupancy</TableHead>
                  <TableHead className="text-right">CV CBI (biased data)</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {r.variants.map((v) => (
                  <TableRow key={v.variant}>
                    <TableCell>
                      {v.variant.replaceAll('_', ' ')}
                      {v.variant === r.best_variant && <Badge className="ml-2">best</Badge>}
                      <div className="text-[11px] text-muted-foreground">{v.error ?? v.description}</div>
                    </TableCell>
                    <TableCell className="text-right font-mono">{num(v.spearman_vs_truth)}</TableCell>
                    <TableCell className="text-right font-mono">{num(v.schoeners_d)}</TableCell>
                    <TableCell className="text-right font-mono">{num(v.auc_vs_true_occupancy)}</TableCell>
                    <TableCell className="text-right font-mono">{num(v.cv_cbi_mean)}</TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </>
        )}
      </CardContent>
    </Card>
  )
}
