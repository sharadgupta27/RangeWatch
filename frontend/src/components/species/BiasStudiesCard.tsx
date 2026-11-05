import { useQueryClient } from '@tanstack/react-query'
import { Loader2, Play, Scale } from 'lucide-react'
import { useEffect, useRef } from 'react'

import {
  jobJustFinished,
  qk,
  usePermissions,
  useRunSharedBias,
  useRunTargetGroupComparison,
  useSharedBias,
  useSpeciesJobs,
  useTargetGroupComparison,
} from '@/api/queries'
import type { JobOut, SharedBiasOut, TargetGroupComparisonOut } from '@/api/types'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table'
import { formatDate, formatNumber } from '@/lib/utils'

const num = (v: number | null | undefined, d = 2) => (v == null ? '–' : v.toFixed(d))

/** Latest species job of one kind; refreshes `key` when it finishes. */
function useJobOfKind(taxonKey: number, kind: JobOut['kind'], key: readonly unknown[]) {
  const qc = useQueryClient()
  const jobs = useSpeciesJobs(taxonKey)
  const job = jobs.data?.find((j) => j.kind === kind)
  const prev = useRef<JobOut | undefined>(undefined)
  useEffect(() => {
    const before = prev.current
    prev.current = job
    if (jobJustFinished(before, job)) void qc.invalidateQueries({ queryKey: key })
  }, [job, key, qc])
  return job
}

const isActive = (j: JobOut | undefined) => j?.status === 'queued' || j?.status === 'running'

/** Independent-data AUC / CBI of one comparison option, compactly. */
export function independentSummary(o: TargetGroupComparisonOut['options'][number]): string {
  if (!o.independent?.length) return 'no independent data'
  return o.independent
    .map((d) => {
      const m = (d.metrics ?? {}) as { auc_presence_absence?: number | null; cbi?: number | null }
      return `${String(d.name)}: AUC ${num(m.auc_presence_absence)} · CBI ${num(m.cbi)}`
    })
    .join('; ')
}

export function TargetGroupTable({ report }: { report: TargetGroupComparisonOut }) {
  return (
    <div className="space-y-2">
      <Table>
        <TableHeader>
          <TableRow>
            <TableHead>Target group</TableHead>
            <TableHead>Background used</TableHead>
            <TableHead className="text-right">CV CBI</TableHead>
            <TableHead className="text-right">CV AUC</TableHead>
            <TableHead>Independent data</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {report.options.map((o) => (
            <TableRow key={o.option}>
              <TableCell className="text-xs">
                {o.option}
                {o.option === report.best_by_cv_cbi && <Badge className="ml-2">best CV</Badge>}
                {o.target_group_name && (
                  <div className="text-[11px] text-muted-foreground">
                    {o.target_group_name} · {formatNumber(o.target_group_records)} records
                  </div>
                )}
                {o.matched && (
                  <div className="break-all text-[11px] text-muted-foreground">{JSON.stringify(o.matched)}</div>
                )}
              </TableCell>
              <TableCell className="text-xs">
                {o.error ? (
                  <span className="text-destructive">{o.error}</span>
                ) : (
                  <>
                    <span className="font-mono">{o.background_method}</span>
                    {o.fallback_reason && <div className="text-[11px] text-warning">{o.fallback_reason}</div>}
                  </>
                )}
              </TableCell>
              <TableCell className="text-right font-mono text-xs">{num(o.cv_cbi_mean)}</TableCell>
              <TableCell className="text-right font-mono text-xs">{num(o.cv_auc_mean)}</TableCell>
              <TableCell className="text-[11px] text-muted-foreground">{independentSummary(o)}</TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
      <p className="text-[11px] text-muted-foreground">
        {report.criterion}. {report.note}. {formatDate(report.created_ts, true)} · {report.reference}
      </p>
    </div>
  )
}

export function SharedBiasSummary({ report }: { report: SharedBiasOut }) {
  return (
    <div className="space-y-1.5">
      <div>
        {report.target_group.rank as string} <strong>{report.target_group.name as string}</strong>, from{' '}
        {report.species.length} species · {report.converged ? 'converged' : 'did not converge'} ·{' '}
        {formatDate(report.created_ts, true)}
      </div>
      <div className="flex flex-wrap gap-2">
        {Object.entries(report.coefficients).map(([k, v]) => (
          <Badge key={k} variant="secondary" className="font-mono">
            {k} {v.toFixed(3)}
          </Badge>
        ))}
      </div>
      <div className="text-muted-foreground">
        {report.species.map((s) => s.scientific_name ?? s.taxon_key).join(', ')}
        {!!report.skipped?.length && ` · ${report.skipped.length} skipped`}
      </div>
      <p className="text-[11px] text-muted-foreground">
        {report.method} ({report.reference}). {report.use}.
      </p>
    </div>
  )
}

/** B1 + B9: which target group mirrors the observers best, and the bias shared by a group. */
export function BiasStudiesCard({ taxonKey }: { taxonKey: number }) {
  const comparison = useTargetGroupComparison(taxonKey)
  const shared = useSharedBias(taxonKey)
  const runComparison = useRunTargetGroupComparison(taxonKey)
  const runShared = useRunSharedBias(taxonKey)
  const { run: canRun } = usePermissions()
  const cJob = useJobOfKind(taxonKey, 'target_groups', qk.targetGroups(taxonKey))
  const sJob = useJobOfKind(taxonKey, 'shared_bias', qk.sharedBias(taxonKey))
  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Scale className="size-4" /> Sampling-bias studies
        </CardTitle>
        <CardDescription className="max-w-3xl">
          Neither changes the published model. The comparison refits with target groups of other
          ranks, and with effort restricted to the presences&apos; data sources and years. The
          shared bias pools every tracked species of the same target group.
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-5 text-xs">
        <section className="space-y-2">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <h4 className="font-medium">Target-group comparison</h4>
            <Button
              size="sm"
              variant="outline"
              disabled={!canRun || isActive(cJob) || runComparison.isPending}
              onClick={() => runComparison.mutate()}
            >
              {isActive(cJob) || runComparison.isPending ? <Loader2 className="animate-spin" /> : <Play />} Compare
            </Button>
          </div>
          {runComparison.isError && <p className="text-destructive">{runComparison.error.message}</p>}
          {comparison.data ? (
            <TargetGroupTable report={comparison.data} />
          ) : (
            <p className="text-muted-foreground">Not run yet.</p>
          )}
        </section>
        <section className="space-y-2 border-t pt-3">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <h4 className="font-medium">Shared observer bias</h4>
            <Button
              size="sm"
              variant="outline"
              disabled={!canRun || isActive(sJob) || runShared.isPending}
              onClick={() => runShared.mutate()}
            >
              {isActive(sJob) || runShared.isPending ? <Loader2 className="animate-spin" /> : <Play />} Estimate
            </Button>
          </div>
          {runShared.isError && <p className="text-destructive">{runShared.error.message}</p>}
          {sJob?.status === 'failed' && <p className="text-destructive">{sJob.message}</p>}
          {shared.data ? (
            <SharedBiasSummary report={shared.data} />
          ) : (
            <p className="text-muted-foreground">
              No estimate for this species&apos; target group yet. It needs at least three tracked
              species of the group with a trained model.
            </p>
          )}
        </section>
      </CardContent>
    </Card>
  )
}
