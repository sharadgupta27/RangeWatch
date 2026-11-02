import { Info, Radar, ShieldCheck } from 'lucide-react'

import type { DataQualityOut, SamplingEffortOut } from '@/api/types'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table'
import { Tooltip, TooltipContent, TooltipTrigger } from '@/components/ui/tooltip'
import { formatNumber } from '@/lib/utils'
import { mapActions } from '@/store/mapStore'

const pct = (v: number | null | undefined) => (v == null ? '–' : `${(v * 100).toFixed(1)}%`)

const TIER_ORDER = ['A', 'B', 'C', 'excluded'] as const

/**
 * The record-level data-quality assessment of the current model version: every test with what
 * it checks, whether its flags exclude records from training, and how many it flagged — tests
 * that could not run are listed as skipped with the reason, never shown as passed.
 */
export function DataQualityCard({ quality }: { quality: DataQualityOut | null | undefined }) {
  if (!quality) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <ShieldCheck className="size-4" /> Data quality
          </CardTitle>
          <CardDescription>
            This model version was trained before the data-quality tests existed. Retrain to
            assess its records.
          </CardDescription>
        </CardHeader>
      </Card>
    )
  }
  const skipped = quality.tests.filter((t) => t.status === 'skipped')
  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <ShieldCheck className="size-4" /> Data quality
        </CardTitle>
        <CardDescription>
          Each record with climate data is checked before training. Error tests (likely wrong
          locations: centroids, institutions, GBIF&apos;s address, imprecise coordinates) exclude
          the record; warning tests only flag it for review. An isolated or climatically unusual
          record can be the invasion front.
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-3">
        <div className="flex flex-wrap gap-2">
          <Badge variant="secondary">{formatNumber(quality.n_records)} records assessed</Badge>
          <Badge variant={quality.n_excluded > 0 ? 'warning' : 'secondary'}>
            {formatNumber(quality.n_excluded)} excluded from training
          </Badge>
          <Badge variant="secondary">{formatNumber(quality.n_flagged_kept)} flagged, kept</Badge>
          {quality.n_outside_climate_grid != null && quality.n_outside_climate_grid > 0 && (
            <Badge variant="secondary">
              {formatNumber(quality.n_outside_climate_grid)} outside the climate grid
            </Badge>
          )}
        </div>
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>Test</TableHead>
              <TableHead>Effect</TableHead>
              <TableHead className="text-right">Flagged</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {quality.tests.map((t) => (
              <TableRow key={t.id} className={t.status === 'skipped' ? 'opacity-60' : undefined}>
                <TableCell className="font-medium">
                  <Tooltip>
                    <TooltipTrigger className="flex items-center gap-1.5 text-left" aria-label={`About ${t.label}`}>
                      {t.label} <Info className="size-3 shrink-0 text-muted-foreground" />
                    </TooltipTrigger>
                    <TooltipContent className="max-w-xs">
                      {t.description}
                      {t.reference ? ` (${t.reference})` : ''}
                    </TooltipContent>
                  </Tooltip>
                </TableCell>
                <TableCell className="text-xs">
                  {t.status === 'skipped' ? (
                    <span className="text-muted-foreground">Skipped: {t.reason}</span>
                  ) : t.excluded ? (
                    <Badge variant="warning">excludes</Badge>
                  ) : (
                    <Badge variant="secondary">flags only</Badge>
                  )}
                </TableCell>
                <TableCell className="text-right tabular-nums">
                  {t.status === 'skipped' ? '–' : formatNumber(t.n_flagged)}
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
        <div className="grid gap-2 sm:grid-cols-4">
          {TIER_ORDER.map((tier) => (
            <Tooltip key={tier}>
              <TooltipTrigger asChild>
                <div className="cursor-help rounded-lg border px-3 py-2 text-xs">
                  <div className="text-muted-foreground">
                    {tier === 'excluded' ? 'Excluded' : `Tier ${tier}`}
                  </div>
                  <div className="text-base font-semibold tabular-nums">
                    {formatNumber(quality.tiers[tier] ?? 0)}
                  </div>
                </div>
              </TooltipTrigger>
              <TooltipContent className="max-w-xs">{quality.tier_rules[tier]}</TooltipContent>
            </Tooltip>
          ))}
        </div>
        <div className="space-y-1 text-[11px] text-muted-foreground">
          <p>
            Tests {quality.version}
            {quality.reference
              ? ` · reference gazetteers ${String(quality.reference.version)}`
              : ' · no reference gazetteers installed'}
            {quality.grid_cell_m != null &&
              ` · precision judged against ${formatNumber(quality.grid_cell_m / 1000, 1)} km grid cells`}
            {quality.n_issues_unknown > 0 &&
              ` · ${formatNumber(quality.n_issues_unknown)} GBIF records ingested before issue flags were kept`}
            .
          </p>
          {skipped.length > 0 && (
            <p>
              {skipped.length} test{skipped.length > 1 ? 's' : ''} could not run; build the
              reference layer with <code>python -m src.features.quality_refs build</code>.
            </p>
          )}
          {quality.snapshot && (
            <p className="font-mono">
              quality_flags.parquet · sha256 {quality.snapshot.sha256.slice(0, 16)}…
            </p>
          )}
        </div>
      </CardContent>
    </Card>
  )
}

/**
 * Where the target group (e.g. the species' order) has been recorded at all. In unsampled
 * cells, missing presences say nothing about suitability.
 */
export function SamplingEffortCard({ effort }: { effort: SamplingEffortOut | null | undefined }) {
  if (!effort) return null
  const group = effort.target_group as { rank?: string; name?: string } | null | undefined
  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Radar className="size-4" /> Sampling effort
        </CardTitle>
        <CardDescription>
          GBIF records of the target group
          {group?.name ? ` (${group.rank} ${group.name})` : ''}, the same observer effort the
          background is drawn from. Where it is zero, an absence of presences is uninformative.
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-3">
        {effort.status !== 'available' ? (
          <p className="text-sm text-muted-foreground">
            {effort.status === 'disabled'
              ? 'The effort layer is disabled (SDM_EFFORT_LAYER_ENABLED).'
              : `Effort data unavailable: ${effort.reason ?? 'unknown reason'}.`}
          </p>
        ) : (
          <>
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Area</TableHead>
                  <TableHead className="text-right">Without any target-group record</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {(
                  [
                    ['All land', effort.land_unsampled_fraction],
                    ['Suitable native range', effort.native_suitable_unsampled_fraction],
                    ['Established outside', effort.established_outside_unsampled_fraction],
                    ['Candidate invasion zone', effort.candidate_unsampled_fraction],
                  ] as const
                ).map(([label, v]) => (
                  <TableRow key={label}>
                    <TableCell>{label}</TableCell>
                    <TableCell className="text-right tabular-nums">{pct(v)}</TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
            <div className="flex flex-wrap items-center justify-between gap-2">
              <p className="text-[11px] text-muted-foreground">
                {formatNumber(effort.n_records)} records · pixels of{' '}
                {effort.pixel_deg?.toFixed(3)}° (GBIF map zoom {effort.map_zoom})
              </p>
              <Button size="sm" variant="outline" onClick={() => mapActions.setShowEffort(true)}>
                <Radar /> Show on map
              </Button>
            </div>
          </>
        )}
      </CardContent>
    </Card>
  )
}
