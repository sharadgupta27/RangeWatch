import { Info, Map as MapIcon, ScanSearch } from 'lucide-react'

import type { ExtrapolationSummaryOut } from '@/api/types'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table'
import { Tooltip, TooltipContent, TooltipTrigger } from '@/components/ui/tooltip'
import { formatCompact, formatNumber } from '@/lib/utils'
import { mapActions } from '@/store/mapStore'

const pct = (v: number) => `${(v * 100).toFixed(1)}%`

/**
 * The five extrapolation diagnostics side by side: what each one flags, how its threshold was
 * derived and how much land / candidate zone it flags — never a bare uncertainty number.
 */
export function ExtrapolationCard({ summary }: { summary: ExtrapolationSummaryOut }) {
  const weights = Object.entries(summary.aoa_weights).sort((a, b) => b[1] - a[1])
  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <ScanSearch className="size-4" /> Extrapolation diagnostics
        </CardTitle>
        <CardDescription>
          Five checks of where the projection leaves the conditions the model was trained on.
          MESS and exDet NT1 test predictor ranges, exDet NT2 finds novel combinations of
          in-range values, MOP and Shape measure distance to the closest analogous training
          conditions, and AOA measures distance in the importance-weighted space the model uses.
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-3">
        <div className="flex flex-wrap gap-2">
          <Badge variant="secondary">
            {pct(summary.candidate_consensus_ok_fraction)} of candidate zone flagged by none
          </Badge>
          <Badge variant={summary.candidate_consensus_majority_fraction > 0.25 ? 'warning' : 'secondary'}>
            {pct(summary.candidate_consensus_majority_fraction)} flagged by ≥ 3 of 5
          </Badge>
        </div>
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>Diagnostic</TableHead>
              <TableHead>Flags extrapolation when</TableHead>
              <TableHead className="text-right">Threshold</TableHead>
              <TableHead className="text-right">Land</TableHead>
              <TableHead className="text-right">Candidate zone</TableHead>
              <TableHead className="w-8" />
            </TableRow>
          </TableHeader>
          <TableBody>
            {summary.diagnostics.map((d) => (
              <TableRow key={d.id}>
                <TableCell className="font-medium">
                  <Tooltip>
                    <TooltipTrigger className="flex items-center gap-1.5" aria-label={`About ${d.label}`}>
                      {d.label} <Info className="size-3 text-muted-foreground" />
                    </TooltipTrigger>
                    <TooltipContent className="max-w-xs">
                      {d.name} ({d.reference}). Threshold: {d.threshold_rule}.
                    </TooltipContent>
                  </Tooltip>
                </TableCell>
                <TableCell className="text-xs text-muted-foreground">{d.flag_rule}</TableCell>
                <TableCell className="text-right tabular-nums">
                  {d.threshold == null ? '–' : d.threshold.toPrecision(3)}
                </TableCell>
                <TableCell className="text-right tabular-nums">{pct(d.land_fraction)}</TableCell>
                <TableCell className="text-right tabular-nums">
                  {pct(d.candidate_flagged_fraction)}
                </TableCell>
                <TableCell>
                  <Button
                    size="icon"
                    variant="ghost"
                    className="size-7"
                    aria-label={`Show ${d.label} on the map`}
                    onClick={() => mapActions.setProjection(d.id)}
                  >
                    <MapIcon />
                  </Button>
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
        <div className="space-y-1 text-[11px] text-muted-foreground">
          <p>
            Reference: {summary.reference_set}, n = {formatNumber(summary.n_reference)}. MOP averages
            the closest {summary.mop_percentage}% ({summary.mop_k} points). MOP, Shape and AOA
            thresholds come from the training data scored against the other {summary.n_cv_folds - 1}{' '}
            spatial CV folds.
          </p>
          <p>
            AOA weights (permutation importance):{' '}
            {weights.map(([v, w]) => `${v.toUpperCase()} ${w.toFixed(2)}`).join(' · ')}
          </p>
          {summary.exdet_combinatorial_area_km2 > 0 && (
            <p>
              Novel combinations (exDet NT2 &gt; 1): {formatCompact(summary.exdet_combinatorial_area_km2)} km²
              {summary.top_combinatorial_variables.length > 0 &&
                `, driven mainly by ${summary.top_combinatorial_variables
                  .map((v) => String(v.variable).toUpperCase())
                  .join(', ')}`}
              .
            </p>
          )}
        </div>
      </CardContent>
    </Card>
  )
}
