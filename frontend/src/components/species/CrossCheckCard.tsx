import { Loader2, Scale } from 'lucide-react'

import { useRunCrossCheck } from '@/api/queries'
import type { BioclimInfo, ClimateCrossCheckOut, ModelVersionDetail } from '@/api/types'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { formatRelative } from '@/lib/utils'

const VERDICT_VARIANT = {
  consistent: 'success',
  moderate: 'warning',
  divergent: 'destructive',
} as const

const fmt = (v: number | null | undefined, digits = 3) => (v == null ? '–' : v.toFixed(digits))

function CrossCheckResult({ c }: { c: ClimateCrossCheckOut }) {
  return (
    <div className="space-y-3">
      <div className="flex flex-wrap items-center gap-2">
        <Badge variant={VERDICT_VARIANT[c.verdict]}>{c.verdict}</Badge>
        <span className="font-mono text-[11px] text-muted-foreground">
          {c.primary_bioclim_version} vs {c.alt_bioclim_version} · {formatRelative(c.created_ts)}
        </span>
      </div>
      <p className="text-xs leading-relaxed">{c.interpretation}</p>

      <div className="grid gap-3 lg:grid-cols-2">
        <table className="w-full text-xs">
          <thead className="text-muted-foreground">
            <tr className="border-b text-left">
              <th className="py-1 font-medium">Spatial-CV metric</th>
              <th className="py-1 text-right font-medium">{c.primary_bioclim_version}</th>
              <th className="py-1 text-right font-medium">{c.alt_bioclim_version}</th>
            </tr>
          </thead>
          <tbody className="tabular-nums">
            {(['auc_mean', 'cbi_mean', 'tss_mean'] as const).map((k) => (
              <tr key={k} className="border-b last:border-0">
                <td className="py-1">{k.replace('_mean', '').toUpperCase()}</td>
                <td className="py-1 text-right">{fmt(c.primary[k])}</td>
                <td className="py-1 text-right">{fmt(c.alternative[k])}</td>
              </tr>
            ))}
            <tr className="border-t">
              <td className="py-1">Suitability rank correlation (ρ)</td>
              <td className="py-1 text-right" colSpan={2}>
                {fmt(c.suitability_rank_correlation)}
              </td>
            </tr>
            <tr>
              <td className="py-1">Suitable/unsuitable agreement (κ)</td>
              <td className="py-1 text-right" colSpan={2}>
                {fmt(c.classification_agreement)} ({fmt(c.classification_kappa)})
              </td>
            </tr>
          </tbody>
        </table>

        <table className="w-full text-xs">
          <thead className="text-muted-foreground">
            <tr className="border-b text-left">
              <th className="py-1 font-medium">Predictor</th>
              <th className="py-1 text-right font-medium">r</th>
              <th className="py-1 text-right font-medium">mean Δ</th>
              <th className="py-1 text-right font-medium">scale ratio</th>
            </tr>
          </thead>
          <tbody className="tabular-nums">
            {c.predictors.map((p) => (
              <tr key={p.predictor} className="border-b last:border-0">
                <td className="py-1 font-mono uppercase">
                  {p.predictor}
                  {p.units_suspect && (
                    <Badge variant="warning" className="ml-1.5">
                      units?
                    </Badge>
                  )}
                </td>
                <td className="py-1 text-right">{fmt(p.pearson_r)}</td>
                <td className="py-1 text-right">{fmt(p.mean_diff, 2)}</td>
                <td className="py-1 text-right">{fmt(p.scale_ratio, 2)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="text-[11px] text-muted-foreground">
        Same {c.n_presence.toLocaleString()} presences and {c.n_background.toLocaleString()}{' '}
        background points, feature classes {c.feature_classes}, β = {c.beta_multiplier}
        {c.n_dropped > 0 && ` (${c.n_dropped} points outside the alternative stack dropped)`}.
      </p>
    </div>
  )
}

/**
 * Climate-data cross-check: the current version refitted on an independent climate source
 * (e.g. CHELSA) with the exact same training set, compared with the original.
 */
export function CrossCheckCard({
  taxonKey,
  model,
  bioclim,
  jobActive,
}: {
  taxonKey: number
  model: ModelVersionDetail
  bioclim: BioclimInfo | undefined
  jobActive: boolean
}) {
  const run = useRunCrossCheck(taxonKey)
  const alt = bioclim?.crosscheck_version ?? null
  const results = model.crosschecks ?? []

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Scale className="size-4" /> Climate-data cross-check · v{model.model_version}
        </CardTitle>
        <CardDescription>
          Refits this version on an independent climate dataset with the same presences,
          background, feature classes, regularisation and CV folds. Low agreement means the
          projection depends on the climate data (often in mountains) — an uncertainty to report
          next to MESS, not a reason to switch datasets.
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        {results.length === 0 && (
          <p className="text-xs text-muted-foreground">Not cross-checked yet.</p>
        )}
        {results.map((c) => (
          <CrossCheckResult key={c.alt_bioclim_version} c={c} />
        ))}
        {alt ? (
          <div className="space-y-1">
            <Button size="sm" variant="outline" onClick={() => run.mutate()} disabled={jobActive || run.isPending}>
              {run.isPending || jobActive ? <Loader2 className="animate-spin" /> : <Scale />}
              {results.some((c) => c.alt_bioclim_version === alt) ? 'Re-run' : 'Run'} cross-check
              against {alt}
            </Button>
            {bioclim?.crosscheck_source && (
              <p className="text-[11px] text-muted-foreground">{bioclim.crosscheck_source}</p>
            )}
          </div>
        ) : (
          <p className="text-xs text-muted-foreground">
            No cross-check climate stack is installed. An operator can build one with{' '}
            <code className="font-mono">bioclim_store build-chelsa</code> and set{' '}
            <code className="font-mono">SDM_CROSSCHECK_BIOCLIM_VERSION</code>.
          </p>
        )}
        {run.isError && <p className="text-xs text-destructive">{run.error.message}</p>}
      </CardContent>
    </Card>
  )
}
