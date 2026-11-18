import { ListOrdered, Loader2 } from 'lucide-react'

import { useComparePredictorSets, usePermissions, useUpdateEnvironment } from '@/api/queries'
import type { ModelVersionDetail, PredictorSetResultOut, SpeciesDetail } from '@/api/types'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { formatDate } from '@/lib/utils'

const fmt = (v: number | null | undefined, d = 3) => (v == null ? '–' : v.toFixed(d))

function metric(mean: number | null | undefined, sd: number | null | undefined) {
  return mean == null ? '–' : `${mean.toFixed(3)}${sd != null ? ` ± ${sd.toFixed(3)}` : ''}`
}

/**
 * Which variables explain this species best? Candidate sets (temperature, precipitation,
 * humidity/wind, …) refitted on the current version's exact training data and ranked by
 * spatial-CV CBI. "Use this set" stores it as the species' predictor choice (flags a retrain).
 */
export function PredictorComparisonCard({
  sp,
  model,
  jobActive,
}: {
  sp: SpeciesDetail
  model: ModelVersionDetail
  jobActive: boolean
}) {
  const compare = useComparePredictorSets(sp.taxon_key)
  const update = useUpdateEnvironment(sp.taxon_key)
  const { run } = usePermissions()
  const c = model.predictor_comparison
  const fitted = (c?.sets ?? []).filter((s) => s.status === 'fitted')
  const skipped = (c?.sets ?? []).filter((s) => s.status === 'skipped')
  const similar = new Set(c?.similar_to_best ?? [])
  const chosen =
    sp.predictor_config.mode === 'custom' ? (sp.predictor_config.predictors ?? []) : null
  const used = (s: PredictorSetResultOut) => s.predictors ?? []
  const isChosen = (s: PredictorSetResultOut) =>
    !!chosen &&
    chosen.length === used(s).length &&
    used(s).every((p) => chosen.includes(p)) &&
    (sp.climate_version ?? sp.effective_climate_version) === c?.bioclim_version

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <ListOrdered className="size-4" /> Predictor-set comparison
        </CardTitle>
        <CardDescription>
          Refits v{model.model_version}&apos;s exact presences and background with alternative
          predictor sets — temperature, precipitation, humidity/wind, water balance, soil and
          combinations, each VIF-screened — and ranks them by spatial block CV (CBI, then AUC).
          A relative comparison on the same data: confirm a choice on independent field records.
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-3 text-xs">
        {run && (
          <Button
            size="sm"
            variant="outline"
            disabled={compare.isPending || jobActive}
            onClick={() => compare.mutate(null)}
          >
            {compare.isPending && <Loader2 className="animate-spin" />}
            {c ? 'Run again' : 'Compare predictor sets'}
          </Button>
        )}
        {!c ? (
          <p className="text-muted-foreground">Not run for this version yet.</p>
        ) : (
          <>
            <p className="text-muted-foreground">
              {c.bioclim_version} ({c.climate_family}, {c.resolution}) ·{' '}
              {formatDate(c.created_ts)} · ranked by {c.ranking_metric}
            </p>
            <div className="overflow-x-auto">
              <table className="w-full text-xs">
                <thead className="text-muted-foreground">
                  <tr className="border-b text-left">
                    <th className="py-1 pr-2 font-medium">#</th>
                    <th className="py-1 pr-2 font-medium">Set</th>
                    <th className="py-1 pr-2 text-right font-medium">CBI</th>
                    <th className="py-1 pr-2 text-right font-medium">AUC</th>
                    <th className="py-1 pr-2 text-right font-medium">TSS</th>
                    <th className="py-1 font-medium" />
                  </tr>
                </thead>
                <tbody className="tabular-nums">
                  {fitted.map((s) => (
                    <tr key={s.name} className="border-b align-top last:border-0">
                      <td className="py-1.5 pr-2">{s.rank}</td>
                      <td className="py-1.5 pr-2">
                        <div className="flex flex-wrap items-center gap-1">
                          <span className="font-medium">{s.label}</span>
                          {s.name === c.best && <Badge>best</Badge>}
                          {similar.has(s.name) && <Badge variant="secondary">≈ best</Badge>}
                          {(s.present_only ?? []).length > 0 && (
                            <Badge variant="warning">present only</Badge>
                          )}
                          {isChosen(s) && <Badge variant="secondary">chosen</Badge>}
                        </div>
                        <div className="font-mono text-[11px] text-muted-foreground">
                          {used(s).join(', ')}
                          {(s.dropped_by_vif ?? []).length > 0 && (
                            <span> (VIF dropped {(s.dropped_by_vif ?? []).join(', ')})</span>
                          )}
                        </div>
                      </td>
                      <td className="py-1.5 pr-2 text-right">{metric(s.cbi_mean, s.cbi_std)}</td>
                      <td className="py-1.5 pr-2 text-right">{fmt(s.auc_mean)}</td>
                      <td className="py-1.5 pr-2 text-right">{fmt(s.tss_mean)}</td>
                      <td className="py-1.5 text-right">
                        {run && !isChosen(s) && (
                          <Button
                            size="sm"
                            variant="ghost"
                            className="h-6 px-2 text-[11px]"
                            disabled={update.isPending}
                            onClick={() =>
                              update.mutate({
                                climate_version: c.bioclim_version,
                                predictor_config: {
                                  mode: 'custom',
                                  predictors: used(s),
                                  vif_screen: false,
                                  protected: [],
                                },
                              })
                            }
                          >
                            Use this set
                          </Button>
                        )}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            {skipped.length > 0 && (
              <p className="text-muted-foreground">
                Skipped: {skipped.map((s) => `${s.label} (${s.reason})`).join('; ')}
              </p>
            )}
            <p className="text-[11px] text-muted-foreground">{c.note}</p>
          </>
        )}
      </CardContent>
    </Card>
  )
}
