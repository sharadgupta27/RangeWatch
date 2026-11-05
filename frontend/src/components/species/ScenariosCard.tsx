import { CloudSun, Loader2 } from 'lucide-react'

import { useProjectScenarios } from '@/api/queries'
import type { LayerSet } from '@/api/types'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { mapActions } from '@/store/mapStore'

const pct = (v: number) => `${(v * 100).toFixed(1)}%`

/** CMIP6 "what-if" projections of the current model (no retraining), each with its own MESS. */
export function ScenariosCard({
  taxonKey,
  layers,
  jobActive,
}: {
  taxonKey: number
  layers: LayerSet
  jobActive: boolean
}) {
  const project = useProjectScenarios(taxonKey)
  const available = layers.scenarios_available ?? []
  const projected = new Map((layers.scenarios ?? []).map((s) => [s.name, s]))
  const missing = available.filter((n) => !projected.get(n)?.extrapolation)
  const ensembles = (layers.scenarios ?? []).filter((s) => s.gcm_ensemble)

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <CloudSun className="size-4" /> Climate scenarios
        </CardTitle>
        <CardDescription>
          The current model projected onto future climates without retraining. Each scenario has
          its own extrapolation mask — future climates usually leave the training range more often.
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-3">
        {available.length === 0 ? (
          <p className="text-xs text-muted-foreground">
            No scenarios are registered for this model's bioclim version. An operator can add one
            with <code className="font-mono">bioclim_store add-scenario</code>.
          </p>
        ) : (
          <ul className="space-y-1.5 text-xs">
            {available.map((name) => {
              const sc = projected.get(name)
              return (
                <li key={name} className="flex items-center justify-between gap-2">
                  <button
                    type="button"
                    className="truncate text-left font-mono hover:text-primary disabled:hover:text-inherit"
                    disabled={!sc}
                    onClick={() => sc && mapActions.setProjection(sc.id as `scenario:${string}`)}
                  >
                    {name}
                  </button>
                  {sc?.extrapolation ? (
                    <Badge variant={(sc.extrapolated_land_fraction ?? 0) > 0.25 ? 'warning' : 'secondary'}>
                      {((sc.extrapolated_land_fraction ?? 0) * 100).toFixed(0)}% of land extrapolated
                    </Badge>
                  ) : (
                    <Badge variant="outline">not projected</Badge>
                  )}
                </li>
              )
            })}
          </ul>
        )}
        {ensembles.length > 0 && (
          <div className="space-y-1.5 border-t pt-3 text-xs">
            <div className="font-medium">Multi-GCM ensembles</div>
            {ensembles.map((s) => {
              const e = s.gcm_ensemble!
              return (
                <div key={s.id} className="space-y-0.5">
                  <button
                    type="button"
                    className="text-left font-mono hover:text-primary"
                    onClick={() => mapActions.setProjection(s.id as `scenario:${string}`)}
                  >
                    {s.label}
                  </button>
                  <div className="text-muted-foreground">
                    Suitable under all GCMs {pct(e.suitable_all_gcms_land_fraction)} of land · some
                    but not all {pct(e.gcm_disagreement_land_fraction)} · extrapolated under any GCM{' '}
                    {pct(e.extrapolated_any_gcm_land_fraction)}
                  </div>
                </div>
              )
            })}
          </div>
        )}
        {missing.length > 0 && (
          <Button size="sm" onClick={() => project.mutate()} disabled={project.isPending || jobActive}>
            {project.isPending || jobActive ? <Loader2 className="animate-spin" /> : <CloudSun />}
            Project {missing.length} scenario{missing.length === 1 ? '' : 's'}
          </Button>
        )}
        {project.isError && <p className="text-xs text-destructive">{project.error.message}</p>}
      </CardContent>
    </Card>
  )
}
