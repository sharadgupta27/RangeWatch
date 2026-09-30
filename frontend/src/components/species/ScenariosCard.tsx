import { CloudSun, Loader2 } from 'lucide-react'

import { useProjectScenarios } from '@/api/queries'
import type { LayerSet } from '@/api/types'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { mapActions } from '@/store/mapStore'

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
