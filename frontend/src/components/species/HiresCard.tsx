import { useSelector } from '@tanstack/react-store'
import { Loader2, ScanSearch } from 'lucide-react'

import { useProjectHires } from '@/api/queries'
import type { LayerSet } from '@/api/types'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { formatRelative, resolutionLabel } from '@/lib/utils'
import { mapActions, mapStore, type Bounds } from '@/store/mapStore'

/** Cell size of the WorldClim resolution codes (for the size estimate shown before submitting). */
const RESOLUTION_DEG: Record<string, number> = { '10m': 1 / 6, '5m': 1 / 12, '2.5m': 1 / 24, '30s': 1 / 120 }

export function estimateCells(bounds: Bounds, resolution: string): number | null {
  const cell = RESOLUTION_DEG[resolution]
  if (!cell) return null
  const [west, south, east, north] = bounds
  return Math.round((east - west) / cell) * Math.round((north - south) / cell)
}

function formatBounds([west, south, east, north]: number[]): string {
  const lon = (v: number) => `${Math.abs(v).toFixed(1)}°${v < 0 ? 'W' : 'E'}`
  const lat = (v: number) => `${Math.abs(v).toFixed(1)}°${v < 0 ? 'S' : 'N'}`
  return `${lon(west!)}–${lon(east!)}, ${lat(south!)}–${lat(north!)}`
}

function fitView([west, south, east, north]: number[]) {
  const span = Math.max(east! - west!, (north! - south!) * 1.6, 0.1)
  mapActions.setProjection('suitability')
  mapActions.setHiresOverlay(true)
  mapActions.setViewState({
    longitude: (west! + east!) / 2,
    latitude: (south! + north!) / 2,
    zoom: Math.max(0.8, Math.min(9, Math.log2(360 / span) - 0.6)),
    pitch: 0,
    bearing: 0,
  })
}

/** Regional high-resolution projection of the current model (no retraining, own MESS). */
export function HiresCard({
  taxonKey,
  layers,
  jobActive,
}: {
  taxonKey: number
  layers: LayerSet
  jobActive: boolean
}) {
  const project = useProjectHires(taxonKey)
  const bounds = useSelector(mapStore, (s) => s.viewBounds)
  const resolution = layers.hires_available
  const hires = layers.hires
  const cells = bounds && resolution ? estimateCells(bounds, resolution) : null
  const maxCells = layers.hires_max_cells ?? null
  const tooBig = cells !== null && maxCells !== null && cells > maxCells
  const label = resolution ? resolutionLabel(resolution) : null

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <ScanSearch className="size-4" /> High-resolution detail
        </CardTitle>
        <CardDescription>
          The current model projected on a finer climate grid inside the area shown on the
          projection map — no retraining. The region gets its own extrapolation (MESS) mask.
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-3 text-xs">
        {!resolution ? (
          <p className="text-muted-foreground">
            No high-resolution climate stack is installed for this model's bioclim version. An
            operator can add one with <code className="font-mono">bioclim_store add-hires</code>.
          </p>
        ) : (
          <>
            {hires && (
              <div className="flex flex-wrap items-center justify-between gap-2">
                <button
                  type="button"
                  className="text-left hover:text-primary"
                  onClick={() => fitView(hires.bbox)}
                >
                  <span className="font-medium">{resolutionLabel(hires.resolution)} region</span>{' '}
                  <span className="font-mono">{formatBounds(hires.bbox)}</span>
                  <span className="block text-muted-foreground">
                    projected {formatRelative(hires.created_ts)}
                  </span>
                </button>
                <Badge
                  variant={(hires.extrapolated_land_fraction ?? 0) > 0.25 ? 'warning' : 'secondary'}
                >
                  {((hires.extrapolated_land_fraction ?? 0) * 100).toFixed(0)}% of land extrapolated
                </Badge>
              </div>
            )}
            <Button
              size="sm"
              onClick={() => bounds && project.mutate(bounds)}
              disabled={!bounds || tooBig || jobActive || project.isPending}
            >
              {project.isPending || jobActive ? <Loader2 className="animate-spin" /> : <ScanSearch />}
              {hires ? `Re-project ${label} for current view` : `Project ${label} for current view`}
            </Button>
            {bounds && cells !== null && (
              <p className={tooBig ? 'text-warning' : 'text-muted-foreground'}>
                {tooBig
                  ? `The view covers ~${cells.toLocaleString()} cells at ${label}; zoom in to at most ${maxCells?.toLocaleString()}.`
                  : `~${cells.toLocaleString()} cells (${formatBounds(bounds)}). Replaces the previous region.`}
              </p>
            )}
          </>
        )}
        {project.isError && <p className="text-destructive">{project.error.message}</p>}
      </CardContent>
    </Card>
  )
}
