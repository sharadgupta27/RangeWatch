import type { PickingInfo } from '@deck.gl/core'
import { useSelector } from '@tanstack/react-store'
import { AlertTriangle, Layers, Map as MapIcon } from 'lucide-react'
import { useMemo } from 'react'

import type { LayerSet, NativeRangeOut, RasterLayer, Timeline } from '@/api/types'
import { Badge } from '@/components/ui/badge'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Switch } from '@/components/ui/switch'
import { ToggleGroup, ToggleGroupItem } from '@/components/ui/toggle-group'
import { Tooltip, TooltipContent, TooltipTrigger } from '@/components/ui/tooltip'
import { resolutionLabel } from '@/lib/utils'
import {
  DIAGNOSTIC_LAYER_IDS,
  isDiagnosticLayer,
  mapActions,
  mapStore,
  type DiagnosticLayerId,
  type OverlaySource,
  type ProjectionLayerId,
} from '@/store/mapStore'

import { DeckMap } from './DeckMap'
import {
  effortLayer,
  hiresLayers,
  nativeRangeLayer,
  occurrenceLayer,
  projectionLayers,
  referenceLayer,
} from './layers'
import { OccurrenceLegend, RasterLegend } from './Legend'
import { TimeSlider } from './TimeSlider'

interface Props {
  layers: LayerSet
  nativeRange: NativeRangeOut | null | undefined
  timeline: Timeline | undefined
}

function occurrenceTooltip(info: PickingInfo): string | null {
  const p = (info.object as { properties?: Record<string, unknown> } | undefined)?.properties
  if (!p) return null
  if (info.layer?.id?.startsWith('reference-records')) {
    if (p.record_id !== undefined) {
      const status = Number(p.n_absent) > 0 ? 'reported absent' : 'present'
      return `Independent record #${p.record_id}
${status} · ${p.dataset_id}
(evaluation only)`
    }
    return `${p.n} independent records
${p.n_present ?? 0} present · ${p.n_absent ?? 0} absent`
  }
  if (p.record_id !== undefined) {
    return `${p.source} #${p.record_id}\n${p.range_label}${p.year ? ` · ${p.year}` : ''}`
  }
  return `${p.n} records\n${p.n_native ?? 0} native · ${p.n_introduced ?? 0} introduced`
}

/** MESS plus the advanced diagnostics, in the order offered by the map's diagnostic picker. */
export function diagnosticLayers(layers: LayerSet): RasterLayer[] {
  const all = [...(layers.mess ? [layers.mess] : []), ...(layers.diagnostics ?? [])]
  return DIAGNOSTIC_LAYER_IDS.flatMap((id) => all.filter((l) => l.id === id))
}

/** Further map products (binary maps, limiting factor, agreement, change, survey priority),
 * plus each scenario's present → scenario change map and each GCM ensemble's agreement map. */
export function productLayers(layers: LayerSet): RasterLayer[] {
  return [
    ...(layers.products ?? []),
    ...(layers.scenarios ?? []).flatMap((s) => [
      ...(s.change ? [s.change] : []),
      ...(s.gcm_ensemble?.agreement ? [s.gcm_ensemble.agreement] : []),
    ]),
  ]
}

/** The scenario a scenario-derived product (change:<name>, gcm-agreement:<name>) belongs to. */
function productScenario(layers: LayerSet, pid: string) {
  const name = pid.startsWith('change:')
    ? pid.slice('change:'.length)
    : pid.startsWith('gcm-agreement:')
      ? pid.slice('gcm-agreement:'.length)
      : null
  return name == null ? undefined : (layers.scenarios?.find((s) => s.id === `scenario:${name}`) ?? null)
}

/**
 * The raster shown for a projection choice, plus the extrapolation overlay that belongs to it:
 * scenarios carry their own MESS mask and consensus (future climates extrapolate more than the
 * present). The consensus overlay is used when chosen and available, else the MESS mask.
 */
export function resolveProjection(
  layers: LayerSet,
  id: ProjectionLayerId,
  overlaySource: OverlaySource = 'mess',
): { base: RasterLayer | null; extrapolation: RasterLayer | null } {
  const consensus = overlaySource === 'consensus'
  if (id.startsWith('product:')) {
    const pid = id.slice('product:'.length)
    const sc = productScenario(layers, pid)
    if (sc !== undefined) {
      // A scenario's products carry that scenario's own extrapolation mask (for a GCM
      // ensemble: MESS < 0 under any GCM).
      const base = pid.startsWith('change:') ? sc?.change : sc?.gcm_ensemble?.agreement
      return { base: base ?? null, extrapolation: (consensus && sc?.consensus) || sc?.extrapolation || null }
    }
    const current = (consensus && layers.consensus_overlay) || layers.extrapolation || null
    return { base: productLayers(layers).find((l) => l.id === pid) ?? null, extrapolation: current }
  }
  if (id.startsWith('scenario:')) {
    const sc = layers.scenarios?.find((s) => s.id === id)
    return {
      base: sc?.suitability ?? null,
      extrapolation: (consensus && sc?.consensus) || sc?.extrapolation || null,
    }
  }
  if (isDiagnosticLayer(id)) {
    return { base: diagnosticLayers(layers).find((l) => l.id === id) ?? null, extrapolation: null }
  }
  const current = (consensus && layers.consensus_overlay) || layers.extrapolation || null
  if (id === 'zones') return { base: layers.zones ?? null, extrapolation: current }
  return { base: layers.suitability ?? null, extrapolation: current }
}

export function DualMapView({ layers, nativeRange, timeline }: Props) {
  const viewState = useSelector(mapStore, (s) => s.viewState)
  const projection = useSelector(mapStore, (s) => s.projection)
  const messOverlay = useSelector(mapStore, (s) => s.messOverlay)
  const overlaySource = useSelector(mapStore, (s) => s.overlaySource)
  const maxYear = useSelector(mapStore, (s) => s.maxYear)
  const showNative = useSelector(mapStore, (s) => s.showNativeRange)
  const hiresOverlay = useSelector(mapStore, (s) => s.hiresOverlay)
  const showEffort = useSelector(mapStore, (s) => s.showEffort)
  const showReference = useSelector(mapStore, (s) => s.showReference)

  // Separate instances per map — deck.gl layers cannot be shared between Deck instances.
  const makeNative = useMemo(
    () => (id: string) =>
      nativeRange
        ? [
            nativeRangeLayer({
              id,
              geometry: nativeRange.geometry as GeoJSON.MultiPolygon,
              status: nativeRange.status,
              visible: showNative,
            }),
          ]
        : [],
    [nativeRange, showNative],
  )

  const leftLayers = useMemo(
    () => [
      ...(layers.effort && showEffort ? [effortLayer({ layer: layers.effort })] : []),
      ...makeNative('native-range-left'),
      occurrenceLayer({ layer: layers.occurrences, maxYear }),
      ...(layers.reference && showReference ? [referenceLayer({ layer: layers.reference })] : []),
    ],
    [layers.occurrences, layers.effort, layers.reference, maxYear, makeNative, showEffort, showReference],
  )

  const { base, extrapolation } = resolveProjection(layers, projection, overlaySource)
  const isScenario = projection.startsWith('scenario:')
  const isProduct = projection.startsWith('product:')
  const products = productLayers(layers)
  const isDiagnostic = isDiagnosticLayer(projection)
  const diagnostics = diagnosticLayers(layers)
  const hasConsensusOverlay = isScenario
    ? !!layers.scenarios?.find((s) => s.id === projection)?.consensus
    : !!layers.consensus_overlay
  // The high-resolution region refines current-climate suitability only.
  const hires = projection === 'suitability' ? layers.hires : null
  const rightLayers = useMemo(
    () => [
      ...projectionLayers({ base, extrapolation }, messOverlay),
      ...(hiresOverlay ? hiresLayers(hires, messOverlay, overlaySource) : []),
      ...makeNative('native-range-right'),
    ],
    [base, extrapolation, messOverlay, overlaySource, hires, hiresOverlay, makeNative],
  )

  const hasModel = !!layers.suitability
  const caveats = layers.caveats
  // "Extrapolation" opens the consensus when the advanced diagnostics exist, else MESS.
  const defaultDiagnostic: DiagnosticLayerId =
    diagnostics.find((l) => l.id === 'consensus') ? 'consensus' : 'mess'

  return (
    <div className="grid h-[560px] grid-cols-1 gap-3 lg:grid-cols-2">
      {/* ---------------- Left: observed occurrences ---------------- */}
      <div className="relative overflow-hidden rounded-xl border">
        <DeckMap
          layers={leftLayers}
          viewState={viewState}
          onViewStateChange={mapActions.setViewState}
          getTooltip={occurrenceTooltip}
        >
          <div className="absolute top-3 right-3 left-3 flex flex-wrap items-center gap-2">
            <Badge variant="secondary" className="bg-card/85 backdrop-blur">
              <MapIcon /> Observed occurrences
            </Badge>
            {nativeRange && (
              <label className="flex items-center gap-2 rounded-full border bg-card/85 px-2.5 py-0.5 text-xs backdrop-blur">
                <Switch
                  checked={showNative}
                  onCheckedChange={mapActions.setShowNativeRange}
                  aria-label="Show native range"
                />
                Native range
                {nativeRange.status === 'draft' && (
                  <span className="text-warning">(draft)</span>
                )}
              </label>
            )}
            {layers.effort && (
              <Tooltip>
                <TooltipTrigger asChild>
                  <label className="flex items-center gap-2 rounded-full border bg-card/85 px-2.5 py-0.5 text-xs backdrop-blur">
                    <Switch
                      checked={showEffort}
                      onCheckedChange={mapActions.setShowEffort}
                      aria-label="Show sampling effort"
                    />
                    Effort
                  </label>
                </TooltipTrigger>
                <TooltipContent>{layers.effort.description}</TooltipContent>
              </Tooltip>
            )}
            {layers.reference && (
              <Tooltip>
                <TooltipTrigger asChild>
                  <label className="flex items-center gap-2 rounded-full border bg-card/85 px-2.5 py-0.5 text-xs backdrop-blur">
                    <Switch
                      checked={showReference}
                      onCheckedChange={mapActions.setShowReference}
                      aria-label="Show independent records"
                    />
                    Independent
                  </label>
                </TooltipTrigger>
                <TooltipContent>
                  Uploaded surveys and GBIF-reported absences. They evaluate the model and are
                  never used to train it.
                </TooltipContent>
              </Tooltip>
            )}
          </div>
          <div className="absolute bottom-3 left-3 flex flex-col gap-2">
            {layers.effort && showEffort && <RasterLegend layer={layers.effort} />}
            <OccurrenceLegend showReference={!!layers.reference && showReference} />
          </div>
          {timeline?.min_year != null && timeline.max_year != null && (
            <div className="absolute top-12 left-3">
              <TimeSlider minYear={timeline.min_year} maxYear={timeline.max_year} />
            </div>
          )}
        </DeckMap>
      </div>

      {/* ---------------- Right: projection ---------------- */}
      <div className="relative overflow-hidden rounded-xl border">
        <DeckMap
          layers={rightLayers}
          viewState={viewState}
          onViewStateChange={mapActions.setViewState}
          onBoundsChange={mapActions.setViewBounds}
        >
          {hasModel ? (
            <>
              <div className="absolute top-3 left-3 flex flex-wrap items-center gap-2 pr-14">
                <ToggleGroup
                  type="single"
                  value={isScenario || isProduct ? '' : isDiagnostic ? 'extrapolation' : projection}
                  onValueChange={(v) => {
                    if (!v || (v === 'extrapolation' && isDiagnostic)) return
                    mapActions.setProjection(
                      v === 'extrapolation' ? defaultDiagnostic : (v as ProjectionLayerId),
                    )
                  }}
                  aria-label="Projection layer"
                >
                  <ToggleGroupItem value="suitability">
                    <Layers /> Suitability
                  </ToggleGroupItem>
                  <ToggleGroupItem value="zones">Range zones</ToggleGroupItem>
                  <ToggleGroupItem value="extrapolation">
                    {diagnostics.length > 1 ? 'Extrapolation' : 'MESS'}
                  </ToggleGroupItem>
                </ToggleGroup>
                {isDiagnostic && diagnostics.length > 1 && (
                  <Select
                    value={projection}
                    onValueChange={(v) => mapActions.setProjection(v as DiagnosticLayerId)}
                  >
                    <SelectTrigger className="w-48" aria-label="Extrapolation diagnostic">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      {diagnostics.map((l) => (
                        <SelectItem key={l.id} value={l.id}>
                          {l.label}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                )}
                {products.length > 0 && (
                  <Select
                    value={isProduct ? projection : 'none'}
                    onValueChange={(v) =>
                      mapActions.setProjection(v === 'none' ? 'suitability' : (v as ProjectionLayerId))
                    }
                  >
                    <SelectTrigger className="w-48" aria-label="Map product">
                      <SelectValue placeholder="More maps" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="none">More maps…</SelectItem>
                      {products.map((l) => (
                        <SelectItem key={l.id} value={`product:${l.id}`}>
                          {l.label}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                )}
                {layers.scenarios && layers.scenarios.length > 0 && (
                  <Select
                    value={projection.startsWith('scenario:') ? projection : 'current'}
                    onValueChange={(v) =>
                      mapActions.setProjection(v === 'current' ? 'suitability' : (v as ProjectionLayerId))
                    }
                  >
                    <SelectTrigger className="w-48" aria-label="Climate scenario">
                      <SelectValue placeholder="Climate scenario" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="current">Current climate</SelectItem>
                      {layers.scenarios.map((s) => (
                        <SelectItem key={s.id} value={s.id}>
                          {s.label}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                )}
                {!isDiagnostic && (
                  <div className="flex items-center gap-1.5 rounded-lg border bg-card/85 px-2.5 py-1 text-xs shadow-sm backdrop-blur">
                    <Tooltip>
                      <TooltipTrigger asChild>
                        <label className="flex items-center gap-2">
                          <Switch
                            checked={messOverlay}
                            onCheckedChange={mapActions.setMessOverlay}
                            aria-label="Extrapolation overlay"
                          />
                          {hasConsensusOverlay ? 'Overlay' : 'MESS overlay'}
                        </label>
                      </TooltipTrigger>
                      <TooltipContent>{(extrapolation ?? layers.extrapolation)?.description}</TooltipContent>
                    </Tooltip>
                    {hasConsensusOverlay && (
                      <Select
                        value={overlaySource}
                        onValueChange={(v) => mapActions.setOverlaySource(v as OverlaySource)}
                        disabled={!messOverlay}
                      >
                        <SelectTrigger className="h-6 w-28" aria-label="Overlay source">
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="mess">MESS &lt; 0</SelectItem>
                          <SelectItem value="consensus">Consensus</SelectItem>
                        </SelectContent>
                      </Select>
                    )}
                  </div>
                )}
                {hires && (
                  <Tooltip>
                    <TooltipTrigger asChild>
                      <label className="flex items-center gap-2 rounded-lg border bg-card/85 px-2.5 py-1 text-xs shadow-sm backdrop-blur">
                        <Switch
                          checked={hiresOverlay}
                          onCheckedChange={mapActions.setHiresOverlay}
                          aria-label="High-resolution detail"
                        />
                        {resolutionLabel(hires.resolution)} detail
                      </label>
                    </TooltipTrigger>
                    <TooltipContent>{hires.suitability.description}</TooltipContent>
                  </Tooltip>
                )}
              </div>
              <div className="absolute bottom-3 left-3 flex flex-col gap-2">
                {base && (
                  <RasterLegend
                    layer={base}
                    showDescription={(isDiagnostic && base.id !== 'mess') || isProduct}
                  />
                )}
                {messOverlay && extrapolation && <RasterLegend layer={extrapolation} />}
                {isScenario && !extrapolation && (
                  <div className="max-w-56 rounded-lg border border-warning/40 bg-card/90 px-3 py-2 text-[11px] text-warning">
                    This scenario projection predates per-scenario MESS; re-project it to see
                    where it extrapolates.
                  </div>
                )}
              </div>
              {caveats && (
                <Tooltip>
                  <TooltipTrigger asChild>
                    <Badge
                      variant={caveats.confidence_label === 'lower' ? 'warning' : 'secondary'}
                      className="absolute top-3 right-3 cursor-help bg-card/85 backdrop-blur"
                    >
                      <AlertTriangle />
                      Model {caveats.model_type}
                      {caveats.confidence_label === 'lower' ? ' · lower confidence' : ''}
                    </Badge>
                  </TooltipTrigger>
                  <TooltipContent side="left">{caveats.transferability_caveat}</TooltipContent>
                </Tooltip>
              )}
            </>
          ) : (
            <div className="absolute inset-0 flex items-center justify-center bg-background/40 backdrop-blur-[1px]">
              <div className="max-w-xs rounded-lg border bg-card/90 p-4 text-center text-sm shadow">
                <p className="font-medium">No projection yet</p>
                <p className="mt-1 text-xs text-muted-foreground">
                  Suitability, range zones and the MESS uncertainty layer appear here once a
                  model has been trained on a confirmed native range.
                </p>
              </div>
            </div>
          )}
        </DeckMap>
      </div>
    </div>
  )
}
