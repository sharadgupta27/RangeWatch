/**
 * Composable deck.gl layer-config functions (CLAUDE.md: never inline layers in pages), reused
 * by the dual-map view, the native-range editor and future bulletin previews.
 *
 * Occurrences arrive as pg_tileserv vector tiles that are already aggregated server-side at
 * low zoom (count per grid cell), so large point sets are never rendered unaggregated.
 */
import type { Layer } from '@deck.gl/core'
import { MVTLayer, TileLayer } from '@deck.gl/geo-layers'
import { BitmapLayer, GeoJsonLayer } from '@deck.gl/layers'
import type { Feature, MultiPolygon, Polygon } from 'geojson'

import type { RasterLayer, VectorLayer } from '@/api/types'

export type RGBA = [number, number, number, number]

export const COLORS = {
  native: [42, 157, 143, 230] as RGBA,
  introduced: [244, 132, 95, 235] as RGBA,
  unknown: [148, 163, 184, 210] as RGBA,
  nativeRangeFill: [42, 157, 143, 28] as RGBA,
  nativeRangeLine: [94, 234, 212, 255] as RGBA,
}

/** Occurrence tile URL with the time-slider bound appended as a pg_tileserv function arg. */
export function occurrenceTileUrl(layer: VectorLayer, maxYear: number | null): string {
  return maxYear === null ? layer.tile_url : `${layer.tile_url}&max_year=${maxYear}`
}

interface OccurrenceProps {
  n?: number
  n_native?: number
  n_introduced?: number
  range_label?: 'native' | 'introduced' | 'unknown'
}

/** Colour of an (aggregated) occurrence feature: blend by the introduced share. */
export function occurrenceColor(p: OccurrenceProps): RGBA {
  if (p.range_label) return COLORS[p.range_label]
  const n = p.n ?? 1
  const share = n > 0 ? (p.n_introduced ?? 0) / n : 0
  const nativeShare = n > 0 ? (p.n_native ?? 0) / n : 0
  if (share === 0 && nativeShare === 0) return COLORS.unknown
  const mix = share / Math.max(share + nativeShare, 1e-9)
  const a = COLORS.native
  const b = COLORS.introduced
  return [0, 1, 2, 3].map((i) => Math.round(a[i]! * (1 - mix) + b[i]! * mix)) as RGBA
}

/** Pixel radius: sqrt-scaled by the aggregated count so dense cells read as larger. */
export function occurrenceRadius(p: OccurrenceProps): number {
  const n = Math.max(1, p.n ?? 1)
  return Math.min(18, 2.5 + Math.sqrt(n) * 1.1)
}

export function occurrenceLayer(opts: {
  layer: VectorLayer
  maxYear: number | null
  visible?: boolean
}): Layer {
  const url = occurrenceTileUrl(opts.layer, opts.maxYear)
  return new MVTLayer<OccurrenceProps>({
    id: 'occurrences',
    data: url,
    visible: opts.visible ?? true,
    minZoom: 0,
    maxZoom: 14,
    pointType: 'circle',
    pointRadiusUnits: 'pixels',
    getPointRadius: (f: Feature) => occurrenceRadius((f.properties ?? {}) as OccurrenceProps),
    getFillColor: (f: Feature) => occurrenceColor((f.properties ?? {}) as OccurrenceProps),
    getLineColor: [15, 23, 42, 160],
    lineWidthUnits: 'pixels',
    getLineWidth: 0.5,
    stroked: true,
    pickable: true,
    updateTriggers: { getFillColor: url, getPointRadius: url },
  })
}

/** Raster XYZ tiles from titiler rendered through a BitmapLayer. */
export function rasterTileLayer(opts: {
  id: string
  layer: RasterLayer
  opacity?: number
  visible?: boolean
}): Layer {
  return new TileLayer({
    id: opts.id,
    data: opts.layer.tile_url,
    visible: opts.visible ?? true,
    opacity: opts.opacity ?? 0.85,
    minZoom: 0,
    maxZoom: 10,
    tileSize: 256,
    maxRequests: 12,
    renderSubLayers: (props) => {
      const [[west, south], [east, north]] = props.tile.boundingBox as [
        [number, number],
        [number, number],
      ]
      return new BitmapLayer({
        ...props,
        id: `${props.id}-bitmap`,
        data: undefined,
        image: props.data,
        bounds: [west, south, east, north],
      })
    },
  })
}

export function nativeRangeLayer(opts: {
  geometry: Polygon | MultiPolygon
  status: 'draft' | 'confirmed'
  visible?: boolean
  /** Unique per map: a layer instance cannot be shared between Deck instances. */
  id?: string
}): Layer {
  return new GeoJsonLayer({
    id: opts.id ?? 'native-range',
    data: { type: 'Feature', geometry: opts.geometry, properties: {} } as Feature,
    visible: opts.visible ?? true,
    stroked: true,
    filled: true,
    getFillColor: COLORS.nativeRangeFill,
    getLineColor: COLORS.nativeRangeLine,
    getLineWidth: 2,
    lineWidthUnits: 'pixels',
    // Dashed look for unconfirmed drafts is conveyed via reduced opacity.
    opacity: opts.status === 'confirmed' ? 1 : 0.6,
  })
}

export interface ProjectionLayerConfig {
  base: RasterLayer | null
  extrapolation: RasterLayer | null
}

/**
 * The projection stack for the right-hand map: the chosen projection plus — whenever that
 * projection is a suitability/zones view — the MESS extrapolation overlay when enabled.
 */
export function projectionLayers(cfg: ProjectionLayerConfig, messOverlay: boolean): Layer[] {
  const out: Layer[] = []
  if (cfg.base) out.push(rasterTileLayer({ id: `projection-${cfg.base.id}`, layer: cfg.base }))
  const showsSuitability = cfg.base !== null && cfg.base.id !== 'mess'
  if (cfg.extrapolation && messOverlay && showsSuitability) {
    out.push(
      rasterTileLayer({ id: 'extrapolation-overlay', layer: cfg.extrapolation, opacity: 0.75 }),
    )
  }
  return out
}
