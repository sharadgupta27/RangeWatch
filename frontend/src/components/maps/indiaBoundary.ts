// India's official boundary (Survey of India) on the CARTO vector basemap, using the correction
// data of india_boundary_corrector (https://github.com/ramSeraph/india_boundary_corrector).
//
// The data has paired line layers: `to-del-*` are the boundary segments OpenStreetMap / Natural
// Earth draw differently from India's official boundary, `to-add-*` is India's boundary there.
// Deletions are painted over in the land colour, then the additions are drawn in the basemap's
// own country/state boundary style. The data file ships with the app (same origin, ~1.6 MB,
// read by HTTP range requests for the tiles in view).
import correctionsUrl from '@india-boundary-corrector/data/pmtiles?url'
import { addProtocol, type ExpressionSpecification, type LayerSpecification } from 'maplibre-gl'
import { Protocol } from 'pmtiles'

import type { Theme } from '@/lib/theme'

export const INDIA_BOUNDARY_SOURCE = 'india-boundary'
export const INDIA_BOUNDARY_ATTRIBUTION =
  'India boundary: <a href="https://github.com/ramSeraph/india_boundary_corrector">india_boundary_corrector</a>'

let registered = false

/** Register MapLibre's `pmtiles://` protocol (once). */
export function registerPmtilesProtocol() {
  if (registered) return
  registered = true
  addProtocol('pmtiles', new Protocol().tile)
}

/** `pmtiles://` URL of the bundled correction data (absolute, as MapLibre's worker needs). */
export function indiaBoundaryUrl(base: string = globalThis.location?.href ?? 'http://localhost/'): string {
  return `pmtiles://${new URL(correctionsUrl, base).href}`
}

/** Colours of the CARTO no-label styles (dark-matter / positron) the corrections must match. */
const PALETTE = {
  dark: {
    land: '#0e0e0e',
    halo: '#2C353C',
    country: ['rgba(92, 94, 94, 1)', 'rgba(96, 96, 96, 1)', 'rgba(102, 102, 102, 1)'],
    state: 'rgba(103, 103, 114, 1)',
  },
  light: {
    land: '#fafaf8',
    halo: '#f3efed',
    country: ['#f2e6e7', '#ebd6d8', '#ebd6d8'],
    state: '#e1c5c7',
  },
} as const

/** CARTO draws Natural Earth-like boundaries at low zoom and OSM ones from z5 (as the corrector's own CARTO configs). */
const OSM_FROM_ZOOM = 5

const zoomWidth = (stops: [number, number][]): ExpressionSpecification => [
  'interpolate',
  ['linear'],
  ['zoom'],
  ...stops.flat(),
]

/**
 * MapLibre layers (in drawing order) that replace the basemap's boundary with India's
 * official one. Deletions use both the Natural Earth and OSM geometries at every zoom, since
 * the two differ slightly and either may be what the basemap drew.
 */
export function indiaBoundaryLayers(theme: Theme): LayerSpecification[] {
  const c = PALETTE[theme]
  const line = (
    id: string,
    sourceLayer: string,
    paint: Record<string, unknown>,
    zoom: { minzoom?: number; maxzoom?: number } = {},
  ): LayerSpecification =>
    ({
      id: `india-boundary-${id}`,
      type: 'line',
      source: INDIA_BOUNDARY_SOURCE,
      'source-layer': sourceLayer,
      layout: { 'line-cap': 'round', 'line-join': 'round' },
      paint,
      ...zoom,
    }) as LayerSpecification

  // Wide enough to cover the country line, and from z6 its 8 px halo.
  const delCountry = { 'line-color': c.land, 'line-width': zoomWidth([[0, 3], [5, 4], [5.99, 4], [6, 10]]) }
  const delState = { 'line-color': c.land, 'line-width': zoomWidth([[4, 2], [9, 3.5]]) }
  const halo = { 'line-color': c.halo, 'line-opacity': 0.5, 'line-width': 8 }
  const country = {
    'line-color': ['interpolate', ['linear'], ['zoom'], 4, c.country[0], 5, c.country[1], 6, c.country[2]],
    'line-width': zoomWidth([[3, 1], [6, 1.5]]),
  }
  const state = {
    'line-color': c.state,
    'line-width': zoomWidth([[4, 0.5], [7, 1], [9, 1.2]]),
    'line-dasharray': theme === 'dark' ? [1, 2, 3] : [2, 2],
  }

  const layers: LayerSpecification[] = []
  for (const src of ['ne', 'osm'] as const) {
    layers.push(line(`del-${src}`, `to-del-${src}`, delCountry))
    layers.push(line(`del-${src}-disp`, `to-del-${src}-disp`, delCountry))
    layers.push(line(`del-${src}-internal`, `to-del-${src}-internal`, delState, { minzoom: 4 }))
  }
  for (const src of ['ne', 'osm'] as const) {
    const zoom = src === 'ne' ? { maxzoom: OSM_FROM_ZOOM } : { minzoom: OSM_FROM_ZOOM }
    layers.push(line(`add-${src}-internal`, `to-add-${src}-internal`, state, { ...zoom, minzoom: Math.max(4, zoom.minzoom ?? 0) }))
    if (src === 'osm') {
      layers.push(line('add-osm-halo', 'to-add-osm', halo, { minzoom: 6 }))
      layers.push(line('add-osm-disp-halo', 'to-add-osm-disp', halo, { minzoom: 6 }))
    }
    layers.push(line(`add-${src}`, `to-add-${src}`, country, zoom))
    layers.push(line(`add-${src}-disp`, `to-add-${src}-disp`, country, zoom))
  }
  return layers
}
