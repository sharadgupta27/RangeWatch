import { describe, expect, it } from 'vitest'

import type { LayerSet, RasterLayer, VectorLayer } from '@/api/types'

import { resolveProjection } from './DualMapView'
import {
  COLORS,
  occurrenceColor,
  occurrenceLayer,
  occurrenceRadius,
  occurrenceTileUrl,
  projectionLayers,
} from './layers'
import { toFeatureCollection, toMultiPolygon } from './NativeRangeEditor'

const occ: VectorLayer = {
  id: 'occurrences',
  label: 'Occurrences',
  source_layer: 'occurrences',
  tile_url: 'http://tiles/public.occurrence_tiles/{z}/{x}/{y}.pbf?taxon_key=42',
}

const raster = (id: string): RasterLayer => ({
  id,
  label: id,
  kind: 'continuous',
  tile_url: `http://titiler/cog/tiles/WebMercatorQuad/{z}/{x}/{y}.png?url=${id}`,
  legend: [],
  description: '',
})

describe('occurrence layer config', () => {
  it('passes the time-slider bound to pg_tileserv as max_year', () => {
    expect(occurrenceTileUrl(occ, null)).toBe(occ.tile_url)
    expect(occurrenceTileUrl(occ, 2012)).toBe(`${occ.tile_url}&max_year=2012`)
    const layer = occurrenceLayer({ layer: occ, maxYear: 2012 })
    expect(layer.id).toBe('occurrences')
    expect(layer.props.data).toBe(`${occ.tile_url}&max_year=2012`)
  })

  it('colours raw points by label and blends aggregated cells by introduced share', () => {
    expect(occurrenceColor({ range_label: 'introduced' })).toEqual(COLORS.introduced)
    expect(occurrenceColor({ n: 10, n_native: 10, n_introduced: 0 })).toEqual(COLORS.native)
    expect(occurrenceColor({ n: 10, n_native: 0, n_introduced: 10 })).toEqual(COLORS.introduced)
    expect(occurrenceColor({ n: 10, n_native: 0, n_introduced: 0 })).toEqual(COLORS.unknown)
    const mixed = occurrenceColor({ n: 10, n_native: 5, n_introduced: 5 })
    expect(mixed[0]).toBeGreaterThan(COLORS.native[0])
    expect(mixed[0]).toBeLessThan(COLORS.introduced[0])
  })

  it('scales aggregated circles by sqrt(count) with a cap', () => {
    expect(occurrenceRadius({ n: 1 })).toBeLessThan(occurrenceRadius({ n: 100 }))
    expect(occurrenceRadius({ n: 1_000_000 })).toBe(18)
  })
})

describe('projection layers (MESS pairing)', () => {
  const cfg = { base: raster('suitability'), extrapolation: raster('extrapolation') }

  it('adds the MESS extrapolation overlay on top of suitability by default', () => {
    const ids = projectionLayers(cfg, true).map((l) => l.id)
    expect(ids).toEqual(['projection-suitability', 'extrapolation-overlay'])
  })

  it('lets the user toggle the overlay off', () => {
    expect(projectionLayers(cfg, false).map((l) => l.id)).toEqual(['projection-suitability'])
  })

  it('does not stack the overlay on the MESS layer itself', () => {
    const ids = projectionLayers({ ...cfg, base: raster('mess') }, true).map((l) => l.id)
    expect(ids).toEqual(['projection-mess'])
  })

  it('pairs each projection with its own extrapolation mask', () => {
    const layers = {
      suitability: raster('suitability'),
      zones: raster('zones'),
      mess: raster('mess'),
      extrapolation: raster('extrapolation'),
      scenarios: [
        {
          id: 'scenario:ssp245',
          name: 'ssp245',
          label: 'ssp245',
          suitability: raster('suit-ssp245'),
          extrapolation: raster('extra-ssp245'),
        },
      ],
    } as unknown as LayerSet
    expect(resolveProjection(layers, 'zones').extrapolation?.id).toBe('extrapolation')
    const sc = resolveProjection(layers, 'scenario:ssp245')
    expect([sc.base?.id, sc.extrapolation?.id]).toEqual(['suit-ssp245', 'extra-ssp245'])
    expect(resolveProjection(layers, 'mess').extrapolation).toBeNull()
    expect(resolveProjection(layers, 'scenario:missing').base).toBeNull()
  })
})

describe('native-range polygon helpers', () => {
  it('round-trips MultiPolygon ↔ editable features', () => {
    const mp: GeoJSON.MultiPolygon = {
      type: 'MultiPolygon',
      coordinates: [
        [[[0, 0], [1, 0], [1, 1], [0, 0]]],
        [[[5, 5], [6, 5], [6, 6], [5, 5]]],
      ],
    }
    const fc = toFeatureCollection(mp)
    expect(fc.features).toHaveLength(2)
    expect(toMultiPolygon(fc)).toEqual(mp)
    expect(toMultiPolygon({ type: 'FeatureCollection', features: [] })).toBeNull()
  })
})
