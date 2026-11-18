import { describe, expect, it } from 'vitest'

import { INDIA_BOUNDARY_SOURCE, indiaBoundaryLayers, indiaBoundaryUrl } from './indiaBoundary'

type LineLayer = {
  id: string
  source: string
  'source-layer': string
  minzoom?: number
  maxzoom?: number
  paint: Record<string, unknown>
}

const layers = (theme: 'dark' | 'light') => indiaBoundaryLayers(theme) as unknown as LineLayer[]

describe('India boundary layers', () => {
  it('paints every deletion before any addition', () => {
    const ids = layers('dark').map((l) => l.id)
    const lastDel = Math.max(...ids.map((id, i) => (id.includes('-del-') ? i : -1)))
    const firstAdd = ids.findIndex((id) => id.includes('-add-'))
    expect(lastDel).toBeLessThan(firstAdd)
  })

  it('uses all twelve correction layers from the bundled source', () => {
    const used = new Set(layers('dark').map((l) => l['source-layer']))
    for (const op of ['add', 'del'])
      for (const src of ['ne', 'osm'])
        for (const kind of ['', '-disp', '-internal']) expect(used).toContain(`to-${op}-${src}${kind}`)
    expect(layers('dark').every((l) => l.source === INDIA_BOUNDARY_SOURCE)).toBe(true)
  })

  it('erases in the land colour of each basemap theme', () => {
    const del = (t: 'dark' | 'light') => layers(t).find((l) => l.id.endsWith('del-osm'))!
    expect(del('dark').paint['line-color']).toBe('#0e0e0e')
    expect(del('light').paint['line-color']).toBe('#fafaf8')
  })

  it('draws Natural Earth lines at low zoom and OSM lines from z5, never both', () => {
    const add = (id: string) => layers('dark').find((l) => l.id === `india-boundary-add-${id}`)!
    expect(add('ne').maxzoom).toBe(5)
    expect(add('ne').minzoom).toBeUndefined()
    expect(add('osm').minzoom).toBe(5)
    expect(add('osm').maxzoom).toBeUndefined()
  })

  it('points MapLibre at the same-origin data file', () => {
    expect(indiaBoundaryUrl('http://localhost:8080/species/1')).toMatch(
      /^pmtiles:\/\/http:\/\/localhost:8080\/.*india_boundary_corrections.*\.pmtiles$/,
    )
  })
})

describe('which basemaps get the India boundary', () => {
  it('covers the CARTO styles, including the Docker build default, and skips custom styles', async () => {
    const { basemapHasIndiaBoundary } = await import('@/lib/theme')
    const carto = 'https://basemaps.cartocdn.com/gl'
    expect(basemapHasIndiaBoundary('dark', `${carto}/dark-matter-nolabels-gl-style/style.json`)).toBe(true)
    expect(basemapHasIndiaBoundary('dark', `${carto}/dark-matter-gl-style/style.json`)).toBe(true)
    expect(basemapHasIndiaBoundary('light', `${carto}/positron-nolabels-gl-style/style.json`)).toBe(true)
    expect(basemapHasIndiaBoundary('dark', 'https://tiles.example.org/style.json')).toBe(false)
  })
})
