import type { ViewState } from '@/store/mapStore'

/** MapLibre's Web Mercator constants (tile size and default latitude range). */
const TILE_SIZE = 512
const MAX_LAT = 85.051129
export const MAX_ZOOM = 14

const mercatorX = (lng: number) => (180 + lng) / 360
const lngFromMercatorX = (x: number) => x * 360 - 180
const mercatorY = (lat: number) =>
  (180 - (180 / Math.PI) * Math.log(Math.tan(Math.PI / 4 + (lat * Math.PI) / 360))) / 360
const latFromMercatorY = (y: number) =>
  (360 / Math.PI) * Math.atan(Math.exp(((180 - y * 360) * Math.PI) / 180)) - 90

const clamp = (v: number, lo: number, hi: number) => Math.min(Math.max(v, lo), hi)

/**
 * One bounded world, no copies: the camera rule MapLibre enforces with
 * `renderWorldCopies={false}`, applied to deck.gl's view state too.
 *  - the world must fill the viewport (no zooming out past it),
 *  - the view may not pan past ±180° or ±85.05°.
 * deck.gl does not apply this rule itself, so without it the two drift apart whenever MapLibre
 * clamps a pan that deck.gl does not — the data slides off the basemap.
 */
export function constrainView(v: ViewState, width: number, height: number): ViewState {
  let zoom = clamp(v.zoom, 0, MAX_ZOOM)
  if (!(width > 0 && height > 0)) return { ...v, zoom }
  // The clipped Mercator world is square: fill the larger viewport dimension.
  const fill = Math.max(width, height) / (TILE_SIZE * 2 ** zoom)
  if (fill > 1) zoom += Math.log2(fill)
  const worldSize = TILE_SIZE * 2 ** zoom
  const minY = mercatorY(MAX_LAT) * worldSize
  const maxY = mercatorY(-MAX_LAT) * worldSize
  const x = clamp(mercatorX(clamp(v.longitude, -180, 180)) * worldSize, width / 2, worldSize - width / 2)
  const y = clamp(mercatorY(clamp(v.latitude, -MAX_LAT, MAX_LAT)) * worldSize, minY + height / 2, maxY - height / 2)
  return { ...v, zoom, longitude: lngFromMercatorX(x / worldSize), latitude: latFromMercatorY(y / worldSize) }
}
