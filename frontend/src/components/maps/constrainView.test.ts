import { describe, expect, it } from 'vitest'

import { constrainView } from './constrainView'

const base = { longitude: 10, latitude: 0, zoom: 2, pitch: 0, bearing: 0 }
const W = 900
const H = 520
// World size in px at zoom z (512 px tiles; the clipped Mercator world is square).
const worldPx = (z: number) => 512 * 2 ** z
const mercX = (lng: number) => (180 + lng) / 360
const mercY = (lat: number) =>
  (180 - (180 / Math.PI) * Math.log(Math.tan(Math.PI / 4 + (lat * Math.PI) / 360))) / 360

describe('constrainView (MapLibre single-world camera rule applied to deck.gl)', () => {
  it('leaves an unconstrained view untouched', () => {
    const v = constrainView(base, W, H)
    expect(v.zoom).toBe(2)
    expect(v.longitude).toBeCloseTo(10, 9)
    expect(v.latitude).toBeCloseTo(0, 9)
  })

  it('does not zoom out past the point where the world fills the viewport', () => {
    const v = constrainView({ ...base, zoom: 0 }, W, H)
    expect(worldPx(v.zoom)).toBeCloseTo(W, 6) // width is the larger dimension here
  })

  it('does not pan past the east/west edges', () => {
    const v = constrainView({ ...base, longitude: 175 }, W, H)
    // The viewport's right edge sits exactly on the antimeridian.
    expect(mercX(v.longitude) * worldPx(v.zoom) + W / 2).toBeCloseTo(worldPx(v.zoom), 6)
    expect(constrainView({ ...base, longitude: -175 }, W, H).longitude).toBeCloseTo(-v.longitude, 6)
  })

  it('does not pan past the poles', () => {
    const v = constrainView({ ...base, latitude: 84 }, W, H)
    expect(mercY(v.latitude) * worldPx(v.zoom)).toBeCloseTo(H / 2, 3)
    expect(constrainView({ ...base, latitude: -84 }, W, H).latitude).toBeCloseTo(-v.latitude, 6)
  })

  it('clamps out-of-range longitudes from a wrapped drag', () => {
    const v = constrainView({ ...base, longitude: 400 }, W, H)
    expect(v.longitude).toBeLessThanOrEqual(180)
  })

  it('only clamps zoom before the viewport is measured', () => {
    expect(constrainView({ ...base, latitude: 89, zoom: 20 }, 0, 0)).toEqual({ ...base, latitude: 89, zoom: 14 })
  })
})
