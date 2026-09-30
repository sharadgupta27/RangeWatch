import { describe, expect, it, vi } from 'vitest'

import { fallbackStyle, releaseWebGLContext } from './DeckMap'

describe('WebGL context hygiene', () => {
  it('explicitly loses the context of an unmounted map canvas', () => {
    const loseContext = vi.fn()
    const canvas = {
      getContext: (type: string) =>
        type === 'webgl2' ? { getExtension: () => ({ loseContext }) } : null,
    } as unknown as HTMLCanvasElement
    releaseWebGLContext(canvas)
    expect(loseContext).toHaveBeenCalledOnce()
    expect(() => releaseWebGLContext(undefined)).not.toThrow()
  })
})

describe('basemap fallback', () => {
  it('is a valid style with only a background layer (data layers still render)', () => {
    const style = fallbackStyle('dark')
    expect(style.version).toBe(8)
    expect(style.layers).toHaveLength(1)
    expect(style.layers[0]?.type).toBe('background')
  })
})
