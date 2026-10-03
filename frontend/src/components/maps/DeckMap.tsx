import { WebMercatorViewport, type Layer, type PickingInfo } from '@deck.gl/core'
import DeckGL, { type DeckGLRef } from '@deck.gl/react'
import { AlertTriangle, Minus, Plus, RefreshCw, RotateCcw } from 'lucide-react'
import { setWorkerUrl, type StyleSpecification } from 'maplibre-gl'
// MapLibre 6 spawns its worker from a URL; let Vite bundle the worker (and the shared chunk it
// imports) and hand MapLibre the emitted asset URL.
import maplibreWorkerUrl from 'maplibre-gl/dist/maplibre-gl-worker.mjs?worker&url'
import { useCallback, useEffect, useRef, useState, type ReactNode } from 'react'
import { Map as MapLibre, type ErrorEvent, type MapInstance } from 'react-map-gl/maplibre'

import { Button } from '@/components/ui/button'
import { basemapStyle, useTheme, type Theme } from '@/lib/theme'
import { cn } from '@/lib/utils'
import { INITIAL_VIEW, type Bounds, type ViewState } from '@/store/mapStore'

import { constrainView, MAX_ZOOM } from './constrainView'

setWorkerUrl(maplibreWorkerUrl)

/** Plain background used when the remote basemap style cannot be fetched (offline, blocked). */
export function fallbackStyle(theme: Theme): StyleSpecification {
  return {
    version: 8,
    sources: {},
    layers: [
      {
        id: 'background',
        type: 'background',
        paint: { 'background-color': theme === 'dark' ? '#1f2530' : '#e8ecf1' },
      },
    ],
  }
}

/**
 * Browsers cap live WebGL contexts per page (~16) and silently drop the *oldest* one when the
 * cap is exceeded, which blanks a visible map. Contexts from unmounted maps must therefore be
 * released eagerly instead of waiting for garbage collection.
 */
export function releaseWebGLContext(canvas: HTMLCanvasElement | null | undefined) {
  if (!canvas) return
  const gl = (canvas.getContext('webgl2') ?? canvas.getContext('webgl')) as
    | WebGL2RenderingContext
    | WebGLRenderingContext
    | null
  gl?.getExtension('WEBGL_lose_context')?.loseContext()
}

export interface DeckMapProps {
  layers: Layer[]
  viewState: ViewState
  onViewStateChange: (v: ViewState) => void
  getTooltip?: (info: PickingInfo) => string | null
  getCursor?: (state: { isDragging: boolean; isHovering: boolean }) => string
  doubleClickZoom?: boolean
  /** Visible [west, south, east, north] whenever the view or the map size changes. */
  onBoundsChange?: (bounds: Bounds) => void
  className?: string
  children?: ReactNode
}

/** Visible lon/lat box of a web-mercator view (clamped to the globe). */
export function viewBounds(view: ViewState, width: number, height: number): Bounds {
  const [west, south, east, north] = new WebMercatorViewport({
    ...view,
    width,
    height,
  }).getBounds()
  return [Math.max(west, -180), Math.max(south, -85), Math.min(east, 180), Math.min(north, 85)]
}

const TOOLTIP_STYLE = {
  background: 'var(--popover)',
  color: 'var(--popover-foreground)',
  border: '1px solid var(--border)',
  borderRadius: '6px',
  fontSize: '11px',
  padding: '6px 8px',
  whiteSpace: 'pre',
}

/**
 * deck.gl as the root (owns view state + events) with a MapLibre basemap underneath.
 * View state is fully controlled so several maps can be kept in sync via mapStore.
 */
export function DeckMap(props: DeckMapProps) {
  // Remounting with a new key recreates both WebGL contexts after a context loss.
  const [generation, setGeneration] = useState(0)
  const [lost, setLost] = useState(false)
  const recover = useCallback(() => {
    setLost(false)
    setGeneration((g) => g + 1)
  }, [])
  return (
    <div className={cn('relative h-full w-full overflow-hidden', props.className)}>
      <DeckMapInner key={generation} {...props} onContextLost={() => setLost(true)} />
      {lost && (
        <div className="absolute inset-0 z-20 flex items-center justify-center bg-background/70 backdrop-blur-sm">
          <div className="max-w-xs rounded-lg border bg-card p-4 text-center text-sm shadow-lg">
            <AlertTriangle className="mx-auto mb-2 size-5 text-warning" />
            <p className="font-medium">The browser dropped this map's graphics context</p>
            <p className="mt-1 text-xs text-muted-foreground">
              This happens when too many WebGL canvases are open (other tabs or apps).
            </p>
            <Button size="sm" className="mt-3" onClick={recover}>
              <RefreshCw /> Reload map
            </Button>
          </div>
        </div>
      )}
    </div>
  )
}

function DeckMapInner({
  layers,
  viewState,
  onViewStateChange,
  getTooltip,
  getCursor,
  doubleClickZoom = true,
  onBoundsChange,
  children,
  onContextLost,
}: DeckMapProps & { onContextLost: () => void }) {
  const [theme] = useTheme()
  const [styleFailed, setStyleFailed] = useState(false)
  const deckRef = useRef<DeckGLRef>(null)
  const canvases = useRef<{ deck?: HTMLCanvasElement; map?: HTMLCanvasElement }>({})
  const styleUrl = basemapStyle(theme)
  // Viewport size: needed to apply MapLibre's camera constraint to deck.gl's view state.
  const [size, setSize] = useState({ width: 0, height: 0 })
  const constrain = (v: ViewState) => constrainView(v, size.width, size.height)
  const view = constrain(viewState)

  useEffect(() => setStyleFailed(false), [styleUrl])

  const { longitude, latitude, zoom } = view
  useEffect(() => {
    if (!onBoundsChange || size.width === 0 || size.height === 0) return
    onBoundsChange(
      viewBounds({ longitude, latitude, zoom, pitch: 0, bearing: 0 }, size.width, size.height),
    )
  }, [onBoundsChange, longitude, latitude, zoom, size.width, size.height])

  const watch = useCallback(
    (canvas: HTMLCanvasElement | undefined) => {
      canvas?.addEventListener('webglcontextlost', (e) => {
        // Ignore the loss we trigger ourselves on unmount.
        if (!(e.target as HTMLCanvasElement).dataset.released) onContextLost()
      })
    },
    [onContextLost],
  )

  useEffect(
    () => () => {
      const { deck, map } = canvases.current
      // After React/deck/MapLibre finish tearing down, free anything still holding a context.
      window.setTimeout(() => {
        for (const c of [deck, map]) {
          if (c) {
            c.dataset.released = '1'
            releaseWebGLContext(c)
          }
        }
      }, 0)
    },
    [],
  )

  const zoomBy = (d: number) =>
    onViewStateChange(
      constrain({ ...view, zoom: Math.min(MAX_ZOOM, Math.max(0, view.zoom + d)) }),
    )

  const onMapError = useCallback(
    (e: ErrorEvent) => {
      const err = e.error as { url?: string; message?: string } | undefined
      if (err?.url === styleUrl || /style/i.test(err?.message ?? '')) setStyleFailed(true)
    },
    [styleUrl],
  )

  return (
    <>
      <DeckGL
        ref={deckRef}
        layers={layers}
        viewState={view}
        onResize={setSize}
        controller={{ doubleClickZoom, dragRotate: false, touchRotate: false }}
        onLoad={() => {
          const canvas = deckRef.current?.deck?.getCanvas() ?? undefined
          canvases.current.deck = canvas
          watch(canvas)
        }}
        onViewStateChange={({ viewState: v }) => {
          const vs = v as ViewState & { width?: number; height?: number }
          onViewStateChange(
            constrainView(
              { longitude: vs.longitude, latitude: vs.latitude, zoom: vs.zoom, pitch: 0, bearing: 0 },
              vs.width ?? size.width,
              vs.height ?? size.height,
            ),
          )
        }}
        getTooltip={
          getTooltip
            ? (info) => {
                const text = getTooltip(info)
                return text ? { text, style: TOOLTIP_STYLE } : null
              }
            : undefined
        }
        // Passing getCursor={undefined} would override deck.gl's default cursor function.
        {...(getCursor ? { getCursor } : {})}
      >
        {/* No `reuseMaps`: pooled maps keep their WebGL context alive; `remove()` frees it. */}
        <MapLibre
          mapStyle={styleFailed ? fallbackStyle(theme) : styleUrl}
          // A single bounded world; constrainView keeps deck.gl on the same camera.
          renderWorldCopies={false}
          onError={onMapError}
          onLoad={(e) => {
            const canvas = (e.target as MapInstance).getCanvas()
            canvases.current.map = canvas
            watch(canvas)
          }}
        />
      </DeckGL>
      {styleFailed && (
        <div className="absolute bottom-3 left-1/2 z-10 -translate-x-1/2 rounded-md border bg-card/90 px-2 py-1 text-[11px] text-muted-foreground">
          Basemap unavailable — showing data layers only
        </div>
      )}
      <div className="absolute right-3 bottom-8 z-10 flex flex-col gap-1">
        <Button size="icon" variant="outline" className="size-7 bg-card/85 backdrop-blur" onClick={() => zoomBy(1)} aria-label="Zoom in">
          <Plus />
        </Button>
        <Button size="icon" variant="outline" className="size-7 bg-card/85 backdrop-blur" onClick={() => zoomBy(-1)} aria-label="Zoom out">
          <Minus />
        </Button>
        <Button size="icon" variant="outline" className="size-7 bg-card/85 backdrop-blur" onClick={() => onViewStateChange(constrain(INITIAL_VIEW))} aria-label="Reset view">
          <RotateCcw />
        </Button>
      </div>
      {children}
    </>
  )
}
