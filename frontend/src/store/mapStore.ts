/**
 * Ephemeral, non-persisted map/UI state (TanStack Store). Server data never lives here.
 */
import { createStore } from '@tanstack/react-store'

export interface ViewState {
  longitude: number
  latitude: number
  zoom: number
  pitch: number
  bearing: number
}

/** [west, south, east, north] in degrees. */
export type Bounds = [number, number, number, number]

/** Extrapolation diagnostics that can be shown as the projection map's own view. */
export type DiagnosticLayerId = 'mess' | 'exdet' | 'mop' | 'shape' | 'aoa' | 'consensus'
export const DIAGNOSTIC_LAYER_IDS: readonly DiagnosticLayerId[] = [
  'consensus',
  'mess',
  'exdet',
  'mop',
  'shape',
  'aoa',
]
export function isDiagnosticLayer(id: string): id is DiagnosticLayerId {
  return (DIAGNOSTIC_LAYER_IDS as readonly string[]).includes(id)
}

/** 'suitability' | 'zones' | a diagnostic | 'scenario:<name>' */
export type ProjectionLayerId = 'suitability' | 'zones' | DiagnosticLayerId | `scenario:${string}`

/** What the extrapolation overlay draws: the MESS < 0 mask or the graded diagnostic consensus. */
export type OverlaySource = 'mess' | 'consensus'

export interface MapUiState {
  viewState: ViewState
  /** What the projection map currently shows (used to pick a high-resolution region). */
  viewBounds: Bounds | null
  projection: ProjectionLayerId
  /** Extrapolation overlay on top of the projection (on by default — see CLAUDE.md). */
  messOverlay: boolean
  overlaySource: OverlaySource
  /** Regional high-resolution suitability drawn over the global current-climate layer. */
  hiresOverlay: boolean
  showNativeRange: boolean
  /** Upper bound for the occurrence time slider; null = all years. */
  maxYear: number | null
  playing: boolean
}

export const INITIAL_VIEW: ViewState = { longitude: 10, latitude: 25, zoom: 1.4, pitch: 0, bearing: 0 }

export const mapStore = createStore<MapUiState>({
  viewState: INITIAL_VIEW,
  viewBounds: null,
  projection: 'suitability',
  messOverlay: true,
  overlaySource: 'mess',
  hiresOverlay: true,
  showNativeRange: true,
  maxYear: null,
  playing: false,
})

export const mapActions = {
  setViewState: (viewState: ViewState) => mapStore.setState((s) => ({ ...s, viewState })),
  setViewBounds: (viewBounds: Bounds) => mapStore.setState((s) => ({ ...s, viewBounds })),
  setProjection: (projection: ProjectionLayerId) => mapStore.setState((s) => ({ ...s, projection })),
  setMessOverlay: (messOverlay: boolean) => mapStore.setState((s) => ({ ...s, messOverlay })),
  setOverlaySource: (overlaySource: OverlaySource) =>
    mapStore.setState((s) => ({ ...s, overlaySource })),
  setHiresOverlay: (hiresOverlay: boolean) => mapStore.setState((s) => ({ ...s, hiresOverlay })),
  setShowNativeRange: (showNativeRange: boolean) =>
    mapStore.setState((s) => ({ ...s, showNativeRange })),
  setMaxYear: (maxYear: number | null) => mapStore.setState((s) => ({ ...s, maxYear })),
  setPlaying: (playing: boolean) => mapStore.setState((s) => ({ ...s, playing })),
  resetForSpecies: () =>
    mapStore.setState((s) => ({
      ...s,
      viewState: INITIAL_VIEW,
      projection: 'suitability',
      messOverlay: true,
      overlaySource: 'mess',
      hiresOverlay: true,
      maxYear: null,
      playing: false,
    })),
}
