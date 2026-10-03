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

/** 'suitability' | 'zones' | 'mess' | 'scenario:<name>' */
export type ProjectionLayerId = 'suitability' | 'zones' | 'mess' | `scenario:${string}`

export interface MapUiState {
  viewState: ViewState
  /** What the projection map currently shows (used to pick a high-resolution region). */
  viewBounds: Bounds | null
  projection: ProjectionLayerId
  /** MESS extrapolation overlay on top of the projection (on by default — see CLAUDE.md). */
  messOverlay: boolean
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
      hiresOverlay: true,
      maxYear: null,
      playing: false,
    })),
}
