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

/** 'suitability' | 'zones' | 'mess' | 'scenario:<name>' */
export type ProjectionLayerId = 'suitability' | 'zones' | 'mess' | `scenario:${string}`

export interface MapUiState {
  viewState: ViewState
  projection: ProjectionLayerId
  /** MESS extrapolation overlay on top of the projection (on by default — see CLAUDE.md). */
  messOverlay: boolean
  showNativeRange: boolean
  /** Upper bound for the occurrence time slider; null = all years. */
  maxYear: number | null
  playing: boolean
}

export const INITIAL_VIEW: ViewState = { longitude: 10, latitude: 25, zoom: 1.4, pitch: 0, bearing: 0 }

export const mapStore = createStore<MapUiState>({
  viewState: INITIAL_VIEW,
  projection: 'suitability',
  messOverlay: true,
  showNativeRange: true,
  maxYear: null,
  playing: false,
})

export const mapActions = {
  setViewState: (viewState: ViewState) => mapStore.setState((s) => ({ ...s, viewState })),
  setProjection: (projection: ProjectionLayerId) => mapStore.setState((s) => ({ ...s, projection })),
  setMessOverlay: (messOverlay: boolean) => mapStore.setState((s) => ({ ...s, messOverlay })),
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
      maxYear: null,
      playing: false,
    })),
}
