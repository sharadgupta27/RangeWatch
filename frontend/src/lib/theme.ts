import { useSyncExternalStore } from 'react'

export type Theme = 'dark' | 'light'
const KEY = 'sdm-theme'

function read(): Theme {
  try {
    const v = localStorage.getItem(KEY)
    if (v === 'light' || v === 'dark') return v
  } catch {
    /* storage unavailable (private mode) — fall back to default */
  }
  return 'dark'
}

const listeners = new Set<() => void>()

export function applyTheme(theme: Theme) {
  document.documentElement.classList.toggle('dark', theme === 'dark')
  try {
    localStorage.setItem(KEY, theme)
  } catch {
    /* ignore */
  }
  listeners.forEach((l) => l())
}

export function initTheme() {
  document.documentElement.classList.toggle('dark', read() === 'dark')
}

export function useTheme(): [Theme, (t: Theme) => void] {
  const theme = useSyncExternalStore(
    (cb) => {
      listeners.add(cb)
      return () => listeners.delete(cb)
    },
    () => (document.documentElement.classList.contains('dark') ? 'dark' : 'light'),
    () => 'dark' as Theme,
  )
  return [theme, applyTheme]
}

const DARK_STYLE =
  import.meta.env.VITE_BASEMAP_STYLE_URL ??
  'https://basemaps.cartocdn.com/gl/dark-matter-nolabels-gl-style/style.json'
const LIGHT_STYLE = 'https://basemaps.cartocdn.com/gl/positron-nolabels-gl-style/style.json'

export function basemapStyle(theme: Theme): string {
  return theme === 'dark' ? DARK_STYLE : LIGHT_STYLE
}
