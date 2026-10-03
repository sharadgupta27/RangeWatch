import { clsx, type ClassValue } from 'clsx'
import { twMerge } from 'tailwind-merge'

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}

const nf0 = new Intl.NumberFormat('en', { maximumFractionDigits: 0 })
const nfCompact = new Intl.NumberFormat('en', { notation: 'compact', maximumFractionDigits: 1 })

export function formatNumber(v: number | null | undefined, digits = 0): string {
  if (v === null || v === undefined || Number.isNaN(v)) return '–'
  return digits === 0
    ? nf0.format(v)
    : v.toLocaleString('en', { minimumFractionDigits: digits, maximumFractionDigits: digits })
}

export function formatCompact(v: number | null | undefined): string {
  if (v === null || v === undefined || Number.isNaN(v)) return '–'
  return nfCompact.format(v)
}

export function formatDate(v: string | null | undefined, withTime = false): string {
  if (!v) return '–'
  const d = new Date(v)
  return withTime
    ? d.toLocaleString('en-GB', { dateStyle: 'medium', timeStyle: 'short' })
    : d.toLocaleDateString('en-GB', { dateStyle: 'medium' })
}

export function formatRelative(v: string | null | undefined): string {
  if (!v) return 'never'
  const diff = (Date.now() - new Date(v).getTime()) / 1000
  const rtf = new Intl.RelativeTimeFormat('en', { numeric: 'auto' })
  if (diff < 60) return 'just now'
  if (diff < 3600) return rtf.format(-Math.round(diff / 60), 'minute')
  if (diff < 86400) return rtf.format(-Math.round(diff / 3600), 'hour')
  return rtf.format(-Math.round(diff / 86400), 'day')
}

/** WorldClim resolution code → display label ('30s' → 30″, '2.5m' → 2.5′). */
export function resolutionLabel(code: string): string {
  return code.replace(/s$/, '″').replace(/m$/, '′')
}
