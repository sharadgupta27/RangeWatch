import createClient from 'openapi-fetch'

import type { paths } from './schema'

/** Base URL of the FastAPI backend (public, build-time; never a secret). */
// Same-origin by default: the Nginx gateway (and the Vite dev proxy) route /api to FastAPI.
export const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL ?? '/api').replace(
  /\/$/,
  '',
)

/** Typed client generated from the backend's OpenAPI schema (see `npm run gen:api`). */
export const api = createClient<paths>({ baseUrl: API_BASE_URL })

export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
  ) {
    super(message)
  }
}

/** Unwrap an openapi-fetch result, turning FastAPI error bodies into ApiError. */
export function unwrap<T>(result: { data?: T; error?: unknown; response: Response }): T {
  if (result.error !== undefined || result.data === undefined) {
    const err = result.error as { detail?: unknown } | undefined
    const detail =
      typeof err?.detail === 'string'
        ? err.detail
        : Array.isArray(err?.detail)
          ? (err.detail as Array<{ msg?: string }>).map((d) => d.msg).join('; ')
          : result.response.statusText
    throw new ApiError(detail || 'Request failed', result.response.status)
  }
  return result.data
}

export function absoluteApiUrl(path: string): string {
  return `${API_BASE_URL}${path}`
}
