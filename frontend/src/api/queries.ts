/**
 * TanStack Query hooks — the only way components talk to the backend.
 *
 * The retrain/refresh *decision* lives entirely in the backend pipeline; the frontend only
 * enqueues runs, polls job + registry state, and invalidates caches when jobs finish.
 */
import {
  keepPreviousData,
  queryOptions,
  useMutation,
  useQuery,
  useQueryClient,
} from '@tanstack/react-query'
import { useEffect, useRef } from 'react'

import { api, unwrap } from './client'
import type { JobOut, NativeRangeUpdate, SeverityConfigModel } from './types'

export const qk = {
  search: (q: string) => ['species', 'search', q] as const,
  speciesList: () => ['species', 'list'] as const,
  species: (k: number) => ['species', k] as const,
  nativeRange: (k: number) => ['species', k, 'native-range'] as const,
  occurrences: (k: number, p: object) => ['species', k, 'occurrences', p] as const,
  timeline: (k: number) => ['species', k, 'timeline'] as const,
  models: (k: number) => ['species', k, 'models'] as const,
  model: (k: number, v: number) => ['species', k, 'models', v] as const,
  layers: (k: number) => ['species', k, 'layers'] as const,
  bulletin: (k: number) => ['species', k, 'bulletin'] as const,
  jobs: (k: number) => ['jobs', { taxonKey: k }] as const,
  bioclim: () => ['bioclim'] as const,
}

const ACTIVE = new Set<JobOut['status']>(['queued', 'running'])
export const isActiveJob = (job: JobOut | undefined) => !!job && ACTIVE.has(job.status)

/**
 * Whether the latest job finished since the previous poll: either the job we saw running
 * completed, or a new job appeared that had already finished (fast runs, e.g. "no new
 * records", can start and end between two polls).
 */
export function jobJustFinished(before: JobOut | undefined, latest: JobOut | undefined): boolean {
  if (!before || !latest || isActiveJob(latest)) return false
  return before.job_id === latest.job_id ? isActiveJob(before) : true
}

// ------------------------------------------------------------------ species
export function useSpeciesSearch(q: string) {
  return useQuery({
    queryKey: qk.search(q),
    queryFn: async () =>
      unwrap(await api.GET('/species/search', { params: { query: { q, limit: 10 } } })),
    enabled: q.trim().length >= 2,
    staleTime: 5 * 60_000,
    placeholderData: keepPreviousData,
  })
}

export const speciesListOptions = () =>
  queryOptions({
    queryKey: qk.speciesList(),
    queryFn: async () => unwrap(await api.GET('/species')),
    staleTime: 30_000,
  })

export function useSpeciesList() {
  // The nightly sweep changes statuses without any job the page could watch: poll slowly.
  return useQuery({ ...speciesListOptions(), refetchInterval: 60_000 })
}

export const speciesOptions = (taxonKey: number) =>
  queryOptions({
    queryKey: qk.species(taxonKey),
    queryFn: async () => {
      const res = await api.GET('/species/{taxon_key}', {
        params: { path: { taxon_key: taxonKey } },
      })
      if (res.response.status === 404) return null // not registered yet
      return unwrap(res)
    },
  })

export function useSpecies(taxonKey: number) {
  return useQuery(speciesOptions(taxonKey))
}

export function useBioclim() {
  return useQuery({
    queryKey: qk.bioclim(),
    queryFn: async () => unwrap(await api.GET('/bioclim')),
    // The active layer version can change on the server (new tag); don't pin it for the session.
    staleTime: 10 * 60_000,
    refetchOnWindowFocus: true,
    retry: false,
  })
}

// ------------------------------------------------------------------ jobs
export function useSpeciesJobs(taxonKey: number) {
  return useQuery({
    queryKey: qk.jobs(taxonKey),
    queryFn: async () =>
      unwrap(await api.GET('/jobs', { params: { query: { taxon_key: taxonKey, limit: 10 } } })),
    // Poll quickly while a job is active, slowly otherwise (catches scheduled runs).
    refetchInterval: (query) => (query.state.data?.some(isActiveJob) ? 2_000 : 30_000),
  })
}

/**
 * The most recent job for a species, plus cache invalidation when a job completes so
 * status badges, layers and lineage refresh without a page reload.
 */
export function useLatestJob(taxonKey: number) {
  const qc = useQueryClient()
  const jobs = useSpeciesJobs(taxonKey)
  const latest = jobs.data?.[0]
  const prev = useRef<JobOut | undefined>(undefined)

  useEffect(() => {
    const before = prev.current
    prev.current = latest
    if (jobJustFinished(before, latest)) void qc.invalidateQueries({ queryKey: ['species'] })
  }, [latest, qc])

  return { ...jobs, latest, active: isActiveJob(latest) }
}

export function useRunPipeline(taxonKey: number) {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: async (opts: { forceRetrain?: boolean; fullResync?: boolean } = {}) =>
      unwrap(
        await api.POST('/species/{taxon_key}/runs', {
          params: { path: { taxon_key: taxonKey } },
          body: { force_retrain: !!opts.forceRetrain, full_resync: !!opts.fullResync },
        }),
      ),
    onSuccess: () => qc.invalidateQueries({ queryKey: qk.jobs(taxonKey) }),
  })
}

// ------------------------------------------------------------ native range
export function useNativeRange(taxonKey: number, enabled = true) {
  return useQuery({
    queryKey: qk.nativeRange(taxonKey),
    queryFn: async () => {
      const res = await api.GET('/species/{taxon_key}/native-range', {
        params: { path: { taxon_key: taxonKey } },
      })
      if (res.response.status === 404) return null
      return unwrap(res)
    },
    enabled,
  })
}

export function useUpdateNativeRange(taxonKey: number) {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: async (body: NativeRangeUpdate) =>
      unwrap(
        await api.PUT('/species/{taxon_key}/native-range', {
          params: { path: { taxon_key: taxonKey } },
          body,
        }),
      ),
    onSuccess: () => {
      void qc.invalidateQueries({ queryKey: qk.species(taxonKey) })
      void qc.invalidateQueries({ queryKey: qk.jobs(taxonKey) })
    },
  })
}

// ------------------------------------------------------------ occurrences
export interface OccurrenceQuery {
  pageIndex: number
  pageSize: number
  source?: 'GBIF' | 'iNaturalist'
  rangeLabel?: 'native' | 'introduced' | 'unknown'
}

export function useOccurrences(taxonKey: number, q: OccurrenceQuery) {
  return useQuery({
    queryKey: qk.occurrences(taxonKey, q),
    queryFn: async () =>
      unwrap(
        await api.GET('/species/{taxon_key}/occurrences', {
          params: {
            path: { taxon_key: taxonKey },
            query: {
              offset: q.pageIndex * q.pageSize,
              limit: q.pageSize,
              source: q.source,
              range_label: q.rangeLabel,
            },
          },
        }),
      ),
    placeholderData: keepPreviousData,
  })
}

export function useTimeline(taxonKey: number, enabled = true) {
  return useQuery({
    queryKey: qk.timeline(taxonKey),
    queryFn: async () =>
      unwrap(
        await api.GET('/species/{taxon_key}/occurrences/timeline', {
          params: { path: { taxon_key: taxonKey } },
        }),
      ),
    enabled,
  })
}

// ------------------------------------------------------------ models + layers
export function useModelVersions(taxonKey: number, enabled = true) {
  return useQuery({
    queryKey: qk.models(taxonKey),
    queryFn: async () =>
      unwrap(
        await api.GET('/species/{taxon_key}/models', { params: { path: { taxon_key: taxonKey } } }),
      ),
    enabled,
  })
}

export function useModelVersion(taxonKey: number, version: number | null | undefined) {
  return useQuery({
    queryKey: qk.model(taxonKey, version ?? 0),
    queryFn: async () =>
      unwrap(
        await api.GET('/species/{taxon_key}/models/{model_version}', {
          params: { path: { taxon_key: taxonKey, model_version: version as number } },
        }),
      ),
    enabled: !!version,
    staleTime: Infinity, // model versions are immutable
  })
}

export function useLayers(taxonKey: number, enabled = true) {
  return useQuery({
    queryKey: qk.layers(taxonKey),
    queryFn: async () =>
      unwrap(
        await api.GET('/species/{taxon_key}/layers', { params: { path: { taxon_key: taxonKey } } }),
      ),
    enabled,
  })
}

// ------------------------------------------------------------ severity
export function useUpdateSeverityConfig(taxonKey: number) {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: async (body: SeverityConfigModel) =>
      unwrap(
        await api.PUT('/species/{taxon_key}/severity-config', {
          params: { path: { taxon_key: taxonKey } },
          body,
        }),
      ),
    onSuccess: () => {
      void qc.invalidateQueries({ queryKey: qk.species(taxonKey) })
      void qc.invalidateQueries({ queryKey: qk.models(taxonKey) })
      void qc.invalidateQueries({ queryKey: qk.speciesList() })
    },
  })
}

// ------------------------------------------------------------ bulletin
export function useBulletinStatus(taxonKey: number, enabled = true) {
  return useQuery({
    queryKey: qk.bulletin(taxonKey),
    queryFn: async () =>
      unwrap(
        await api.GET('/species/{taxon_key}/bulletin', {
          params: { path: { taxon_key: taxonKey } },
        }),
      ),
    enabled,
  })
}

export function useGenerateBulletin(taxonKey: number) {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: async () =>
      unwrap(
        await api.POST('/species/{taxon_key}/bulletin', {
          params: { path: { taxon_key: taxonKey } },
          body: {},
        }),
      ),
    onSuccess: () => qc.invalidateQueries({ queryKey: qk.jobs(taxonKey) }),
  })
}

// ------------------------------------------------------------ climate scenarios
export function useProjectScenarios(taxonKey: number) {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: async () =>
      unwrap(
        await api.POST('/species/{taxon_key}/scenarios', {
          params: { path: { taxon_key: taxonKey } },
        }),
      ),
    onSuccess: () => qc.invalidateQueries({ queryKey: qk.jobs(taxonKey) }),
  })
}

// ------------------------------------------------------------ validation suite
export function useValidationReport() {
  return useQuery({
    queryKey: ['validation'],
    queryFn: async () => {
      const res = await api.GET('/validation')
      if (res.response.status === 404) return null
      return unwrap(res)
    },
  })
}

export function useValidationJob() {
  const qc = useQueryClient()
  return useQuery({
    queryKey: ['jobs', 'validation'],
    queryFn: async () => {
      const jobs = unwrap(await api.GET('/jobs', { params: { query: { limit: 50 } } }))
      const job = jobs.find((j) => j.kind === 'validation') ?? null
      if (job && !isActiveJob(job)) void qc.invalidateQueries({ queryKey: ['validation'] })
      return job
    },
    refetchInterval: (q) => (isActiveJob(q.state.data ?? undefined) ? 3_000 : 60_000),
  })
}

export function useRunValidation() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: async (train: boolean) =>
      unwrap(await api.POST('/validation/runs', { body: { train } })),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['jobs', 'validation'] }),
  })
}
