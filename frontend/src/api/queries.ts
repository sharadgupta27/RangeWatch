/**
 * TanStack Query hooks — the only way components talk to the backend.
 *
 * The retrain/refresh *decision* lives entirely in the backend pipeline; the frontend only
 * enqueues runs, follows job + registry state (an SSE stream, with polling as the fallback),
 * and invalidates caches when jobs finish.
 */
import {
  keepPreviousData,
  queryOptions,
  useMutation,
  useQuery,
  useQueryClient,
} from '@tanstack/react-query'
import { useEffect, useRef, useState } from 'react'

import { absoluteApiUrl, api, unwrap } from './client'
import type { paths } from './schema'
import type {
  AlertSubscriptionIn,
  AnnotationIn,
  BatchRunRequest,
  BenchmarkRequest,
  DatasetUploadFields,
  JobOut,
  ModelCardOut,
  NativeRangeUpdate,
  SeverityConfigModel,
  SpeciesListIn,
  StudyRegionUpdate,
} from './types'

/** Payload of one `jobs` event on GET /jobs/events (typed from the OpenAPI contract). */
type JobEventPayload =
  paths['/jobs/events']['get']['responses'][200]['content']['text/event-stream']

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
  datasets: (k: number) => ['species', k, 'datasets'] as const,
  jobs: (k: number) => ['jobs', { taxonKey: k }] as const,
  bioclim: () => ['bioclim'] as const,
  annotations: (k: number) => ['species', k, 'annotations'] as const,
  alerts: (k: number) => ['species', k, 'alerts'] as const,
  subscriptions: (k: number) => ['alerts', 'subscriptions', k] as const,
  exports: (k: number, v: number) => ['species', k, 'models', v, 'exports'] as const,
  me: () => ['me'] as const,
  benchmark: () => ['validation', 'benchmark'] as const,
  lists: () => ['lists'] as const,
  listSpecies: (listId: string) => ['species', 'list', { listId }] as const,
  modelCard: (k: number, v: number) => ['species', k, 'models', v, 'model-card'] as const,
  targetGroups: (k: number) => ['species', k, 'target-group-comparison'] as const,
  sharedBias: (k: number) => ['species', k, 'shared-bias'] as const,
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
export const JOB_LIST_LIMIT = 10

export function jobEventsUrl(taxonKey: number): string {
  return absoluteApiUrl(`/jobs/events?taxon_key=${taxonKey}&limit=${JOB_LIST_LIMIT}`)
}

/**
 * Live job list over Server-Sent Events: every `jobs` event replaces the cached list, so
 * progress shows as soon as a worker reports it. Returns whether the stream is open; the
 * browser's EventSource reconnects by itself after drops and server-side recycling.
 */
export function useJobStream(taxonKey: number): boolean {
  const qc = useQueryClient()
  const [connected, setConnected] = useState(false)
  useEffect(() => {
    if (typeof EventSource === 'undefined') return
    const source = new EventSource(jobEventsUrl(taxonKey))
    const onJobs = (e: MessageEvent<string>) => {
      qc.setQueryData<JobOut[]>(qk.jobs(taxonKey), JSON.parse(e.data) as JobEventPayload)
      setConnected(true)
    }
    const onOpen = () => setConnected(true)
    const onError = () => setConnected(false)
    source.addEventListener('jobs', onJobs)
    source.addEventListener('open', onOpen)
    source.addEventListener('error', onError)
    return () => {
      source.close()
      setConnected(false)
    }
  }, [qc, taxonKey])
  return connected
}

/** Poll interval for the job list: none while the live stream is up, else fast/slow. */
export function jobPollInterval(live: boolean, jobs: JobOut[] | undefined): number | false {
  if (live) return false
  // Poll quickly while a job is active, slowly otherwise (catches scheduled runs).
  return jobs?.some(isActiveJob) ? 2_000 : 30_000
}

export function useSpeciesJobs(taxonKey: number) {
  const live = useJobStream(taxonKey)
  return useQuery({
    queryKey: qk.jobs(taxonKey),
    queryFn: async () =>
      unwrap(
        await api.GET('/jobs', {
          params: { query: { taxon_key: taxonKey, limit: JOB_LIST_LIMIT } },
        }),
      ),
    refetchInterval: (query) => jobPollInterval(live, query.state.data),
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
    // Model versions are immutable; derived results (cross-checks) arrive via invalidation
    // when their job finishes.
    staleTime: Infinity,
  })
}

/** Layer descriptors; `threshold` adds a binary map at a user-chosen value (U5). */
export function useLayers(taxonKey: number, enabled = true, threshold: number | null = null) {
  return useQuery({
    queryKey: threshold == null ? qk.layers(taxonKey) : [...qk.layers(taxonKey), { threshold }],
    queryFn: async () =>
      unwrap(
        await api.GET('/species/{taxon_key}/layers', {
          params: {
            path: { taxon_key: taxonKey },
            query: threshold == null ? {} : { threshold },
          },
        }),
      ),
    enabled,
    placeholderData: keepPreviousData,
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

// ------------------------------------------------------------ high-resolution region
export function useProjectHires(taxonKey: number) {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: async (bbox: [number, number, number, number]) =>
      unwrap(
        await api.POST('/species/{taxon_key}/hires', {
          params: { path: { taxon_key: taxonKey } },
          body: { bbox },
        }),
      ),
    onSuccess: () => qc.invalidateQueries({ queryKey: qk.jobs(taxonKey) }),
  })
}

// ------------------------------------------------------------ climate cross-check
export function useRunCrossCheck(taxonKey: number) {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: async () =>
      unwrap(
        await api.POST('/species/{taxon_key}/crosscheck', {
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

// ------------------------------------------------------------ reference datasets
/** Multipart body for an upload: the typed fields plus the file itself. */
export function datasetFormData(file: File, fields: DatasetUploadFields): FormData {
  const fd = new FormData()
  fd.append('file', file, file.name)
  for (const [key, value] of Object.entries(fields)) {
    if (typeof value === 'string' && value.trim() !== '') fd.append(key, value.trim())
  }
  return fd
}

export function useReferenceDatasets(taxonKey: number, enabled = true) {
  return useQuery({
    queryKey: qk.datasets(taxonKey),
    queryFn: async () =>
      unwrap(
        await api.GET('/species/{taxon_key}/datasets', {
          params: { path: { taxon_key: taxonKey } },
        }),
      ),
    enabled,
  })
}

export function useUploadDataset(taxonKey: number) {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: async ({ file, fields }: { file: File; fields: DatasetUploadFields }) =>
      unwrap(
        await api.POST('/species/{taxon_key}/datasets', {
          params: { path: { taxon_key: taxonKey } },
          // The generated body types the file as a string (OpenAPI "binary"); the serializer
          // sends the real File in a FormData instead.
          body: { ...fields, file: file.name },
          bodySerializer: () => datasetFormData(file, fields),
        }),
      ),
    onSuccess: () => {
      void qc.invalidateQueries({ queryKey: qk.datasets(taxonKey) })
      void qc.invalidateQueries({ queryKey: qk.layers(taxonKey) })
      void qc.invalidateQueries({ queryKey: qk.jobs(taxonKey) })
      // a training upload flags the species for retraining (status badge)
      void qc.invalidateQueries({ queryKey: qk.species(taxonKey) })
    },
  })
}

export function useDeleteDataset(taxonKey: number) {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: async (datasetId: string) => {
      const res = await api.DELETE('/species/{taxon_key}/datasets/{dataset_id}', {
        params: { path: { taxon_key: taxonKey, dataset_id: datasetId } },
      })
      if (!res.response.ok) unwrap(res)
    },
    onSuccess: () => {
      void qc.invalidateQueries({ queryKey: qk.datasets(taxonKey) })
      void qc.invalidateQueries({ queryKey: qk.layers(taxonKey) })
      void qc.invalidateQueries({ queryKey: qk.species(taxonKey) })
    },
  })
}

export function useEvaluateDatasets(taxonKey: number) {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: async () =>
      unwrap(
        await api.POST('/species/{taxon_key}/datasets/evaluate', {
          params: { path: { taxon_key: taxonKey } },
        }),
      ),
    onSuccess: () => qc.invalidateQueries({ queryKey: qk.jobs(taxonKey) }),
  })
}

// ------------------------------------------------------------ users (U14)
export function useMe() {
  return useQuery({
    queryKey: qk.me(),
    queryFn: async () => unwrap(await api.GET('/me')),
    staleTime: 5 * 60_000,
  })
}

/** What the current user may do; the API enforces the same rules regardless. */
export function usePermissions(): { run: boolean; curate: boolean } {
  const { data } = useMe()
  return { run: data?.permissions.run ?? false, curate: data?.permissions.curate ?? false }
}

// ------------------------------------------------------------ study region (U2)
export function useUpdateStudyRegion(taxonKey: number) {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: async (body: StudyRegionUpdate) =>
      unwrap(
        await api.PUT('/species/{taxon_key}/study-region', {
          params: { path: { taxon_key: taxonKey } },
          body,
        }),
      ),
    onSuccess: (data) => {
      qc.setQueryData(qk.species(taxonKey), data)
      void qc.invalidateQueries({ queryKey: qk.speciesList() })
    },
  })
}

// ------------------------------------------------------------ annotations (U11)
export function useAnnotations(taxonKey: number, enabled = true) {
  return useQuery({
    queryKey: qk.annotations(taxonKey),
    queryFn: async () =>
      unwrap(
        await api.GET('/species/{taxon_key}/annotations', {
          params: { path: { taxon_key: taxonKey } },
        }),
      ),
    enabled,
  })
}

export function useAnnotateRecord(taxonKey: number) {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: async (body: AnnotationIn) =>
      unwrap(
        await api.POST('/species/{taxon_key}/annotations', {
          params: { path: { taxon_key: taxonKey } },
          body,
        }),
      ),
    onSuccess: () => {
      void qc.invalidateQueries({ queryKey: qk.annotations(taxonKey) })
      void qc.invalidateQueries({ queryKey: ['species', taxonKey, 'occurrences'] })
      void qc.invalidateQueries({ queryKey: qk.species(taxonKey), exact: true })
    },
  })
}

// ------------------------------------------------------------ alerts (U10)
export function useSpeciesAlerts(taxonKey: number, enabled = true) {
  return useQuery({
    queryKey: qk.alerts(taxonKey),
    queryFn: async () =>
      unwrap(
        await api.GET('/species/{taxon_key}/alerts', {
          params: { path: { taxon_key: taxonKey } },
        }),
      ),
    enabled,
  })
}

export function alertFeedUrl(taxonKey?: number): string {
  return absoluteApiUrl(`/alerts/feed.xml${taxonKey ? `?taxon_key=${taxonKey}` : ''}`)
}

export function useAlertSubscriptions(taxonKey: number) {
  return useQuery({
    queryKey: qk.subscriptions(taxonKey),
    queryFn: async () =>
      unwrap(
        await api.GET('/alerts/subscriptions', { params: { query: { taxon_key: taxonKey } } }),
      ),
  })
}

export function useCreateSubscription(taxonKey: number) {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: async (body: AlertSubscriptionIn) =>
      unwrap(await api.POST('/alerts/subscriptions', { body })),
    onSuccess: () => qc.invalidateQueries({ queryKey: qk.subscriptions(taxonKey) }),
  })
}

export function useDeleteSubscription(taxonKey: number) {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: async (subscriptionId: string) => {
      const res = await api.DELETE('/alerts/subscriptions/{subscription_id}', {
        params: { path: { subscription_id: subscriptionId } },
      })
      if (!res.response.ok) unwrap(res)
    },
    onSuccess: () => qc.invalidateQueries({ queryKey: qk.subscriptions(taxonKey) }),
  })
}

// ------------------------------------------------------------ exports (U8)
export function useExports(taxonKey: number, version: number | null | undefined) {
  return useQuery({
    queryKey: qk.exports(taxonKey, version ?? 0),
    queryFn: async () =>
      unwrap(
        await api.GET('/species/{taxon_key}/models/{model_version}/exports', {
          params: { path: { taxon_key: taxonKey, model_version: version ?? 0 } },
        }),
      ),
    enabled: !!version,
  })
}

// ------------------------------------------------------------ benchmark (§4.6)
export function useBenchmarkReport() {
  return useQuery({
    queryKey: qk.benchmark(),
    queryFn: async () => {
      const res = await api.GET('/validation/benchmark')
      if (res.response.status === 404) return null
      return unwrap(res)
    },
  })
}

export function useRunBenchmark() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: async (body: BenchmarkRequest) =>
      unwrap(await api.POST('/validation/benchmark', { body })),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['jobs', 'validation'] }),
  })
}

/** Latest benchmark job; refreshes the report when it finishes. */
export function useBenchmarkJob() {
  const qc = useQueryClient()
  return useQuery({
    queryKey: ['jobs', 'validation', 'benchmark'],
    queryFn: async () => {
      const jobs = unwrap(await api.GET('/jobs', { params: { query: { limit: 50 } } }))
      const job = jobs.find((j) => j.kind === 'benchmark') ?? null
      if (job && !isActiveJob(job)) void qc.invalidateQueries({ queryKey: qk.benchmark() })
      return job
    },
    refetchInterval: (q) => (isActiveJob(q.state.data ?? undefined) ? 3_000 : 60_000),
  })
}

// ------------------------------------------------------------ model cards (U13)
export function useModelCard(taxonKey: number, version: number | null | undefined) {
  return useQuery({
    queryKey: qk.modelCard(taxonKey, version ?? 0),
    // The endpoint also serves Markdown (?format=md); the JSON form is the generated type.
    queryFn: async (): Promise<ModelCardOut> =>
      unwrap(
        await api.GET('/species/{taxon_key}/models/{model_version}/exports/model-card', {
          params: { path: { taxon_key: taxonKey, model_version: version ?? 0 } },
        }),
      ) as ModelCardOut,
    enabled: !!version,
  })
}

// ------------------------------------------------------------ species lists (U9)
export function useSpeciesLists() {
  return useQuery({
    queryKey: qk.lists(),
    queryFn: async () => unwrap(await api.GET('/lists')),
    staleTime: 30_000,
  })
}

/** The registry restricted to one species list (null = the whole registry). */
export function useRegistry(listId: string | null) {
  const all = useSpeciesList()
  const filtered = useQuery({
    queryKey: qk.listSpecies(listId ?? ''),
    queryFn: async () =>
      unwrap(await api.GET('/species', { params: { query: { list_id: listId ?? '' } } })),
    enabled: !!listId,
    refetchInterval: 60_000,
  })
  return listId ? filtered : all
}

export function useCreateSpeciesList() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: async (body: SpeciesListIn) => unwrap(await api.POST('/lists', { body })),
    onSuccess: () => qc.invalidateQueries({ queryKey: qk.lists() }),
  })
}

export function useDeleteSpeciesList() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: async (listId: string) => {
      const res = await api.DELETE('/lists/{list_id}', { params: { path: { list_id: listId } } })
      if (!res.response.ok) unwrap(res)
    },
    onSuccess: () => qc.invalidateQueries({ queryKey: qk.lists() }),
  })
}

export function useRunSpeciesList() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: async ({ listId, body }: { listId: string; body: BatchRunRequest }) =>
      unwrap(await api.POST('/lists/{list_id}/runs', { params: { path: { list_id: listId } }, body })),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['jobs', 'batch'] }),
  })
}

/** Latest batch job (all species lists); refreshes the registry when it finishes. */
export function useBatchJob() {
  const qc = useQueryClient()
  return useQuery({
    queryKey: ['jobs', 'batch'],
    queryFn: async () => {
      const jobs = unwrap(await api.GET('/jobs', { params: { query: { limit: 50 } } }))
      const job = jobs.find((j) => j.kind === 'batch') ?? null
      if (job && !isActiveJob(job)) void qc.invalidateQueries({ queryKey: ['species'] })
      return job
    },
    refetchInterval: (q) => (isActiveJob(q.state.data ?? undefined) ? 3_000 : 60_000),
  })
}

// ------------------------------------------------------------ bias studies (B1, B9)
function useSpeciesReport<T>(key: readonly unknown[], fetcher: () => Promise<T | null>) {
  return useQuery({ queryKey: key, queryFn: fetcher, staleTime: 60_000 })
}

export function useTargetGroupComparison(taxonKey: number) {
  return useSpeciesReport(qk.targetGroups(taxonKey), async () => {
    const res = await api.GET('/species/{taxon_key}/target-group-comparison', {
      params: { path: { taxon_key: taxonKey } },
    })
    if (res.response.status === 404) return null
    return unwrap(res)
  })
}

export function useSharedBias(taxonKey: number) {
  return useSpeciesReport(qk.sharedBias(taxonKey), async () => {
    const res = await api.GET('/species/{taxon_key}/shared-bias', {
      params: { path: { taxon_key: taxonKey } },
    })
    if (res.response.status === 404) return null
    return unwrap(res)
  })
}

export function useRunTargetGroupComparison(taxonKey: number) {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: async () =>
      unwrap(
        await api.POST('/species/{taxon_key}/target-group-comparison', {
          params: { path: { taxon_key: taxonKey } },
        }),
      ),
    onSuccess: () => qc.invalidateQueries({ queryKey: qk.jobs(taxonKey) }),
  })
}

export function useRunSharedBias(taxonKey: number) {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: async () =>
      unwrap(
        await api.POST('/species/{taxon_key}/shared-bias', {
          params: { path: { taxon_key: taxonKey } },
        }),
      ),
    onSuccess: () => qc.invalidateQueries({ queryKey: qk.jobs(taxonKey) }),
  })
}
