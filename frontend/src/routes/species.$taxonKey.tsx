import { useQuery } from '@tanstack/react-query'
import { createFileRoute, Link, useNavigate } from '@tanstack/react-router'
import { useSelector } from '@tanstack/react-store'
import {
  ArrowLeft,
  ChevronRight,
  DatabaseZap,
  Gauge,
  GitBranch,
  Leaf,
  ListTree,
  Loader2,
  MapPinned,
  PenLine,
  Play,
  RefreshCw,
  SlidersHorizontal,
} from 'lucide-react'
import { useEffect, useRef } from 'react'
import { z } from 'zod'

import {
  speciesOptions,
  useBioclim,
  useLatestJob,
  useLayers,
  useModelVersion,
  useModelVersions,
  useNativeRange,
  useRunPipeline,
  useTimeline,
  useUpdateNativeRange,
} from '@/api/queries'
import type { NativeRangeOut } from '@/api/types'
import { CvFoldsChart } from '@/components/charts/CvFoldsChart'
import { OccurrenceTimelineChart } from '@/components/charts/OccurrenceTimelineChart'
import { VariableImportanceChart } from '@/components/charts/VariableImportanceChart'
import { DualMapView } from '@/components/maps/DualMapView'
import { NativeRangeEditor } from '@/components/maps/NativeRangeEditor'
import { BulletinCard } from '@/components/species/BulletinCard'
import { CaveatBanner } from '@/components/species/CaveatBanner'
import { CrossCheckCard } from '@/components/species/CrossCheckCard'
import { HiresCard } from '@/components/species/HiresCard'
import { JobStatusBar } from '@/components/species/JobStatusBar'
import { MetricsCards } from '@/components/species/MetricsCards'
import { ModelLineageTable } from '@/components/species/ModelLineageTable'
import { OccurrenceTable } from '@/components/species/OccurrenceTable'
import { ScenariosCard } from '@/components/species/ScenariosCard'
import { SeverityConfigForm } from '@/components/species/SeverityConfigForm'
import { SeverityPanel } from '@/components/species/SeverityPanel'
import { SourceSummary } from '@/components/species/SourceSummary'
import { StatusBadge } from '@/components/species/StatusBadge'
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Skeleton } from '@/components/ui/skeleton'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Tooltip, TooltipContent, TooltipTrigger } from '@/components/ui/tooltip'
import { mapActions, mapStore } from '@/store/mapStore'

const TABS = ['overview', 'native-range', 'severity', 'lineage', 'occurrences'] as const
type Tab = (typeof TABS)[number]

const searchSchema = z.object({ tab: z.enum(TABS).optional().catch(undefined) })

export const Route = createFileRoute('/species/$taxonKey')({
  params: {
    parse: (p) => ({ taxonKey: z.coerce.number().int().positive().parse(p.taxonKey) }),
    stringify: (p) => ({ taxonKey: String(p.taxonKey) }),
  },
  validateSearch: (search) => searchSchema.parse(search),
  loader: ({ context, params }) =>
    context.queryClient.ensureQueryData(speciesOptions(params.taxonKey)).catch(() => null),
  component: SpeciesWorkspace,
  pendingComponent: () => (
    <div className="space-y-4">
      <Skeleton className="h-20" />
      <Skeleton className="h-[560px]" />
    </div>
  ),
})

/** Centre the shared map viewport on the native range once per species. */
function useFitNativeRange(taxonKey: number, nr: NativeRangeOut | null | undefined) {
  const fitted = useRef<number | null>(null)
  useEffect(() => {
    if (!nr || fitted.current === taxonKey) return
    fitted.current = taxonKey
    let minX = 180, minY = 90, maxX = -180, maxY = -90
    for (const poly of nr.geometry.coordinates)
      for (const ring of poly)
        for (const [x, y] of ring as [number, number][]) {
          minX = Math.min(minX, x); maxX = Math.max(maxX, x)
          minY = Math.min(minY, y); maxY = Math.max(maxY, y)
        }
    const span = Math.max(maxX - minX, (maxY - minY) * 1.6, 1)
    mapActions.setViewState({
      longitude: (minX + maxX) / 2,
      latitude: (minY + maxY) / 2,
      zoom: Math.max(0.8, Math.min(6, Math.log2(360 / span) - 0.6)),
      pitch: 0,
      bearing: 0,
    })
  }, [taxonKey, nr])
}

function SpeciesWorkspace() {
  const { taxonKey } = Route.useParams()
  const tab: Tab = Route.useSearch().tab ?? 'overview'
  const navigate = useNavigate({ from: Route.fullPath })
  const setTab = (t: Tab) => void navigate({ search: { tab: t }, replace: true })

  const job = useLatestJob(taxonKey)
  const species = useQuery({
    ...speciesOptions(taxonKey),
    refetchInterval: job.active ? 3_000 : false,
  })
  const sp = species.data
  const registered = !!sp
  const nativeRange = useNativeRange(taxonKey, registered)
  const layers = useLayers(taxonKey, registered)
  const timeline = useTimeline(taxonKey, registered)
  const versions = useModelVersions(taxonKey, registered)
  const model = useModelVersion(taxonKey, sp?.model_version || null)
  const bioclim = useBioclim()
  const run = useRunPipeline(taxonKey)
  const updateNative = useUpdateNativeRange(taxonKey)
  const maxYear = useSelector(mapStore, (s) => s.maxYear)

  useEffect(() => mapActions.resetForSpecies(), [taxonKey])
  useFitNativeRange(taxonKey, nativeRange.data)

  if (species.isLoading) return <Skeleton className="h-96" />

  // ------------------------------------------------------------ not tracked yet
  if (!sp) {
    return (
      <div className="mx-auto max-w-2xl space-y-4 py-10">
        <Link to="/" className="inline-flex items-center gap-1 text-xs text-muted-foreground hover:text-foreground">
          <ArrowLeft className="size-3" /> Registry
        </Link>
        <Card>
          <CardHeader>
            <CardTitle className="text-lg">Start tracking GBIF taxon {taxonKey}</CardTitle>
            <CardDescription>
              This species is not in the registry yet. Starting fetches its GBIF occurrences (a
              DOI-backed download when GBIF credentials are configured, otherwise the search API)
              plus research-grade iNaturalist records, then proposes a draft native range for you
              to review. No model is trained until you confirm that range.
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <Button onClick={() => run.mutate({})} disabled={run.isPending || job.active}>
              {run.isPending || job.active ? <Loader2 className="animate-spin" /> : <Play />}
              Start analysis
            </Button>
            {run.isError && <p className="text-xs text-destructive">{run.error.message}</p>}
            <JobStatusBar job={job.latest} />
          </CardContent>
        </Card>
      </div>
    )
  }

  const current = sp.current_model
  const awaitingReview = sp.status === 'awaiting_native_range_review'

  return (
    <div className="space-y-5">
      {/* ---------------------------------------------------------------- header */}
      <div className="flex flex-wrap items-start gap-4">
        {sp.thumbnail_url ? (
          <img src={sp.thumbnail_url} alt="" className="size-16 rounded-xl object-cover shadow" />
        ) : (
          <span className="flex size-16 items-center justify-center rounded-xl bg-muted">
            <Leaf className="size-6 text-muted-foreground" />
          </span>
        )}
        <div className="min-w-0 flex-1">
          <div className="flex items-center gap-1 text-xs text-muted-foreground">
            <Link to="/" className="hover:text-foreground">Registry</Link>
            <ChevronRight className="size-3" />
            <span className="font-mono">taxon {sp.taxon_key}</span>
          </div>
          <h1 className="truncate text-2xl font-semibold italic tracking-tight">{sp.scientific_name}</h1>
          <div className="mt-1 flex flex-wrap items-center gap-2 text-sm text-muted-foreground">
            {sp.common_name && <span>{sp.common_name}</span>}
            <StatusBadge
              status={sp.status}
              pendingRecords={sp.n_pending_records}
              trainedTs={sp.model_trained_ts}
              jobActive={job.active}
            />
            <SourceSummary sp={sp} />
          </div>
        </div>
        <div className="flex gap-2">
          <Button variant="outline" size="sm" onClick={() => run.mutate({})} disabled={job.active || run.isPending}>
            <RefreshCw className={job.active ? 'animate-spin' : undefined} /> Check for new records
          </Button>
          <Tooltip>
            <TooltipTrigger asChild>
              <Button
                variant="ghost"
                size="sm"
                onClick={() => run.mutate({ fullResync: true })}
                disabled={job.active || run.isPending}
              >
                <DatabaseZap /> Re-sync all
              </Button>
            </TooltipTrigger>
            <TooltipContent>
              Re-fetch every GBIF/iNaturalist record (not only changes since the last check) and
              append any that are missing — e.g. after an interrupted or capped first ingest.
            </TooltipContent>
          </Tooltip>
          <Button variant="secondary" size="sm" onClick={() => run.mutate({ forceRetrain: true })} disabled={job.active || run.isPending || sp.native_range_status !== 'confirmed'}>
            <Play /> Force retrain
          </Button>
        </div>
      </div>

      <JobStatusBar job={job.latest} />

      {sp.status === 'failed' && sp.last_error && (
        <Alert variant="destructive">
          <AlertTitle>Last run failed</AlertTitle>
          <AlertDescription className="font-mono">{sp.last_error}</AlertDescription>
        </Alert>
      )}

      {awaitingReview && (
        <Alert variant="warning">
          <PenLine />
          <AlertTitle>Review the proposed native range before training</AlertTitle>
          <AlertDescription className="flex flex-wrap items-center gap-3">
            <span>
              A draft polygon was generated heuristically. Training is blocked until you confirm it.
            </span>
            <Button size="sm" variant="outline" onClick={() => setTab('native-range')}>
              Open native-range editor
            </Button>
          </AlertDescription>
        </Alert>
      )}

      {current && <CaveatBanner model={current} />}

      {/* ---------------------------------------------------------------- maps */}
      {layers.data ? (
        <DualMapView layers={layers.data} nativeRange={nativeRange.data} timeline={timeline.data} />
      ) : (
        <Skeleton className="h-[560px]" />
      )}

      {/* ---------------------------------------------------------------- tabs */}
      <Tabs value={tab} onValueChange={(v) => setTab(v as Tab)}>
        <TabsList className="flex-wrap">
          <TabsTrigger value="overview"><Gauge /> Overview</TabsTrigger>
          <TabsTrigger value="native-range"><MapPinned /> Native range</TabsTrigger>
          <TabsTrigger value="severity"><SlidersHorizontal /> Severity settings</TabsTrigger>
          <TabsTrigger value="lineage"><GitBranch /> Model lineage</TabsTrigger>
          <TabsTrigger value="occurrences"><ListTree /> Occurrences</TabsTrigger>
        </TabsList>

        <TabsContent value="overview" className="space-y-4">
          {model.data ? (
            <>
              <MetricsCards model={model.data} />
              <div className="grid gap-4 xl:grid-cols-[1.3fr_1fr]">
                {model.data.severity && <SeverityPanel severity={model.data.severity} />}
                <Card>
                  <CardHeader>
                    <CardTitle>Variable importance</CardTitle>
                    <CardDescription>
                      Permutation importance · predictors from the global VIF &lt; 5 screen
                    </CardDescription>
                  </CardHeader>
                  <CardContent>
                    <VariableImportanceChart
                      importance={model.data.variable_importance}
                      descriptions={bioclim.data?.descriptions}
                    />
                  </CardContent>
                </Card>
              </div>
            </>
          ) : (
            <Card>
              <CardContent className="py-8 text-center text-sm text-muted-foreground">
                Model diagnostics and the severity index appear after the first training run.
              </CardContent>
            </Card>
          )}
          <div className="grid gap-4 xl:grid-cols-[1.3fr_1fr]">
            <Card>
              <CardHeader>
                <CardTitle>Invasion timeline</CardTitle>
                <CardDescription>
                  Records per year by ingested label. Growth partly reflects observer effort.
                </CardDescription>
              </CardHeader>
              <CardContent>
                {timeline.data && timeline.data.years.length > 0 ? (
                  <OccurrenceTimelineChart timeline={timeline.data} highlightYear={maxYear} />
                ) : (
                  <Skeleton className="h-52" />
                )}
              </CardContent>
            </Card>
            <div className="space-y-4">
              {model.data && (
                <Card>
                  <CardHeader>
                    <CardTitle>Spatial block cross-validation</CardTitle>
                    <CardDescription>{model.data.cv_method}</CardDescription>
                  </CardHeader>
                  <CardContent>
                    <CvFoldsChart folds={model.data.folds} />
                  </CardContent>
                </Card>
              )}
              <BulletinCard taxonKey={taxonKey} jobActive={job.active} />
              {layers.data && sp.model_version > 0 && (
                <>
                  <ScenariosCard taxonKey={taxonKey} layers={layers.data} jobActive={job.active} />
                  <HiresCard taxonKey={taxonKey} layers={layers.data} jobActive={job.active} />
                </>
              )}
            </div>
          </div>
        </TabsContent>

        <TabsContent value="native-range">
          {nativeRange.data && layers.data ? (
            <NativeRangeEditor
              nativeRange={nativeRange.data}
              occurrences={layers.data.occurrences}
              saving={updateNative.isPending}
              onSave={(geometry, confirm) =>
                updateNative.mutate(
                  { geometry, confirm, train: confirm },
                  { onSuccess: () => void nativeRange.refetch() },
                )
              }
            />
          ) : (
            <Card>
              <CardContent className="py-8 text-center text-sm text-muted-foreground">
                A draft native range is proposed after occurrences have been ingested.
              </CardContent>
            </Card>
          )}
          {updateNative.isError && (
            <p className="mt-2 text-xs text-destructive">{updateNative.error.message}</p>
          )}
        </TabsContent>

        <TabsContent value="severity" className="grid gap-4 xl:grid-cols-2">
          <SeverityConfigForm key={JSON.stringify(sp.severity_config)} taxonKey={taxonKey} config={sp.severity_config} />
          {current?.severity ? (
            <SeverityPanel severity={current.severity} />
          ) : (
            <Card>
              <CardContent className="py-8 text-center text-sm text-muted-foreground">
                Weights are stored now and applied to the first trained model.
              </CardContent>
            </Card>
          )}
        </TabsContent>

        <TabsContent value="lineage" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Model lineage</CardTitle>
              <CardDescription>
                Every version with its trigger, spatial-CV metrics and full reproducibility record
                (bioclim version, GBIF DOI, pinned package versions, random seeds, MLflow run).
              </CardDescription>
            </CardHeader>
            <CardContent className="px-2 pb-2">
              <ModelLineageTable versions={versions.data} />
            </CardContent>
          </Card>
          {model.data && (
            <CrossCheckCard
              taxonKey={taxonKey}
              model={model.data}
              bioclim={bioclim.data}
              jobActive={job.active}
            />
          )}
        </TabsContent>

        <TabsContent value="occurrences">
          <OccurrenceTable taxonKey={taxonKey} />
        </TabsContent>
      </Tabs>
    </div>
  )
}
