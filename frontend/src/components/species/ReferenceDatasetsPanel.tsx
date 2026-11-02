import { useForm } from '@tanstack/react-form'
import { FlaskConical, Loader2, RefreshCw, Trash2, Upload } from 'lucide-react'
import { useRef } from 'react'

import { useDeleteDataset, useEvaluateDatasets, useReferenceDatasets, useUploadDataset } from '@/api/queries'
import type { IndependentMetricsOut, ReferenceDatasetOut } from '@/api/types'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Skeleton } from '@/components/ui/skeleton'
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table'
import { Tooltip, TooltipContent, TooltipTrigger } from '@/components/ui/tooltip'
import { formatDate, formatNumber } from '@/lib/utils'

import {
  ACCEPTED_UPLOAD_TYPES,
  COLUMN_FIELDS,
  datasetUploadSchema,
  EMPTY_UPLOAD,
  type DatasetUploadForm,
} from './datasetSchema'

const rate = (v: number | null | undefined) => (v == null ? '–' : `${(v * 100).toFixed(0)}%`)
const num = (v: number | null | undefined, d = 2) => (v == null ? '–' : v.toFixed(d))

function firstError(errors: unknown[]): string | null {
  const e = errors[0] as { message?: string } | string | undefined
  if (!e) return null
  return typeof e === 'string' ? e : (e.message ?? null)
}

/**
 * Independent data: surveys a user uploads and absences GBIF reports. They never train the
 * model; each model version is scored on them, so the table shows whether retraining helped
 * on records the models never saw.
 */
export function ReferenceDatasetsPanel({ taxonKey, jobActive }: { taxonKey: number; jobActive: boolean }) {
  const datasets = useReferenceDatasets(taxonKey)
  const evaluate = useEvaluateDatasets(taxonKey)
  return (
    <div className="space-y-4">
      <UploadCard taxonKey={taxonKey} />
      <Card>
        <CardHeader className="flex flex-row flex-wrap items-start justify-between gap-3">
          <div className="space-y-1.5">
            <CardTitle className="flex items-center gap-2">
              <FlaskConical className="size-4" /> Independent evaluation
            </CardTitle>
            <CardDescription className="max-w-3xl">
              Every model version scored on each dataset. Uploaded records that duplicate a
              GBIF/iNaturalist record (same place, same day) are left out, because the model may
              have been trained on them. Sensitivity is the share of presences predicted
              suitable; CBI compares them with the model&apos;s own background; absences add
              specificity, AUC and TSS.
            </CardDescription>
          </div>
          <Button
            variant="outline"
            size="sm"
            onClick={() => evaluate.mutate()}
            disabled={jobActive || evaluate.isPending || !datasets.data?.length}
          >
            <RefreshCw className={evaluate.isPending ? 'animate-spin' : undefined} /> Re-evaluate all
            versions
          </Button>
        </CardHeader>
        <CardContent className="space-y-6">
          {evaluate.isError && <p className="text-xs text-destructive">{evaluate.error.message}</p>}
          {datasets.isLoading ? (
            <Skeleton className="h-32" />
          ) : !datasets.data?.length ? (
            <p className="py-6 text-center text-sm text-muted-foreground">
              No independent data yet. Upload a survey above; explicit absences reported to GBIF
              appear here automatically after the next check for new records.
            </p>
          ) : (
            datasets.data.map((ds) => <DatasetSection key={ds.dataset_id} taxonKey={taxonKey} ds={ds} />)
          )}
        </CardContent>
      </Card>
    </div>
  )
}

function DatasetSection({ taxonKey, ds }: { taxonKey: number; ds: ReferenceDatasetOut }) {
  const remove = useDeleteDataset(taxonKey)
  const report = ds.parse_report as
    | { columns?: Record<string, string | null>; skipped?: Record<string, number>; crs?: string }
    | null
    | undefined
  const skipped = Object.entries(report?.skipped ?? {})
  const columns = Object.entries(report?.columns ?? {}).filter(([, v]) => v)
  return (
    <section className="space-y-2">
      <div className="flex flex-wrap items-center gap-2">
        <h3 className="font-medium">{ds.name}</h3>
        <Badge variant="secondary">{ds.kind === 'upload' ? 'Upload' : 'GBIF absences'}</Badge>
        <Badge variant="secondary">
          {formatNumber(ds.n_present)} present · {formatNumber(ds.n_absent)} absent
        </Badge>
        {ds.overlap && (
          <Tooltip>
            <TooltipTrigger asChild>
              <Badge variant={ds.overlap.fraction > 0.5 ? 'warning' : 'secondary'} className="cursor-help">
                {rate(ds.overlap.fraction)} also in GBIF/iNat
              </Badge>
            </TooltipTrigger>
            <TooltipContent className="max-w-xs">
              {formatNumber(ds.overlap.n_overlap)} of {formatNumber(ds.overlap.n_checked)} records
              have {ds.overlap.rule} ({ds.overlap.distance_m} m, ±{ds.overlap.max_days} day).
              Checked {formatDate(ds.overlap.checked_ts, true)}.
            </TooltipContent>
          </Tooltip>
        )}
        {ds.kind === 'upload' && (
          <Button
            size="icon"
            variant="ghost"
            className="ml-auto size-7"
            aria-label={`Remove ${ds.name}`}
            disabled={remove.isPending}
            onClick={() => {
              if (window.confirm(`Remove "${ds.name}" and its records? Scores already stored with model versions stay in their lineage.`))
                remove.mutate(ds.dataset_id)
            }}
          >
            <Trash2 />
          </Button>
        )}
      </div>
      <p className="text-[11px] text-muted-foreground">
        {[
          ds.citation,
          ds.license && `licence ${ds.license}`,
          ds.file_name && `${ds.file_name} (sha256 ${ds.file_sha256?.slice(0, 12)}…)`,
          columns.length > 0 && `columns: ${columns.map(([k, v]) => `${k} = ${v}`).join(', ')}`,
          skipped.length > 0 && `skipped: ${skipped.map(([k, v]) => `${v} ${k}`).join(', ')}`,
          report?.crs && `CRS ${report.crs}`,
        ]
          .filter(Boolean)
          .join(' · ')}
      </p>
      {ds.note && <p className="text-xs text-muted-foreground">{ds.note}</p>}
      {remove.isError && <p className="text-xs text-destructive">{remove.error.message}</p>}
      {!ds.evaluations?.length ? (
        <p className="text-xs text-muted-foreground">
          Not evaluated yet. Scores appear when the evaluation job finishes (or after the next
          training run).
        </p>
      ) : (
        <EvaluationTable ds={ds} />
      )}
    </section>
  )
}

export function EvaluationTable({ ds }: { ds: ReferenceDatasetOut }) {
  const evaluations = ds.evaluations ?? []
  const hasAbsences = evaluations.some((e) => e.metrics.n_absence > 0)
  const hasPresences = evaluations.some((e) => e.metrics.n_presence > 0)
  return (
    <div className="overflow-x-auto">
      <Table>
        <TableHeader>
          <TableRow>
            <TableHead>Version</TableHead>
            <TableHead>Trained</TableHead>
            {hasPresences && (
              <>
                <TableHead className="text-right">Presences</TableHead>
                <TableHead className="text-right">Sensitivity</TableHead>
                <TableHead className="text-right">
                  <Tooltip>
                    <TooltipTrigger className="cursor-help underline decoration-dotted">
                      Analog / extrapolated
                    </TooltipTrigger>
                    <TooltipContent className="max-w-xs">
                      Sensitivity at presences inside the training climate (MESS ≥ 0) vs in
                      extrapolated cells. A large gap means the extrapolation layer marks where
                      the model fails.
                    </TooltipContent>
                  </Tooltip>
                </TableHead>
                <TableHead className="text-right">CBI</TableHead>
              </>
            )}
            {hasAbsences && (
              <>
                <TableHead className="text-right">Absences</TableHead>
                <TableHead className="text-right">Specificity</TableHead>
                <TableHead className="text-right">AUC</TableHead>
                <TableHead className="text-right">TSS</TableHead>
              </>
            )}
            <TableHead className="text-right">Left out</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {evaluations.map((e) => (
            <MetricsRow
              key={e.model_version}
              version={e.model_version}
              modelType={e.model_type}
              trained={e.trained_ts}
              m={e.metrics}
              excluded={e.n_excluded_overlap}
              hasPresences={hasPresences}
              hasAbsences={hasAbsences}
            />
          ))}
        </TableBody>
      </Table>
    </div>
  )
}

function MetricsRow(props: {
  version: number
  modelType: string | null | undefined
  trained: string | null | undefined
  m: IndependentMetricsOut
  excluded: number
  hasPresences: boolean
  hasAbsences: boolean
}) {
  const { m } = props
  return (
    <TableRow>
      <TableCell className="font-mono">
        v{props.version}
        {props.modelType && <span className="ml-1 text-muted-foreground">({props.modelType})</span>}
      </TableCell>
      <TableCell className="text-xs text-muted-foreground">{formatDate(props.trained)}</TableCell>
      {props.hasPresences && (
        <>
          <TableCell className="text-right tabular-nums">{formatNumber(m.n_presence)}</TableCell>
          <TableCell className="text-right tabular-nums">{rate(m.sensitivity)}</TableCell>
          <TableCell className="text-right tabular-nums">
            {rate(m.sensitivity_analog)} / {rate(m.sensitivity_extrapolated)}
          </TableCell>
          <TableCell className="text-right tabular-nums">{num(m.cbi)}</TableCell>
        </>
      )}
      {props.hasAbsences && (
        <>
          <TableCell className="text-right tabular-nums">{formatNumber(m.n_absence)}</TableCell>
          <TableCell className="text-right tabular-nums">{rate(m.specificity)}</TableCell>
          <TableCell className="text-right tabular-nums">{num(m.auc_presence_absence)}</TableCell>
          <TableCell className="text-right tabular-nums">{num(m.tss)}</TableCell>
        </>
      )}
      <TableCell className="text-right tabular-nums text-muted-foreground">
        {formatNumber(props.excluded + m.n_outside_grid)}
      </TableCell>
    </TableRow>
  )
}

function UploadCard({ taxonKey }: { taxonKey: number }) {
  const upload = useUploadDataset(taxonKey)
  const fileInput = useRef<HTMLInputElement>(null)
  const form = useForm({
    defaultValues: EMPTY_UPLOAD as DatasetUploadForm,
    validators: { onSubmit: datasetUploadSchema },
    onSubmit: async ({ value, formApi }) => {
      const { file, ...fields } = value
      if (!file) return
      await upload.mutateAsync({ file, fields })
      formApi.reset()
      if (fileInput.current) fileInput.current.value = ''
    },
  })
  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Upload className="size-4" /> Upload an independent survey
        </CardTitle>
        <CardDescription>
          CSV/TSV with decimal-degree coordinates, GeoJSON, GeoPackage or a zipped shapefile
          (reprojected from its CRS). Columns are detected automatically; rows of other species
          are dropped when the file has a species column. The data is used for evaluation only,
          never for training.
        </CardDescription>
      </CardHeader>
      <CardContent>
        <form
          className="space-y-4"
          onSubmit={(e) => {
            e.preventDefault()
            void form.handleSubmit()
          }}
        >
          <div className="grid gap-4 md:grid-cols-2">
            <form.Field name="file">
              {(field) => (
                <div className="space-y-2">
                  <Label htmlFor="dataset-file">File</Label>
                  <Input
                    id="dataset-file"
                    ref={fileInput}
                    type="file"
                    accept={ACCEPTED_UPLOAD_TYPES}
                    onChange={(e) => field.handleChange(e.target.files?.[0] ?? null)}
                    aria-invalid={field.state.meta.errors.length > 0}
                  />
                  {firstError(field.state.meta.errors) && (
                    <p className="text-xs text-destructive">{firstError(field.state.meta.errors)}</p>
                  )}
                </div>
              )}
            </form.Field>
            <form.Field name="name">
              {(field) => (
                <div className="space-y-2">
                  <Label htmlFor={field.name}>Dataset name</Label>
                  <Input
                    id={field.name}
                    value={field.state.value}
                    onChange={(e) => field.handleChange(e.target.value)}
                    onBlur={field.handleBlur}
                    placeholder="e.g. INPA field surveys 2016–2023"
                    aria-invalid={field.state.meta.errors.length > 0}
                  />
                  {firstError(field.state.meta.errors) && (
                    <p className="text-xs text-destructive">{firstError(field.state.meta.errors)}</p>
                  )}
                </div>
              )}
            </form.Field>
            {(
              [
                ['citation', 'Citation', 'Who collected it, and how to cite it'],
                ['license', 'Licence', 'e.g. CC BY 4.0, or the data-sharing agreement'],
              ] as const
            ).map(([name, label, placeholder]) => (
              <form.Field key={name} name={name}>
                {(field) => (
                  <div className="space-y-2">
                    <Label htmlFor={field.name}>{label}</Label>
                    <Input
                      id={field.name}
                      value={field.state.value}
                      onChange={(e) => field.handleChange(e.target.value)}
                      placeholder={placeholder}
                    />
                  </div>
                )}
              </form.Field>
            ))}
          </div>
          <form.Field name="note">
            {(field) => (
              <div className="space-y-2">
                <Label htmlFor={field.name}>Notes</Label>
                <Input
                  id={field.name}
                  value={field.state.value}
                  onChange={(e) => field.handleChange(e.target.value)}
                  placeholder="Survey method, effort, known limitations"
                />
              </div>
            )}
          </form.Field>
          <details className="rounded-lg border px-3 py-2 text-sm">
            <summary className="cursor-pointer text-muted-foreground">
              Column names (only if detection picks the wrong ones)
            </summary>
            <div className="mt-3 grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
              {COLUMN_FIELDS.map(([name, label]) => (
                <form.Field key={name} name={name}>
                  {(field) => (
                    <div className="space-y-1">
                      <Label htmlFor={field.name} className="text-xs">
                        {label}
                      </Label>
                      <Input
                        id={field.name}
                        value={field.state.value}
                        onChange={(e) => field.handleChange(e.target.value)}
                        placeholder="auto"
                      />
                    </div>
                  )}
                </form.Field>
              ))}
            </div>
          </details>
          <form.Subscribe selector={(s) => [s.canSubmit, s.isSubmitting] as const}>
            {([canSubmit, isSubmitting]) => (
              <div className="flex flex-wrap items-center gap-2">
                <Button type="submit" size="sm" disabled={!canSubmit || isSubmitting}>
                  {isSubmitting ? <Loader2 className="animate-spin" /> : <Upload />} Upload &amp;
                  evaluate
                </Button>
                {upload.isError && <span className="text-xs text-destructive">{upload.error.message}</span>}
                {upload.isSuccess && !isSubmitting && (
                  <span className="text-xs text-success">
                    Uploaded {formatNumber(upload.data.dataset.n_present + upload.data.dataset.n_absent)}{' '}
                    records — checking independence and scoring every model version.
                  </span>
                )}
              </div>
            )}
          </form.Subscribe>
        </form>
      </CardContent>
    </Card>
  )
}
