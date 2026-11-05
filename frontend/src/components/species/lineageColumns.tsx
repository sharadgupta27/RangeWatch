/**
 * Model-lineage table columns (TanStack Table v9). Reproducibility metadata — bioclim
 * version, GBIF DOI/key, pinned package versions and random seeds — is rendered verbatim,
 * never summarised (CLAUDE.md "Reproducibility Requirements").
 */
import {
  createColumnHelper,
  createSortedRowModel,
  rowSortingFeature,
  sortFn_alphanumeric,
  sortFn_basic,
  sortFn_datetime,
  tableFeatures,
} from '@tanstack/react-table'

import type { ModelVersionOut } from '@/api/types'
import { Badge } from '@/components/ui/badge'
import { formatDate, formatNumber } from '@/lib/utils'

export const lineageFeatures = tableFeatures({
  rowSortingFeature,
  sortedRowModel: createSortedRowModel(),
  sortFns: { basic: sortFn_basic, datetime: sortFn_datetime, alphanumeric: sortFn_alphanumeric },
})

const helper = createColumnHelper<typeof lineageFeatures, ModelVersionOut>()

function Metric({ mean, std }: { mean?: number | null; std?: number | null }) {
  if (mean == null) return <span className="text-muted-foreground">–</span>
  return (
    <span className="tabular-nums">
      {mean.toFixed(3)}
      {std != null && <span className="text-muted-foreground"> ±{std.toFixed(2)}</span>}
    </span>
  )
}

/** Scalars as-is; lists and objects as compact JSON (verbatim, not summarised). */
function flatValues(values: Record<string, unknown>): Record<string, string> {
  return Object.fromEntries(
    Object.entries(values).map(([k, v]) => [
      k,
      v !== null && typeof v === 'object' ? JSON.stringify(v) : String(v),
    ]),
  )
}

function KeyValues({ values }: { values: Record<string, string | number> }) {
  return (
    <ul className="space-y-0.5 font-mono text-[11px] leading-tight">
      {Object.entries(values).map(([k, v]) => (
        <li key={k} className="whitespace-nowrap">
          <span className="text-muted-foreground">{k}</span> {String(v)}
        </li>
      ))}
    </ul>
  )
}

export const lineageColumns = helper.columns([
  helper.accessor('model_version', {
    header: 'Version',
    sortFn: 'basic',
    cell: (info) => <span className="font-semibold">v{info.getValue()}</span>,
  }),
  helper.accessor('trained_ts', {
    header: 'Trained',
    sortFn: 'datetime',
    cell: (info) => <span className="whitespace-nowrap">{formatDate(info.getValue(), true)}</span>,
  }),
  helper.accessor('model_type', {
    header: 'Model',
    cell: (info) => (
      <Badge variant={info.getValue() === 'B' ? 'default' : 'warning'}>
        {info.getValue() === 'B' ? 'B · native+invaded' : 'A · native only'}
      </Badge>
    ),
  }),
  helper.accessor('algorithm', {
    header: 'Algorithm',
    cell: (info) => {
      const r = info.row.original.reproducibility
      const a = r.algorithm
      return (
        <div className="min-w-40 font-mono text-[11px]">
          <div>{info.getValue()}</div>
          <div className="text-muted-foreground">threshold {info.row.original.threshold_rule}</div>
          {a?.fallback_reason && (
            <div className="max-w-64 text-warning">
              requested {a.requested}: {a.fallback_reason}
            </div>
          )}
          {a?.point_process && (
            <div className="max-w-64 break-all text-muted-foreground">
              point process {JSON.stringify(a.point_process)}
            </div>
          )}
          {r.pseudo_absences && (
            <div className="max-w-64 break-all text-muted-foreground">
              pseudo-absences {r.pseudo_absences.method} · n {r.pseudo_absences.n} · seed{' '}
              {r.pseudo_absences.seed}
              {r.pseudo_absences.fallback_reason && ` · fallback: ${r.pseudo_absences.fallback_reason}`}
            </div>
          )}
        </div>
      )
    },
  }),
  helper.accessor('trigger', {
    header: 'Trigger',
    enableSorting: false,
    cell: (info) => <span className="text-xs text-muted-foreground">{info.getValue()}</span>,
  }),
  helper.accessor('auc_mean', {
    header: 'AUC (CV)',
    sortFn: 'basic',
    cell: (info) => <Metric mean={info.getValue()} std={info.row.original.auc_std} />,
  }),
  helper.accessor('cbi_mean', {
    header: 'CBI (CV)',
    sortFn: 'basic',
    cell: (info) => <Metric mean={info.getValue()} std={info.row.original.cbi_std} />,
  }),
  helper.accessor('tss_mean', {
    header: 'TSS (CV)',
    sortFn: 'basic',
    cell: (info) => <Metric mean={info.getValue()} std={info.row.original.tss_std} />,
  }),
  helper.accessor('n_presence', {
    header: 'Presences / background',
    sortFn: 'basic',
    cell: (info) => (
      <span className="tabular-nums">
        {formatNumber(info.getValue())} / {formatNumber(info.row.original.n_background)}
      </span>
    ),
  }),
  helper.accessor((row) => row.reproducibility.background?.method ?? '', {
    id: 'background',
    header: 'Background sampling',
    sortFn: 'alphanumeric',
    cell: (info) => {
      const bg = info.row.original.reproducibility.background
      if (!bg) {
        return (
          <span className="text-[11px] text-muted-foreground">
            not recorded (trained before background logging)
          </span>
        )
      }
      const tg = bg.target_group
      return (
        <div className="min-w-56 max-w-72 font-mono text-[11px]">
          <div>{bg.method}</div>
          <div className="text-muted-foreground">
            requested {bg.requested_method} · buffer {bg.buffer_km} km
          </div>
          {tg && (
            <div className="mt-1 text-muted-foreground">
              <div>
                target group {tg.rank} {tg.name} (taxonKey {tg.taxon_key})
              </div>
              <div>
                {tg.n_records.toLocaleString()} records · {tg.n_pixels.toLocaleString()} pixels ·
                z{tg.map_zoom} ({tg.pixel_deg}°) · {tg.n_tiles} tiles
              </div>
              <div className="break-all">{tg.query}</div>
              <div>fetched {tg.fetched_ts}</div>
              {tg.weighting && (
                <div>
                  weighting {tg.weighting}
                  {tg.n_background_cells != null &&
                    ` · ${tg.n_background_cells.toLocaleString()} distinct cells`}
                </div>
              )}
              {tg.matched && <div className="break-all">matched {JSON.stringify(tg.matched)}</div>}
            </div>
          )}
          {bg.fallback_reason && (
            <div className="mt-1 whitespace-normal text-warning">fallback: {bg.fallback_reason}</div>
          )}
        </div>
      )
    },
  }),
  helper.accessor('severity_score', {
    header: 'Severity',
    sortFn: 'basic',
    cell: (info) => formatNumber(info.getValue()),
  }),
  helper.accessor((row) => row.reproducibility.bioclim_version_used, {
    id: 'bioclim_version_used',
    header: 'Bioclim version',
    sortFn: 'alphanumeric',
    cell: (info) => (
      <div className="max-w-44 font-mono text-[11px]">
        <div>{info.getValue()}</div>
        <div className="break-all text-muted-foreground">
          sha256 {info.row.original.reproducibility.bioclim_sha256 ?? 'n/a'}
        </div>
        <div className="break-all text-muted-foreground">
          dvc {info.row.original.reproducibility.bioclim_dvc_md5 ?? 'not tracked'}
        </div>
      </div>
    ),
  }),
  helper.accessor(
    (row) => row.reproducibility.gbif_download_doi ?? row.reproducibility.gbif_download_key ?? '',
    {
      id: 'gbif_download',
      header: 'GBIF download',
      enableSorting: false,
      cell: (info) => {
        const r = info.row.original.reproducibility
        return (
          <div className="min-w-56 font-mono text-[11px]">
            {r.gbif_download_doi ? (
              <a
                className="text-primary hover:underline"
                href={`https://doi.org/${r.gbif_download_doi}`}
                target="_blank"
                rel="noreferrer"
              >
                doi:{r.gbif_download_doi}
              </a>
            ) : (
              <span>key: {r.gbif_download_key ?? 'search API (no DOI)'}</span>
            )}
            <div className="max-w-72 whitespace-normal text-muted-foreground">{r.gbif_citation}</div>
          </div>
        )
      },
    },
  ),
  helper.accessor((row) => row.reproducibility.training_data?.sha256 ?? '', {
    id: 'training_data',
    header: 'Training data',
    enableSorting: false,
    cell: (info) => {
      const r = info.row.original.reproducibility
      if (!r.training_data) {
        return <span className="text-[11px] text-muted-foreground">not captured (trained before snapshots)</span>
      }
      return (
        <div className="max-w-56 font-mono text-[11px]">
          {[r.training_data, r.occurrence_manifest, r.data_quality?.snapshot].map(
            (f) =>
              f && (
                <div key={f.path} className="mb-1">
                  <div className="break-all">{f.path}</div>
                  <div className="text-muted-foreground">
                    {f.n_rows.toLocaleString()} rows · {(f.bytes / 1024).toFixed(0)} KiB
                  </div>
                  <div className="break-all text-muted-foreground">sha256 {f.sha256}</div>
                </div>
              ),
          )}
        </div>
      )
    },
  }),
  helper.accessor((row) => row.reproducibility.data_quality?.n_excluded ?? -1, {
    id: 'data_quality',
    header: 'Data quality',
    sortFn: 'basic',
    cell: (info) => {
      const dq = info.row.original.reproducibility.data_quality
      if (!dq) {
        return (
          <span className="text-[11px] text-muted-foreground">
            not assessed (trained before the data-quality tests)
          </span>
        )
      }
      const ref = dq.reference as
        | { version?: string; tables?: Record<string, { n_rows?: number; sha256?: string | null }> }
        | null
        | undefined
      return (
        <div className="min-w-64 max-w-80 font-mono text-[11px]">
          <div>
            {dq.version} · {dq.n_records.toLocaleString()} assessed · {dq.n_excluded.toLocaleString()}{' '}
            excluded · {dq.n_flagged_kept.toLocaleString()} flagged kept
          </div>
          <div className="text-muted-foreground">
            exclude: {(dq.config.exclude as string[] | undefined)?.join(', ') || 'none'}
          </div>
          <ul className="mt-1 text-muted-foreground">
            {dq.tests.map((t) => (
              <li key={t.id} className="whitespace-nowrap">
                {t.id} {t.status === 'skipped' ? 'skipped' : t.n_flagged.toLocaleString()}
              </li>
            ))}
          </ul>
          <div className="mt-1 text-muted-foreground">
            {ref?.version ? `references ${ref.version}` : 'no reference gazetteers'}
          </div>
          {Object.entries(ref?.tables ?? {}).map(([name, t]) => (
            <div key={name} className="break-all text-muted-foreground">
              {name} {t.n_rows?.toLocaleString()} rows · sha256 {t.sha256 ?? 'n/a'}
            </div>
          ))}
        </div>
      )
    },
  }),
  helper.accessor((row) => row.reproducibility.bias_covariates?.method ?? '', {
    id: 'sampling_bias',
    header: 'Sampling bias',
    enableSorting: false,
    cell: (info) => {
      const r = info.row.original.reproducibility
      const w = r.presence_weights
      const b = r.bias_covariates
      if (!w && !b) {
        return (
          <span className="text-[11px] text-muted-foreground">
            not recorded (trained before bias options)
          </span>
        )
      }
      return (
        <div className="min-w-56 max-w-80 font-mono text-[11px]">
          <div>weights {w?.method ?? 'none'}</div>
          {w && w.method !== 'none' && (
            <div className="text-muted-foreground">
              {w.n_presences?.toLocaleString()} presences · min {w.min} · median {w.median} · max{' '}
              {w.max}
              {w.tier_weights && ` · tiers ${JSON.stringify(w.tier_weights)}`}
              {w.density_neighbors != null && ` · k=${w.density_neighbors}`}
            </div>
          )}
          <div className="mt-1">covariate {b?.method ?? 'none'}</div>
          {b && b.method !== 'none' && (
            <div className="break-all text-muted-foreground">
              {b.status === 'used'
                ? `${b.covariate} = ${b.transform}; held at ${b.condition} = ${b.conditioning_value}${
                    b.importance ? ` · importance ${JSON.stringify(b.importance)}` : ''
                  }`
                : `unavailable: ${b.reason}`}
              {b.query && <div>{b.query}</div>}
              {b.note && <div>{b.note}</div>}
            </div>
          )}
          {b?.static_layers?.map((s) => (
            <div key={s.name} className="mt-1 break-all">
              layer {s.name}{' '}
              <span className="text-muted-foreground">
                {s.status === 'used'
                  ? `${s.covariate} (${s.direction}, ${s.transform}); held at ${s.condition} = ${s.conditioning_value} · ${s.n_missing_filled} filled · sha256 ${s.sha256}`
                  : `unavailable: ${s.reason}`}
              </span>
            </div>
          ))}
        </div>
      )
    },
  }),
  helper.accessor((row) => row.reproducibility.thinning?.method ?? 'distance', {
    id: 'thinning',
    header: 'Thinning',
    enableSorting: false,
    cell: (info) => {
      const t = info.row.original.reproducibility.thinning
      if (!t) {
        return (
          <span className="text-[11px] text-muted-foreground">
            not recorded (fixed-distance thinning, see training config)
          </span>
        )
      }
      return (
        <div className="min-w-48 max-w-80 font-mono text-[11px]">
          <div>
            {t.method}
            {t.distance_km != null && ` · ${t.distance_km} km`}
            {t.reps != null && t.reps > 1 && ` · best of ${t.reps} (seed ${t.selected_seed})`}
          </div>
          <div className="text-muted-foreground">
            {t.n_before.toLocaleString()} → {t.n_after.toLocaleString()} presences · seed {t.seed}
          </div>
          {t.autocorrelation && (
            <div className="break-all text-muted-foreground">
              variogram {JSON.stringify(t.autocorrelation)}
            </div>
          )}
          {t.n_bins != null && (
            <div className="text-muted-foreground">
              {t.n_bins} bins × {t.n_axes} PCA axes · variance {JSON.stringify(t.variance_explained)} ·{' '}
              {t.n_occupied_cells} occupied cells
            </div>
          )}
          {t.reference && <div className="text-muted-foreground">{t.reference}</div>}
          {t.fallback_reason && <div className="whitespace-normal text-warning">{t.fallback_reason}</div>}
        </div>
      )
    },
  }),
  helper.accessor((row) => row.reproducibility.training_uploads?.length ?? 0, {
    id: 'training_uploads',
    header: 'Training uploads',
    sortFn: 'basic',
    cell: (info) => {
      const uploads = info.row.original.reproducibility.training_uploads ?? []
      if (!uploads.length) return <span className="text-[11px] text-muted-foreground">none</span>
      return (
        <ul className="min-w-56 max-w-80 space-y-1 font-mono text-[11px]">
          {uploads.map((u) => (
            <li key={u.dataset_id}>
              <div>
                {u.name} ({u.dataset_id})
              </div>
              {u.used_as === 'survey' ? (
                <div className="text-muted-foreground">
                  survey: {u.n_survey_presences?.toLocaleString()} presences ·{' '}
                  {u.n_survey_absences?.toLocaleString()} absences (integrated model)
                </div>
              ) : (
                <div className="text-muted-foreground">
                  {u.n_added.toLocaleString()} of {u.n_presences.toLocaleString()} presences added ·{' '}
                  {u.n_overlap_excluded.toLocaleString()} duplicates left out ·{' '}
                  {u.n_absences_not_used.toLocaleString()} absences not used
                </div>
              )}
              {u.file_sha256 && <div className="break-all text-muted-foreground">sha256 {u.file_sha256}</div>}
            </li>
          ))}
        </ul>
      )
    },
  }),
  helper.accessor((row) => row.reproducibility.annotations?.n_records_annotated ?? 0, {
    id: 'annotations',
    header: 'Annotations',
    sortFn: 'basic',
    cell: (info) => {
      const a = info.row.original.reproducibility.annotations
      if (!a) {
        return (
          <span className="text-[11px] text-muted-foreground">
            not recorded (trained before annotations)
          </span>
        )
      }
      if (!a.n_records_annotated) return <span className="text-[11px] text-muted-foreground">none</span>
      return (
        <div className="min-w-48 max-w-72 font-mono text-[11px]">
          <div>
            {a.n_records_annotated.toLocaleString()} records · {a.n_annotations.toLocaleString()}{' '}
            annotations
          </div>
          <div className="text-muted-foreground">
            excluded {a.n_excluded_suspicious} suspicious + {a.n_excluded_cultivated} cultivated ·
            kept {a.n_kept_verified} verified · {a.n_label_overrides} labels overridden
          </div>
          {a.snapshot && (
            <div className="break-all text-muted-foreground">sha256 {a.snapshot.sha256}</div>
          )}
        </div>
      )
    },
  }),
  helper.accessor((row) => row.reproducibility.study_region?.resolution ?? 'global', {
    id: 'study_region',
    header: 'Study region',
    enableSorting: false,
    cell: (info) => {
      const r = info.row.original.reproducibility.study_region
      if (!r) return <span className="text-[11px] text-muted-foreground">global</span>
      return (
        <div className="min-w-40 font-mono text-[11px]">
          <div>[{r.bbox.join(', ')}]</div>
          <div className="text-muted-foreground">
            {r.resolution} · {r.n_cells.toLocaleString()} cells · {r.n_records_outside} records
            outside
          </div>
        </div>
      )
    },
  }),
  helper.accessor((row) => row.reproducibility.training_config, {
    id: 'training_config',
    header: 'Training config',
    enableSorting: false,
    cell: (info) => <KeyValues values={flatValues(info.getValue())} />,
  }),
  helper.accessor((row) => row.reproducibility.package_versions, {
    id: 'package_versions',
    header: 'Package versions',
    enableSorting: false,
    cell: (info) => <KeyValues values={info.getValue()} />,
  }),
  helper.accessor((row) => row.reproducibility.random_seeds, {
    id: 'random_seeds',
    header: 'Random seeds',
    enableSorting: false,
    cell: (info) => <KeyValues values={info.getValue()} />,
  }),
  helper.accessor('mlflow_run_id', {
    header: 'MLflow run',
    enableSorting: false,
    cell: (info) => (
      <span className="font-mono text-[11px] text-muted-foreground">{info.getValue() ?? '–'}</span>
    ),
  }),
])
