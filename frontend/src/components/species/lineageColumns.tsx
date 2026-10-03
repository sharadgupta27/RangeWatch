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
          {[r.training_data, r.occurrence_manifest].map(
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
