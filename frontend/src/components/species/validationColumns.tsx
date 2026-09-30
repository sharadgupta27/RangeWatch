import { Link } from '@tanstack/react-router'
import {
  createColumnHelper,
  createSortedRowModel,
  rowSortingFeature,
  sortFn_alphanumeric,
  sortFn_basic,
  tableFeatures,
} from '@tanstack/react-table'

import type { ValidationSpecies } from '@/api/types'
import { Badge } from '@/components/ui/badge'
import { formatNumber } from '@/lib/utils'

export const validationFeatures = tableFeatures({
  rowSortingFeature,
  sortedRowModel: createSortedRowModel(),
  sortFns: { basic: sortFn_basic, alphanumeric: sortFn_alphanumeric },
})

const h = createColumnHelper<typeof validationFeatures, ValidationSpecies>()

const fmt = (v?: number | null) => (v == null ? '–' : v.toFixed(3))

export const validationColumns = h.columns([
  h.accessor('scientific_name', {
    header: 'Species',
    sortFn: 'alphanumeric',
    cell: (i) => (
      <Link
        to="/species/$taxonKey"
        params={{ taxonKey: i.row.original.taxon_key }}
        className="font-medium italic hover:underline"
      >
        {i.getValue()}
      </Link>
    ),
  }),
  h.accessor('passed', {
    header: 'Result',
    cell: (i) => {
      const r = i.row.original
      if (r.status === 'error') return <Badge variant="destructive">error</Badge>
      if (r.status === 'no_model') return <Badge variant="outline">no model</Badge>
      return i.getValue() ? (
        <Badge variant="success">pass</Badge>
      ) : (
        <Badge variant="warning">review</Badge>
      )
    },
  }),
  h.accessor('model_version', {
    header: 'Model',
    cell: (i) => {
      const r = i.row.original
      return r.model_version ? `v${r.model_version} · ${r.model_type} · ${r.bioclim_version}` : '–'
    },
  }),
  h.accessor('n_occurrences', {
    header: 'Records',
    sortFn: 'basic',
    cell: (i) => formatNumber(i.getValue()),
  }),
  h.accessor('cbi_mean', { header: 'CBI (CV)', sortFn: 'basic', cell: (i) => fmt(i.getValue()) }),
  h.accessor('auc_mean', { header: 'AUC (CV)', sortFn: 'basic', cell: (i) => fmt(i.getValue()) }),
  h.accessor((r) => r.invaded_captured ?? -1, {
    id: 'invaded',
    header: 'Invaded regions captured',
    sortFn: 'basic',
    cell: (i) => {
      const r = i.row.original
      return r.invaded_total ? `${r.invaded_captured}/${r.invaded_total}` : '–'
    },
  }),
  h.accessor((r) => r.controls_correct ?? -1, {
    id: 'controls',
    header: 'Controls unsuitable',
    sortFn: 'basic',
    cell: (i) => {
      const r = i.row.original
      return r.controls_total ? `${r.controls_correct}/${r.controls_total}` : '–'
    },
  }),
  h.accessor((r) => r.transferability?.auc ?? -1, {
    id: 'transfer_auc',
    header: 'Native-only → invaded AUC',
    sortFn: 'basic',
    cell: (i) => {
      const t = i.row.original.transferability
      if (!t) return '–'
      if (t.skipped) return <span className="text-xs text-muted-foreground">skipped</span>
      const value = t.auc ?? 0
      return (
        <span className="tabular-nums">
          <span className={value >= (t.literature_mean_auc ?? 0.7) ? 'text-success' : 'text-warning'}>
            {fmt(t.auc)}
          </span>
          <span className="text-xs text-muted-foreground"> (n={t.n_introduced_test})</span>
        </span>
      )
    },
  }),
])
