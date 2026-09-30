import { Link } from '@tanstack/react-router'
import {
  createColumnHelper,
  createSortedRowModel,
  rowSortingFeature,
  sortFn_alphanumeric,
  sortFn_basic,
  sortFn_datetime,
  tableFeatures,
  useTable,
} from '@tanstack/react-table'
import { ArrowDown, ArrowUp, Leaf } from 'lucide-react'

import type { SpeciesSummary } from '@/api/types'
import { Badge } from '@/components/ui/badge'
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table'
import { formatCompact, formatRelative } from '@/lib/utils'

import { StatusBadge } from './StatusBadge'

const features = tableFeatures({
  rowSortingFeature,
  sortedRowModel: createSortedRowModel(),
  sortFns: { basic: sortFn_basic, datetime: sortFn_datetime, alphanumeric: sortFn_alphanumeric },
})
const helper = createColumnHelper<typeof features, SpeciesSummary>()
const EMPTY: SpeciesSummary[] = []

function severityTone(score: number) {
  if (score >= 75) return 'bg-destructive'
  if (score >= 50) return 'bg-[var(--introduced)]'
  if (score >= 25) return 'bg-warning'
  return 'bg-success'
}

export const registryColumns = helper.columns([
  helper.accessor('scientific_name', {
    header: 'Species',
    sortFn: 'alphanumeric',
    cell: (i) => {
      const sp = i.row.original
      return (
        <Link
          to="/species/$taxonKey"
          params={{ taxonKey: sp.taxon_key }}
          className="flex items-center gap-3"
        >
          {sp.thumbnail_url ? (
            <img src={sp.thumbnail_url} alt="" className="size-9 rounded-md object-cover" loading="lazy" />
          ) : (
            <span className="flex size-9 items-center justify-center rounded-md bg-muted">
              <Leaf className="size-4 text-muted-foreground" />
            </span>
          )}
          <span>
            <span className="block font-medium italic hover:underline">{sp.scientific_name}</span>
            <span className="block text-xs text-muted-foreground">
              {sp.common_name ?? `taxon ${sp.taxon_key}`}
            </span>
          </span>
        </Link>
      )
    },
  }),
  helper.accessor('status', {
    header: 'Status',
    enableSorting: false,
    cell: (i) => (
      <StatusBadge
        status={i.getValue()}
        pendingRecords={i.row.original.n_pending_records}
        trainedTs={i.row.original.model_trained_ts}
      />
    ),
  }),
  helper.accessor('severity_score', {
    header: 'Severity',
    sortFn: 'basic',
    cell: (i) => {
      const v = i.getValue()
      if (v == null) return <span className="text-muted-foreground">–</span>
      return (
        <div className="flex items-center gap-2">
          <div className="h-1.5 w-16 overflow-hidden rounded-full bg-muted">
            <div className={`h-full ${severityTone(v)}`} style={{ width: `${v}%` }} />
          </div>
          <span className="tabular-nums text-xs font-medium">{v.toFixed(0)}</span>
          {i.row.original.model_type === 'A' && (
            <Badge variant="warning" className="px-1.5">A</Badge>
          )}
        </div>
      )
    },
  }),
  helper.accessor('n_occurrences_total', {
    header: 'Records',
    sortFn: 'basic',
    cell: (i) => <span className="tabular-nums">{formatCompact(i.getValue())}</span>,
  }),
  helper.accessor('model_version', {
    header: 'Model',
    sortFn: 'basic',
    cell: (i) => (i.getValue() ? `v${i.getValue()}` : '–'),
  }),
  helper.accessor('last_gbif_fetch_ts', {
    header: 'Last GBIF check',
    sortFn: 'datetime',
    cell: (i) => <span className="text-xs text-muted-foreground">{formatRelative(i.getValue())}</span>,
  }),
])

export function RegistryTable({ species }: { species: SpeciesSummary[] | undefined }) {
  const table = useTable({
    features,
    columns: registryColumns,
    data: species ?? EMPTY,
    initialState: { sorting: [{ id: 'severity_score', desc: true }] },
  })
  return (
    <Table>
      <TableHeader>
        {table.getHeaderGroups().map((g) => (
          <TableRow key={g.id} className="hover:bg-transparent">
            {g.headers.map((h) => {
              const dir = h.column.getIsSorted()
              return (
                <TableHead key={h.id}>
                  {h.column.getCanSort() ? (
                    <button
                      type="button"
                      className="inline-flex items-center gap-1 hover:text-foreground"
                      onClick={h.column.getToggleSortingHandler()}
                    >
                      <table.FlexRender header={h} />
                      {dir === 'asc' && <ArrowUp className="size-3" />}
                      {dir === 'desc' && <ArrowDown className="size-3" />}
                    </button>
                  ) : (
                    <table.FlexRender header={h} />
                  )}
                </TableHead>
              )
            })}
          </TableRow>
        ))}
      </TableHeader>
      <TableBody>
        {table.getRowModel().rows.map((row) => (
          <TableRow key={row.id}>
            {row.getAllCells().map((cell) => (
              <TableCell key={cell.id}>
                <table.FlexRender cell={cell} />
              </TableCell>
            ))}
          </TableRow>
        ))}
      </TableBody>
    </Table>
  )
}
