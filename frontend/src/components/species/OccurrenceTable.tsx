import {
  createColumnHelper,
  rowPaginationFeature,
  tableFeatures,
  useTable,
  type PaginationState,
} from '@tanstack/react-table'
import { ChevronLeft, ChevronRight } from 'lucide-react'
import { useState } from 'react'

import { useOccurrences } from '@/api/queries'
import type { OccurrenceRow } from '@/api/types'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table'
import { formatDate, formatNumber } from '@/lib/utils'

const features = tableFeatures({ rowPaginationFeature })
const helper = createColumnHelper<typeof features, OccurrenceRow>()
const EMPTY: OccurrenceRow[] = []

const LABEL_VARIANT = { native: 'success', introduced: 'destructive', unknown: 'secondary' } as const

export const occurrenceColumns = helper.columns([
  helper.accessor('source', {
    header: 'Source',
    cell: (i) => (
      <span className="text-xs">
        {i.getValue()} <span className="font-mono text-muted-foreground">#{i.row.original.record_id}</span>
      </span>
    ),
  }),
  helper.accessor('range_label', {
    header: 'Range label',
    cell: (i) => <Badge variant={LABEL_VARIANT[i.getValue()]}>{i.getValue()}</Badge>,
  }),
  helper.accessor('event_date', { header: 'Observed', cell: (i) => formatDate(i.getValue()) }),
  helper.accessor((r) => `${r.latitude.toFixed(3)}, ${r.longitude.toFixed(3)}`, {
    id: 'coords',
    header: 'Lat, lon',
    cell: (i) => <span className="font-mono text-xs">{i.getValue()}</span>,
  }),
  helper.accessor('coordinate_uncertainty_m', {
    header: 'Uncertainty',
    cell: (i) => (i.getValue() == null ? '–' : `${formatNumber(i.getValue())} m`),
  }),
  helper.accessor('basis_of_record', {
    header: 'Basis of record',
    cell: (i) => <span className="text-xs text-muted-foreground">{i.getValue() ?? '–'}</span>,
  }),
  helper.accessor('country_code', { header: 'Country', cell: (i) => i.getValue() ?? '–' }),
  helper.accessor('ingested_ts', {
    header: 'Ingested',
    cell: (i) => <span className="text-xs">{formatDate(i.getValue(), true)}</span>,
  }),
])

type SourceFilter = 'all' | 'GBIF' | 'iNaturalist'
type LabelFilter = 'all' | 'native' | 'introduced' | 'unknown'

/** Occurrence list (table view only — the map uses pg_tileserv vector tiles). */
export function OccurrenceTable({ taxonKey }: { taxonKey: number }) {
  const [pagination, setPagination] = useState<PaginationState>({ pageIndex: 0, pageSize: 25 })
  const [source, setSource] = useState<SourceFilter>('all')
  const [label, setLabel] = useState<LabelFilter>('all')
  const query = useOccurrences(taxonKey, {
    ...pagination,
    source: source === 'all' ? undefined : source,
    rangeLabel: label === 'all' ? undefined : label,
  })

  const table = useTable({
    features,
    columns: occurrenceColumns,
    data: query.data?.items ?? EMPTY,
    rowCount: query.data?.total,
    manualPagination: true,
    state: { pagination },
    onPaginationChange: (u) => setPagination((p) => (typeof u === 'function' ? u(p) : u)),
  })

  const total = query.data?.total ?? 0
  const from = total === 0 ? 0 : pagination.pageIndex * pagination.pageSize + 1
  const to = Math.min(total, (pagination.pageIndex + 1) * pagination.pageSize)
  const resetPage = () => setPagination((p) => ({ ...p, pageIndex: 0 }))

  return (
    <div className="space-y-3">
      <div className="flex flex-wrap items-center gap-2">
        <Select value={source} onValueChange={(v) => { setSource(v as SourceFilter); resetPage() }}>
          <SelectTrigger className="w-40" aria-label="Filter by source">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="all">All sources</SelectItem>
            <SelectItem value="GBIF">GBIF</SelectItem>
            <SelectItem value="iNaturalist">iNaturalist</SelectItem>
          </SelectContent>
        </Select>
        <Select value={label} onValueChange={(v) => { setLabel(v as LabelFilter); resetPage() }}>
          <SelectTrigger className="w-40" aria-label="Filter by range label">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="all">All labels</SelectItem>
            <SelectItem value="native">Native</SelectItem>
            <SelectItem value="introduced">Introduced</SelectItem>
            <SelectItem value="unknown">Unlabelled</SelectItem>
          </SelectContent>
        </Select>
        <span className="ml-auto text-xs text-muted-foreground">
          Labels shown are as ingested from GBIF/iNaturalist; training uses them together with the
          confirmed native-range polygon.
        </span>
      </div>
      <div className="rounded-xl border">
        <Table>
          <TableHeader>
            {table.getHeaderGroups().map((g) => (
              <TableRow key={g.id} className="hover:bg-transparent">
                {g.headers.map((h) => (
                  <TableHead key={h.id}>
                    <table.FlexRender header={h} />
                  </TableHead>
                ))}
              </TableRow>
            ))}
          </TableHeader>
          <TableBody className={query.isPlaceholderData ? 'opacity-60' : undefined}>
            {table.getRowModel().rows.map((row) => (
              <TableRow key={row.id}>
                {row.getAllCells().map((cell) => (
                  <TableCell key={cell.id}>
                    <table.FlexRender cell={cell} />
                  </TableCell>
                ))}
              </TableRow>
            ))}
            {!query.isLoading && total === 0 && (
              <TableRow>
                <TableCell colSpan={occurrenceColumns.length} className="py-10 text-center text-muted-foreground">
                  No occurrences match these filters.
                </TableCell>
              </TableRow>
            )}
          </TableBody>
        </Table>
      </div>
      <div className="flex items-center justify-end gap-2 text-xs text-muted-foreground">
        <span className="tabular-nums">
          {formatNumber(from)}–{formatNumber(to)} of {formatNumber(total)}
        </span>
        <Button variant="outline" size="icon" className="size-8" onClick={() => table.previousPage()} disabled={!table.getCanPreviousPage()} aria-label="Previous page">
          <ChevronLeft />
        </Button>
        <Button variant="outline" size="icon" className="size-8" onClick={() => table.nextPage()} disabled={!table.getCanNextPage()} aria-label="Next page">
          <ChevronRight />
        </Button>
      </div>
    </div>
  )
}
