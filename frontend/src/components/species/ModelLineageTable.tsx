import { useTable } from '@tanstack/react-table'
import { ArrowDown, ArrowUp, ArrowUpDown } from 'lucide-react'

import type { ModelVersionOut } from '@/api/types'
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table'

import { lineageColumns, lineageFeatures } from './lineageColumns'

const EMPTY: ModelVersionOut[] = []

export function ModelLineageTable({ versions }: { versions: ModelVersionOut[] | undefined }) {
  const table = useTable({
    features: lineageFeatures,
    columns: lineageColumns,
    data: versions ?? EMPTY,
    initialState: { sorting: [{ id: 'model_version', desc: true }] },
  })

  return (
    <Table>
      <TableHeader>
        {table.getHeaderGroups().map((group) => (
          <TableRow key={group.id} className="hover:bg-transparent">
            {group.headers.map((header) => {
              const sortable = header.column.getCanSort()
              const dir = header.column.getIsSorted()
              return (
                <TableHead key={header.id}>
                  {header.isPlaceholder ? null : sortable ? (
                    <button
                      type="button"
                      className="inline-flex items-center gap-1 hover:text-foreground"
                      onClick={header.column.getToggleSortingHandler()}
                    >
                      <table.FlexRender header={header} />
                      {dir === 'asc' ? (
                        <ArrowUp className="size-3" />
                      ) : dir === 'desc' ? (
                        <ArrowDown className="size-3" />
                      ) : (
                        <ArrowUpDown className="size-3 opacity-40" />
                      )}
                    </button>
                  ) : (
                    <table.FlexRender header={header} />
                  )}
                </TableHead>
              )
            })}
          </TableRow>
        ))}
      </TableHeader>
      <TableBody>
        {table.getRowModel().rows.length === 0 ? (
          <TableRow>
            <TableCell colSpan={lineageColumns.length} className="py-10 text-center text-muted-foreground">
              No model versions yet.
            </TableCell>
          </TableRow>
        ) : (
          table.getRowModel().rows.map((row) => (
            <TableRow key={row.id} className="align-top">
              {row.getAllCells().map((cell) => (
                <TableCell key={cell.id} className="align-top">
                  <table.FlexRender cell={cell} />
                </TableCell>
              ))}
            </TableRow>
          ))
        )}
      </TableBody>
    </Table>
  )
}
