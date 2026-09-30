import { createFileRoute } from '@tanstack/react-router'
import { useTable } from '@tanstack/react-table'
import { CheckCircle2, FlaskConical, Loader2, RefreshCw, XCircle } from 'lucide-react'

import { useRunValidation, useValidationJob, useValidationReport } from '@/api/queries'
import type { ValidationSpecies } from '@/api/types'
import { JobStatusBar } from '@/components/species/JobStatusBar'
import { validationColumns, validationFeatures } from '@/components/species/validationColumns'
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Skeleton } from '@/components/ui/skeleton'
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table'
import { formatDate } from '@/lib/utils'

export const Route = createFileRoute('/validation')({ component: ValidationPage })

const EMPTY: ValidationSpecies[] = []

function Tile({ label, value, sub }: { label: string; value: string; sub?: string }) {
  return (
    <Card className="px-4 py-3">
      <div className="text-xs text-muted-foreground">{label}</div>
      <div className="mt-1 text-xl font-semibold tabular-nums">{value}</div>
      {sub && <div className="text-[11px] text-muted-foreground">{sub}</div>}
    </Card>
  )
}

function ValidationPage() {
  const report = useValidationReport()
  const job = useValidationJob()
  const run = useRunValidation()
  const r = report.data
  const table = useTable({
    features: validationFeatures,
    columns: validationColumns,
    data: r?.species ?? EMPTY,
  })
  const active = job.data?.status === 'queued' || job.data?.status === 'running'

  return (
    <div className="space-y-5">
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div>
          <h1 className="flex items-center gap-2 text-2xl font-semibold tracking-tight">
            <FlaskConical className="size-6 text-primary" /> Validation suite
          </h1>
          <p className="mt-1 max-w-3xl text-sm text-muted-foreground">
            Reference invaders checked against the invasion literature: known invaded regions
            should be predicted suitable, control regions unsuitable, and a native-only model is
            scored on the introduced records — the independent transferability test behind the
            AUC ≈ 0.7 caveat.
          </p>
        </div>
        <div className="flex gap-2">
          <Button
            variant="outline"
            size="sm"
            disabled={active || run.isPending}
            onClick={() => run.mutate(false)}
          >
            <RefreshCw /> Re-evaluate models
          </Button>
          <Button size="sm" disabled={active || run.isPending} onClick={() => run.mutate(true)}>
            {active ? <Loader2 className="animate-spin" /> : <FlaskConical />} Run full suite
          </Button>
        </div>
      </div>

      {job.data && <JobStatusBar job={job.data} />}

      {report.isLoading ? (
        <Skeleton className="h-64" />
      ) : !r ? (
        <Card>
          <CardContent className="py-10 text-center text-sm text-muted-foreground">
            The validation suite has not been run yet.
          </CardContent>
        </Card>
      ) : (
        <>
          <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
            <Tile
              label="Species evaluated"
              value={String(r.summary.n_species)}
              sub={`report ${formatDate(r.generated_ts, true)}`}
            />
            <Tile label="Passed" value={`${r.summary.n_passed}/${r.summary.n_species}`} />
            <Tile
              label="Mean native-only → invaded AUC"
              value={r.summary.mean_transfer_auc?.toFixed(3) ?? '–'}
              sub={`literature mean ≈ ${r.summary.literature_mean_transfer_auc}`}
            />
            <Tile
              label="Pass criteria"
              value="≥ 75 % captured"
              sub="and every control region unsuitable"
            />
          </div>

          <Alert variant="info">
            <FlaskConical />
            <AlertTitle>How to read this</AlertTitle>
            <AlertDescription>{String(r.criteria.caveat ?? '')}</AlertDescription>
          </Alert>

          <Card>
            <CardContent className="px-2 py-2">
              <Table>
                <TableHeader>
                  {table.getHeaderGroups().map((g) => (
                    <TableRow key={g.id} className="hover:bg-transparent">
                      {g.headers.map((hd) => (
                        <TableHead key={hd.id}>
                          <button
                            type="button"
                            className="hover:text-foreground"
                            onClick={hd.column.getToggleSortingHandler()}
                          >
                            <table.FlexRender header={hd} />
                          </button>
                        </TableHead>
                      ))}
                    </TableRow>
                  ))}
                </TableHeader>
                <TableBody>
                  {table.getRowModel().rows.map((row) => (
                    <TableRow key={row.id}>
                      {row.getAllCells().map((c) => (
                        <TableCell key={c.id}>
                          <table.FlexRender cell={c} />
                        </TableCell>
                      ))}
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </CardContent>
          </Card>

          <div className="grid gap-4 lg:grid-cols-2 xl:grid-cols-3">
            {r.species.map((sp) => (
              <Card key={sp.taxon_key}>
                <CardHeader>
                  <CardTitle className="italic">{sp.scientific_name}</CardTitle>
                  <CardDescription>Native range: {sp.native_range_reference}</CardDescription>
                </CardHeader>
                <CardContent className="space-y-1.5 text-xs">
                  {sp.error && <p className="font-mono text-destructive">{sp.error}</p>}
                  {(sp.regions ?? []).map((reg) => (
                    <div key={reg.name} className="flex items-center gap-2">
                      {reg.passed ? (
                        <CheckCircle2 className="size-3.5 shrink-0 text-success" />
                      ) : (
                        <XCircle className="size-3.5 shrink-0 text-warning" />
                      )}
                      <span className="flex-1 truncate" title={reg.ref}>
                        {reg.name}
                        <span className="text-muted-foreground"> · expect {reg.expect}</span>
                      </span>
                      <span className="tabular-nums text-muted-foreground">
                        {reg.fraction_suitable == null
                          ? 'no land'
                          : `${(reg.fraction_suitable * 100).toFixed(0)}% suitable`}
                      </span>
                    </div>
                  ))}
                </CardContent>
              </Card>
            ))}
          </div>
        </>
      )}
    </div>
  )
}
