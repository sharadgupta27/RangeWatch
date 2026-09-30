import { createFileRoute } from '@tanstack/react-router'
import { Activity, AlertTriangle, Database, GitBranch } from 'lucide-react'

import { speciesListOptions, useSpeciesList } from '@/api/queries'
import { RegistryTable } from '@/components/species/RegistryTable'
import { SpeciesSearch } from '@/components/species/SpeciesSearch'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Skeleton } from '@/components/ui/skeleton'
import { formatCompact } from '@/lib/utils'

export const Route = createFileRoute('/')({
  loader: ({ context }) => context.queryClient.ensureQueryData(speciesListOptions()).catch(() => []),
  component: RegistryPage,
})

function Kpi({ icon: Icon, label, value }: { icon: typeof Activity; label: string; value: string }) {
  return (
    <Card className="flex items-center gap-3 px-4 py-3">
      <span className="flex size-9 items-center justify-center rounded-lg bg-primary/10 text-primary">
        <Icon className="size-4" />
      </span>
      <span>
        <span className="block text-xs text-muted-foreground">{label}</span>
        <span className="block text-lg font-semibold tabular-nums">{value}</span>
      </span>
    </Card>
  )
}

function RegistryPage() {
  const list = useSpeciesList()
  const species = list.data ?? []
  const records = species.reduce((a, s) => a + s.n_occurrences_total, 0)
  const pending = species.filter((s) =>
    ['minor_update_pending', 'retrain_pending', 'awaiting_native_range_review'].includes(s.status),
  ).length
  const models = species.reduce((a, s) => a + s.model_version, 0)

  return (
    <div className="space-y-6">
      <section className="relative overflow-hidden rounded-2xl border bg-gradient-to-br from-primary/10 via-card to-card px-6 py-10 sm:px-10">
        <div className="max-w-2xl">
          <h1 className="text-2xl font-semibold tracking-tight sm:text-3xl">
            Where could a species establish next?
          </h1>
          <p className="mt-2 text-sm leading-relaxed text-muted-foreground">
            Pick a species: occurrences are pulled from GBIF and iNaturalist, a MaxEnt model is
            trained on WorldClim bioclimate with spatial block cross-validation, and suitability,
            extrapolation (MESS) and a transparent severity index are projected globally. Tracked
            species are re-checked for new records and retrained only when warranted.
          </p>
          <SpeciesSearch className="mt-6 max-w-xl" />
        </div>
      </section>

      <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
        <Kpi icon={Database} label="Tracked species" value={String(species.length)} />
        <Kpi icon={Activity} label="Stored occurrences" value={formatCompact(records)} />
        <Kpi icon={GitBranch} label="Model versions" value={String(models)} />
        <Kpi icon={AlertTriangle} label="Needs attention" value={String(pending)} />
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Species registry</CardTitle>
          <CardDescription>
            Sorted by severity. Model A (native-only) scores are flagged as lower confidence.
          </CardDescription>
        </CardHeader>
        <CardContent className="px-2 pb-2">
          {list.isLoading ? (
            <div className="space-y-2 p-3">
              {Array.from({ length: 4 }, (_, i) => (
                <Skeleton key={i} className="h-12" />
              ))}
            </div>
          ) : list.isError ? (
            <p className="p-6 text-sm text-destructive">Could not reach the API: {list.error.message}</p>
          ) : species.length === 0 ? (
            <p className="p-6 text-sm text-muted-foreground">
              No species tracked yet — search for one above to start.
            </p>
          ) : (
            <RegistryTable species={species} />
          )}
        </CardContent>
      </Card>
    </div>
  )
}
