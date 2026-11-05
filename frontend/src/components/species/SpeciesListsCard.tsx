import { useForm } from '@tanstack/react-form'
import { ListChecks, Loader2, Play, Trash2 } from 'lucide-react'

import {
  useBatchJob,
  useCreateSpeciesList,
  useDeleteSpeciesList,
  usePermissions,
  useRunSpeciesList,
  useSpeciesLists,
} from '@/api/queries'
import type { SpeciesListOut } from '@/api/types'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Textarea } from '@/components/ui/textarea'

import { JobStatusBar } from './JobStatusBar'
import { EMPTY_SPECIES_LIST, speciesListSchema, toSpeciesListIn } from './listSchema'

function firstError(errors: unknown[]): string | null {
  const e = errors[0] as { message?: string } | string | undefined
  if (!e) return null
  return typeof e === 'string' ? e : (e.message ?? null)
}

/** Registry filter: the whole registry or one species list. */
export function ListFilter({
  value,
  onChange,
}: {
  value: string | null
  onChange: (listId: string | null) => void
}) {
  const lists = useSpeciesLists()
  if (!lists.data?.length) return null
  return (
    <Select value={value ?? 'all'} onValueChange={(v) => onChange(v === 'all' ? null : v)}>
      <SelectTrigger className="w-56" aria-label="Species list">
        <SelectValue placeholder="All species" />
      </SelectTrigger>
      <SelectContent>
        <SelectItem value="all">All tracked species</SelectItem>
        {lists.data.map((l) => (
          <SelectItem key={l.list_id} value={l.list_id}>
            {l.name} ({l.n_species})
          </SelectItem>
        ))}
      </SelectContent>
    </Select>
  )
}

/** One-line summary of what a list holds and how far its species are. */
export function listSummary(l: SpeciesListOut): string {
  const parts = [`${l.n_species} species`, `${l.n_registered} tracked`, `${l.n_trained} with a model`]
  if (l.n_unresolved) parts.push(`${l.n_unresolved} not matched in GBIF`)
  return parts.join(' · ')
}

/** U9: watch lists (e.g. the EU list of invasive alien species) and batch runs. */
export function SpeciesListsCard() {
  const lists = useSpeciesLists()
  const create = useCreateSpeciesList()
  const remove = useDeleteSpeciesList()
  const run = useRunSpeciesList()
  const job = useBatchJob()
  const { run: canRun } = usePermissions()
  const form = useForm({
    defaultValues: EMPTY_SPECIES_LIST,
    validators: { onSubmit: speciesListSchema },
    onSubmit: async ({ value, formApi }) => {
      await create.mutateAsync(toSpeciesListIn(value))
      formApi.reset()
    },
  })
  const active = job.data?.status === 'queued' || job.data?.status === 'running'

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <ListChecks className="size-4" /> Species lists
        </CardTitle>
        <CardDescription className="max-w-3xl">
          Watch lists for batch runs and registry filters. Names are matched against the GBIF
          backbone; check fuzzy matches. A batch run checks every species of the list for new
          records; new species stop at the native-range review, as in a single run.
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4 text-xs">
        {job.data && <JobStatusBar job={job.data} />}
        {lists.data?.length ? (
          <ul className="space-y-3">
            {lists.data.map((l) => (
              <li key={l.list_id} className="space-y-1.5 rounded-lg border p-3">
                <div className="flex flex-wrap items-center gap-2">
                  <span className="font-medium">{l.name}</span>
                  <span className="text-muted-foreground">{listSummary(l)}</span>
                  <span className="ml-auto flex gap-1">
                    <Button
                      size="sm"
                      variant="outline"
                      disabled={!canRun || active || run.isPending || l.n_species === 0}
                      onClick={() => run.mutate({ listId: l.list_id, body: { force_retrain: false, only_registered: false } })}
                    >
                      {run.isPending ? <Loader2 className="animate-spin" /> : <Play />} Run all
                    </Button>
                    <Button
                      size="sm"
                      variant="ghost"
                      aria-label={`Delete list ${l.name}`}
                      disabled={!canRun || remove.isPending}
                      onClick={() => remove.mutate(l.list_id)}
                    >
                      <Trash2 />
                    </Button>
                  </span>
                </div>
                {l.source && <div className="break-all text-muted-foreground">{l.source}</div>}
                <div className="flex flex-wrap gap-1">
                  {l.entries.map((e) => (
                    <Badge
                      key={e.query}
                      variant={e.taxon_key == null ? 'destructive' : e.match_type === 'FUZZY' || e.match_type === 'HIGHERRANK' ? 'warning' : 'secondary'}
                      title={e.taxon_key == null ? 'No GBIF backbone match' : `${e.match_type} · GBIF ${e.taxon_key}`}
                    >
                      {e.scientific_name ?? e.query}
                      {e.match_type && e.match_type !== 'EXACT' && e.match_type !== 'KEY' && ` (${e.match_type.toLowerCase()})`}
                    </Badge>
                  ))}
                </div>
              </li>
            ))}
          </ul>
        ) : (
          <p className="text-muted-foreground">No species lists yet.</p>
        )}
        {run.isError && <p className="text-destructive">{run.error.message}</p>}

        {canRun && (
          <form
            className="space-y-2 border-t pt-3"
            onSubmit={(e) => {
              e.preventDefault()
              void form.handleSubmit()
            }}
          >
            <h4 className="font-medium">New list</h4>
            <div className="grid gap-2 sm:grid-cols-2">
              <form.Field name="name">
                {(field) => (
                  <div className="space-y-1">
                    <Label htmlFor="list-name">Name</Label>
                    <Input
                      id="list-name"
                      value={field.state.value}
                      onChange={(e) => field.handleChange(e.target.value)}
                      aria-invalid={!!firstError(field.state.meta.errors)}
                    />
                    {firstError(field.state.meta.errors) && (
                      <p className="text-destructive">{firstError(field.state.meta.errors)}</p>
                    )}
                  </div>
                )}
              </form.Field>
              <form.Field name="source">
                {(field) => (
                  <div className="space-y-1">
                    <Label htmlFor="list-source">Source (citation or URL, optional)</Label>
                    <Input id="list-source" value={field.state.value} onChange={(e) => field.handleChange(e.target.value)} />
                  </div>
                )}
              </form.Field>
            </div>
            <form.Field name="entries">
              {(field) => (
                <div className="space-y-1">
                  <Label htmlFor="list-entries">Species: one scientific name or GBIF taxon key per line</Label>
                  <Textarea
                    id="list-entries"
                    rows={5}
                    value={field.state.value}
                    onChange={(e) => field.handleChange(e.target.value)}
                    aria-invalid={!!firstError(field.state.meta.errors)}
                    placeholder={'Lantana camara\nCarpobrotus edulis\n2891932'}
                  />
                  {firstError(field.state.meta.errors) && (
                    <p className="text-destructive">{firstError(field.state.meta.errors)}</p>
                  )}
                </div>
              )}
            </form.Field>
            <Button type="submit" size="sm" disabled={create.isPending}>
              {create.isPending && <Loader2 className="animate-spin" />} Create list
            </Button>
            {create.isError && <p className="text-destructive">{create.error.message}</p>}
          </form>
        )}
      </CardContent>
    </Card>
  )
}
