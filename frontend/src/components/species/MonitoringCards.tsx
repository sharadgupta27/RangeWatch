import { useForm } from '@tanstack/react-form'
import { useSelector } from '@tanstack/react-store'
import { BellRing, Crop, Loader2, Rss, Trash2 } from 'lucide-react'

import {
  alertFeedUrl,
  useAlertSubscriptions,
  useCreateSubscription,
  useDeleteSubscription,
  usePermissions,
  useSpeciesAlerts,
  useUpdateStudyRegion,
} from '@/api/queries'
import type { AlertOut, BioclimInfo, SpeciesDetail } from '@/api/types'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Switch } from '@/components/ui/switch'
import { formatDate, resolutionLabel } from '@/lib/utils'
import { mapActions, mapStore } from '@/store/mapStore'

import {
  EMPTY_SUBSCRIPTION,
  studyRegionSchema,
  subscriptionSchema,
  type StudyRegionForm,
} from './monitoringSchema'

function firstError(errors: unknown[]): string | null {
  const e = errors[0] as { message?: string } | string | undefined
  if (!e) return null
  return typeof e === 'string' ? e : (e.message ?? null)
}

const KIND_LABEL: Record<AlertOut['kind'], string> = {
  outside_envelope: 'Outside the suitability envelope',
  outside_known_range: 'Far from every known record',
}

/** B10: a record where observers were already active is a likelier real expansion. */
export function EffortClasses({ classes }: { classes: Record<string, number> }) {
  return (
    <span className="flex flex-wrap gap-1">
      {(classes.expansion ?? 0) > 0 && <Badge variant="destructive">{classes.expansion} likely expansion</Badge>}
      {(classes.discovery ?? 0) > 0 && <Badge variant="secondary">{classes.discovery} likely discovery</Badge>}
      {(classes.unknown ?? 0) > 0 && <Badge variant="secondary">{classes.unknown} effort unknown</Badge>}
    </span>
  )
}

/** U10: early-detection alerts and where they are delivered. */
export function AlertsCard({ taxonKey }: { taxonKey: number }) {
  const alerts = useSpeciesAlerts(taxonKey)
  const subs = useAlertSubscriptions(taxonKey)
  const create = useCreateSubscription(taxonKey)
  const remove = useDeleteSubscription(taxonKey)
  const { run } = usePermissions()
  const form = useForm({
    defaultValues: EMPTY_SUBSCRIPTION,
    validators: { onSubmit: subscriptionSchema },
    onSubmit: async ({ value, formApi }) => {
      await create.mutateAsync({
        channel: value.channel,
        target: value.target.trim(),
        taxon_key: value.allSpecies ? null : taxonKey,
        kinds: ['outside_envelope', 'outside_known_range'],
      })
      formApi.reset()
    },
  })
  return (
    <Card>
      <CardHeader className="flex flex-row flex-wrap items-start justify-between gap-3">
        <div className="space-y-1.5">
          <CardTitle className="flex items-center gap-2">
            <BellRing className="size-4" /> Early-detection alerts
          </CardTitle>
          <CardDescription className="max-w-2xl">
            Raised when new records fall outside the current suitability envelope or far from
            every known record. Each record is checked against the target group&apos;s earlier
            sampling effort: where observers were already active it is more likely a real
            expansion; in a cell nobody had recorded before it may just be a discovery.
          </CardDescription>
        </div>
        <a
          href={alertFeedUrl(taxonKey)}
          className="inline-flex items-center gap-1 text-xs text-primary hover:underline"
        >
          <Rss className="size-3" /> RSS feed
        </a>
      </CardHeader>
      <CardContent className="space-y-5 text-xs">
        {alerts.data?.length ? (
          <ul className="space-y-3">
            {alerts.data.map((a) => (
              <li key={a.alert_id} className="space-y-1 rounded-lg border p-3">
                <div className="flex flex-wrap items-center gap-2">
                  <span className="font-medium">{KIND_LABEL[a.kind]}</span>
                  <span className="text-muted-foreground">{formatDate(a.created_ts, true)}</span>
                  <EffortClasses classes={a.effort_classes} />
                </div>
                <p>{a.summary}</p>
                <ul className="flex flex-wrap gap-x-3 font-mono text-[11px] text-muted-foreground">
                  {a.records.slice(0, 8).map((r) => (
                    <li key={`${r.source}-${r.record_id}`}>
                      <button
                        type="button"
                        className="hover:text-primary"
                        onClick={() =>
                          mapActions.setViewState({ longitude: r.longitude, latitude: r.latitude, zoom: 6, pitch: 0, bearing: 0 })
                        }
                      >
                        {r.source} #{r.record_id}
                      </button>
                    </li>
                  ))}
                  {a.records.length > 8 && <li>+{a.records.length - 8} more</li>}
                </ul>
                {a.deliveries.some((d) => d.status === 'failed') && (
                  <p className="text-warning">Some deliveries failed (see subscriptions).</p>
                )}
              </li>
            ))}
          </ul>
        ) : (
          <p className="text-muted-foreground">No alerts yet. They appear after a check for new records.</p>
        )}

        <section className="space-y-2">
          <h4 className="font-medium">Subscriptions</h4>
          {subs.data?.length ? (
            <ul className="space-y-1">
              {subs.data.map((s) => (
                <li key={s.subscription_id} className="flex items-center gap-2">
                  <Badge variant="secondary">{s.channel}</Badge>
                  <span className="font-mono">{s.target}</span>
                  <span className="text-muted-foreground">{s.taxon_key ? 'this species' : 'all species'}</span>
                  {run && (
                    <Button
                      size="icon"
                      variant="ghost"
                      className="ml-auto size-7"
                      aria-label={`Remove subscription ${s.target}`}
                      onClick={() => remove.mutate(s.subscription_id)}
                    >
                      <Trash2 />
                    </Button>
                  )}
                </li>
              ))}
            </ul>
          ) : (
            <p className="text-muted-foreground">Nobody is notified yet.</p>
          )}
          {run && (
            <form
              className="flex flex-wrap items-end gap-2"
              onSubmit={(e) => {
                e.preventDefault()
                void form.handleSubmit()
              }}
            >
              <form.Field name="channel">
                {(field) => (
                  <Select value={field.state.value} onValueChange={(v) => field.handleChange(v as 'webhook' | 'email')}>
                    <SelectTrigger className="w-28" aria-label="Channel">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="webhook">Webhook</SelectItem>
                      <SelectItem value="email">E-mail</SelectItem>
                    </SelectContent>
                  </Select>
                )}
              </form.Field>
              <form.Field name="target">
                {(field) => (
                  <div className="min-w-64 flex-1 space-y-1">
                    <Label htmlFor="sub-target" className="sr-only">
                      Target
                    </Label>
                    <Input
                      id="sub-target"
                      value={field.state.value}
                      onChange={(e) => field.handleChange(e.target.value)}
                      placeholder="https://hooks.example.org/… or team@example.org"
                      aria-invalid={field.state.meta.errors.length > 0}
                    />
                    {firstError(field.state.meta.errors) && (
                      <p className="text-destructive">{firstError(field.state.meta.errors)}</p>
                    )}
                  </div>
                )}
              </form.Field>
              <form.Field name="allSpecies">
                {(field) => (
                  <label className="flex items-center gap-2">
                    <Switch checked={field.state.value} onCheckedChange={field.handleChange} aria-label="All species" />
                    all species
                  </label>
                )}
              </form.Field>
              <Button size="sm" type="submit" disabled={create.isPending}>
                {create.isPending && <Loader2 className="animate-spin" />} Subscribe
              </Button>
            </form>
          )}
          {create.isError && <p className="text-destructive">{create.error.message}</p>}
        </section>
      </CardContent>
    </Card>
  )
}

/** U2: train and project a regional model, optionally at a registered high resolution. */
export function StudyRegionCard({ sp, bioclim }: { sp: SpeciesDetail; bioclim: BioclimInfo | undefined }) {
  const update = useUpdateStudyRegion(sp.taxon_key)
  const { run } = usePermissions()
  const bounds = useSelector(mapStore, (s) => s.viewBounds)
  const region = sp.study_region
  const [w, s, e, n] = region?.bbox ?? bounds ?? [-180, -90, 180, 90]
  const round = (v: number) => Math.round(v * 100) / 100
  const form = useForm({
    defaultValues: {
      west: round(w!),
      south: round(s!),
      east: round(e!),
      north: round(n!),
      resolution: region?.resolution ?? 'base',
    } as StudyRegionForm,
    validators: { onSubmit: studyRegionSchema },
    onSubmit: async ({ value }) => {
      const v = studyRegionSchema.parse(value)
      await update.mutateAsync({
        bbox: [v.west, v.south, v.east, v.north],
        resolution: v.resolution === 'base' ? null : v.resolution,
      })
    },
  })
  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Crop className="size-4" /> Study region
        </CardTitle>
        <CardDescription>
          {region
            ? `Regional model: [${region.bbox.join(', ')}]${region.resolution ? ` at ${resolutionLabel(region.resolution)}` : ''}.`
            : 'Global model.'}{' '}
          A regional model trains and projects only inside the box (background, CV and maps
          included), optionally on a finer climate grid. Changing it flags a retrain.
        </CardDescription>
      </CardHeader>
      <CardContent className="text-xs">
        {run ? (
          <form
            className="space-y-3"
            onSubmit={(ev) => {
              ev.preventDefault()
              void form.handleSubmit()
            }}
          >
            <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
              {(['west', 'south', 'east', 'north'] as const).map((name) => (
                <form.Field key={name} name={name}>
                  {(field) => (
                    <div className="space-y-1">
                      <Label htmlFor={`region-${name}`} className="capitalize">
                        {name}
                      </Label>
                      <Input
                        id={`region-${name}`}
                        type="number"
                        step="0.01"
                        value={String(field.state.value)}
                        onChange={(ev) => field.handleChange(ev.target.value)}
                        aria-invalid={field.state.meta.errors.length > 0}
                      />
                      {firstError(field.state.meta.errors) && (
                        <p className="text-destructive">{firstError(field.state.meta.errors)}</p>
                      )}
                    </div>
                  )}
                </form.Field>
              ))}
            </div>
            <form.Field name="resolution">
              {(field) => (
                <Select value={field.state.value} onValueChange={field.handleChange}>
                  <SelectTrigger className="w-56" aria-label="Resolution">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="base">Base grid ({bioclim?.resolution ?? 'bioclim'})</SelectItem>
                    {(bioclim?.hires_resolutions ?? []).map((r) => (
                      <SelectItem key={r} value={r}>
                        {resolutionLabel(r)} high resolution
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              )}
            </form.Field>
            <div className="flex flex-wrap gap-2">
              <Button
                type="button"
                variant="outline"
                size="sm"
                disabled={!bounds}
                onClick={() => {
                  if (!bounds) return
                  const [bw, bs, be, bn] = bounds
                  form.setFieldValue('west', round(bw))
                  form.setFieldValue('south', round(bs))
                  form.setFieldValue('east', round(be))
                  form.setFieldValue('north', round(bn))
                }}
              >
                Use map view
              </Button>
              <Button type="submit" size="sm" disabled={update.isPending}>
                {update.isPending && <Loader2 className="animate-spin" />} Save region
              </Button>
              {region && (
                <Button
                  type="button"
                  variant="ghost"
                  size="sm"
                  disabled={update.isPending}
                  onClick={() => update.mutate({ bbox: null, resolution: null })}
                >
                  Back to global
                </Button>
              )}
            </div>
            {update.isError && <p className="text-destructive">{update.error.message}</p>}
          </form>
        ) : (
          <p className="text-muted-foreground">Changing the study region needs the analyst role.</p>
        )}
      </CardContent>
    </Card>
  )
}
