import { useForm } from '@tanstack/react-form'
import { useSelector } from '@tanstack/react-store'
import { Loader2, Thermometer } from 'lucide-react'

import { useClimateVersions, usePermissions, useUpdateEnvironment } from '@/api/queries'
import type { ClimateVersionOut, SpeciesDetail } from '@/api/types'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Label } from '@/components/ui/label'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Switch } from '@/components/ui/switch'
import { ToggleGroup, ToggleGroupItem } from '@/components/ui/toggle-group'
import { resolutionLabel } from '@/lib/utils'

import {
  environmentSchema,
  groupPredictors,
  presentOnly,
  type EnvironmentForm,
} from './environmentSchema'

function versionFacts(v: ClimateVersionOut): string {
  const hires = v.hires_resolutions ?? []
  const scenarios = v.scenarios ?? []
  const parts = [
    v.baseline_period ? `baseline ${v.baseline_period}` : null,
    `${resolutionLabel(v.resolution)} grid`,
    hires.length
      ? `${hires.map(resolutionLabel).join(', ')} for regional models`
      : 'no high-resolution grid',
    `${scenarios.length} future scenario${scenarios.length === 1 ? '' : 's'}`,
  ]
  return parts.filter(Boolean).join(' · ')
}

/**
 * Climate data (WorldClim / CHELSA) and predictor choice for one species. Saving flags a
 * retrain; the retrain decision itself stays in the backend.
 */
export function ClimatePredictorsCard({ sp }: Readonly<{ sp: SpeciesDetail }>) {
  const versions = useClimateVersions()
  const update = useUpdateEnvironment(sp.taxon_key)
  const { run } = usePermissions()
  const cfg = sp.predictor_config
  const form = useForm({
    defaultValues: {
      climateVersion: sp.climate_version ?? '',
      mode: cfg.mode ?? 'auto',
      predictors: cfg.predictors ?? [],
      vifScreen: cfg.vif_screen ?? true,
    } as EnvironmentForm,
    validators: { onSubmit: environmentSchema },
    onSubmit: async ({ value }) => {
      const v = environmentSchema.parse(value)
      await update.mutateAsync({
        climate_version: v.climateVersion || null,
        predictor_config:
          v.mode === 'custom'
            ? { mode: 'custom', predictors: v.predictors, vif_screen: v.vifScreen, protected: [] }
            : { mode: 'auto', predictors: [], vif_screen: true, protected: [] },
      })
    },
  })
  const values = useSelector(form.store, (s) => s.values)
  const all = versions.data ?? []
  const fallback = all.find((v) => v.is_default)
  const selected = all.find((v) => v.version === values.climateVersion) ?? fallback
  const offered = selected?.predictors ?? []
  const warnPresentOnly =
    values.mode === 'custom' ? presentOnly(values.predictors, offered) : []

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Thermometer className="size-4" /> Climate data &amp; predictors
        </CardTitle>
        <CardDescription>
          Trained on <span className="font-mono">{sp.effective_climate_version}</span>
          {sp.climate_version ? ' (chosen for this species)' : ' (deployment default)'}. Which
          climate variables limit a species differs between species. Choose them here, or let the
          predictor-set comparison (Model tab) rank candidate sets on this species&apos; data.
          Changing either flags a retrain.
        </CardDescription>
      </CardHeader>
      <CardContent className="text-xs">
        {versions.isError && <p className="text-destructive">Climate data list unavailable.</p>}
        {run ? (
          <form
            className="space-y-4"
            onSubmit={(ev) => {
              ev.preventDefault()
              void form.handleSubmit()
            }}
          >
            <form.Field name="climateVersion">
              {(field) => (
                <div className="space-y-1">
                  <Label htmlFor="climate-version">Climate data</Label>
                  <Select
                    value={field.state.value || '__default'}
                    onValueChange={(v) => {
                      const next = v === '__default' ? '' : v
                      field.handleChange(next)
                      // Keep only the chosen predictors the new version offers.
                      const target = all.find((x) => x.version === next) ?? fallback
                      const names = new Set((target?.predictors ?? []).map((p) => p.name))
                      form.setFieldValue(
                        'predictors',
                        form.getFieldValue('predictors').filter((p) => names.has(p)),
                      )
                    }}
                  >
                    <SelectTrigger id="climate-version" className="w-full sm:w-96">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="__default">
                        Deployment default{fallback ? ` — ${fallback.label}` : ''}
                      </SelectItem>
                      {all.map((v) => (
                        <SelectItem key={v.version} value={v.version}>
                          {v.label} · {v.version}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                  {selected && <p className="text-muted-foreground">{versionFacts(selected)}</p>}
                </div>
              )}
            </form.Field>

            <form.Field name="mode">
              {(field) => (
                <div className="space-y-1">
                  <Label>Predictors</Label>
                  <ToggleGroup
                    type="single"
                    className="w-fit"
                    value={field.state.value}
                    onValueChange={(v) => v && field.handleChange(v as 'auto' | 'custom')}
                  >
                    <ToggleGroupItem value="auto">Automatic</ToggleGroupItem>
                    <ToggleGroupItem value="custom">Choose variables</ToggleGroupItem>
                  </ToggleGroup>
                  {field.state.value === 'auto' && selected && (
                    <p className="text-muted-foreground">
                      The version&apos;s global VIF &lt; 5 shortlist:{' '}
                      <span className="font-mono">{selected.selected_predictors.join(', ')}</span>
                    </p>
                  )}
                </div>
              )}
            </form.Field>

            {values.mode === 'custom' && (
              <form.Field name="predictors">
                {(field) => {
                  const chosen = new Set(field.state.value)
                  const toggle = (names: string[], on: boolean) => {
                    const next = new Set(chosen)
                    for (const n of names) {
                      if (on) next.add(n)
                      else next.delete(n)
                    }
                    field.handleChange(offered.map((p) => p.name).filter((n) => next.has(n)))
                  }
                  return (
                    <div className="space-y-3">
                      {groupPredictors(offered).map(([group, items]) => {
                        const names = items.map((p) => p.name)
                        const allOn = names.every((n) => chosen.has(n))
                        return (
                          <fieldset key={group} className="rounded-md border p-2">
                            <legend className="flex items-center gap-2 px-1 font-medium">
                              {group}
                              <Button
                                type="button"
                                variant="ghost"
                                size="sm"
                                className="h-6 px-1.5 text-[11px]"
                                onClick={() => toggle(names, !allOn)}
                              >
                                {allOn ? 'Clear' : 'All'}
                              </Button>
                            </legend>
                            <div className="grid gap-x-4 gap-y-1 sm:grid-cols-2">
                              {items.map((p) => (
                                <label key={p.name} className="flex items-start gap-2">
                                  <input
                                    type="checkbox"
                                    className="mt-0.5 accent-primary"
                                    checked={chosen.has(p.name)}
                                    onChange={(ev) => toggle([p.name], ev.target.checked)}
                                  />
                                  <span>
                                    <span className="font-mono">{p.name}</span>{' '}
                                    <span className="text-muted-foreground">{p.description}</span>
                                    {p.future_available === false && (
                                      <Badge variant="warning" className="ml-1">
                                        present only
                                      </Badge>
                                    )}
                                  </span>
                                </label>
                              ))}
                            </div>
                          </fieldset>
                        )
                      })}
                      {field.state.meta.errors.length > 0 && (
                        <p className="text-destructive">Choose at least two predictors</p>
                      )}
                    </div>
                  )
                }}
              </form.Field>
            )}

            {values.mode === 'custom' && (
              <form.Field name="vifScreen">
                {(field) => (
                  <div className="flex items-center gap-2">
                    <Switch
                      id="vif-screen"
                      checked={field.state.value}
                      onCheckedChange={field.handleChange}
                    />
                    <Label htmlFor="vif-screen" className="font-normal">
                      Drop collinear variables (VIF ≥ 5, on this species&apos; background)
                    </Label>
                  </div>
                )}
              </form.Field>
            )}

            {warnPresentOnly.length > 0 && (
              <p className="rounded-md border border-warning/40 bg-warning/10 p-2">
                <span className="font-mono">{warnPresentOnly.join(', ')}</span>{' '}
                {warnPresentOnly.length === 1 ? 'has' : 'have'} no future-climate data in this
                source: the model will map the present distribution (and can be validated), but
                cannot be projected onto future scenarios.
              </p>
            )}

            <Button type="submit" size="sm" disabled={update.isPending || !versions.data}>
              {update.isPending && <Loader2 className="animate-spin" />} Save
            </Button>
            {update.isError && <p className="text-destructive">{String(update.error)}</p>}
          </form>
        ) : (
          <p className="text-muted-foreground">
            Predictors: {cfg.mode === 'custom' ? (cfg.predictors ?? []).join(', ') : 'automatic'}.
          </p>
        )}
      </CardContent>
    </Card>
  )
}
