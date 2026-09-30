import { useForm } from '@tanstack/react-form'
import { Loader2, RotateCcw, Save } from 'lucide-react'

import { useUpdateSeverityConfig } from '@/api/queries'
import type { SeverityConfigModel } from '@/api/types'
import { SEVERITY_COMPONENTS } from '@/components/charts/SeverityGauge'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Slider } from '@/components/ui/slider'

import { normalizedWeights, severityConfigSchema, type SeverityConfigForm as FormValues } from './severitySchema'

const DEFAULTS: FormValues = {
  weights: { suitability: 0.35, climate_analogy: 0.25, spread_rate: 0.2, ecological_impact_prior: 0.2 },
  impact_prior: 0.5,
  impact_prior_source: 'default (unknown)',
  suitability_saturation: 0.1,
  spread_half_saturation: 5,
  spread_window_years: 10,
}

const NUMERIC_FIELDS = [
  { name: 'suitability_saturation', label: 'Suitability saturation f_sat', step: 0.01 },
  { name: 'spread_half_saturation', label: 'Spread half-saturation k (cells/yr)', step: 0.5 },
  { name: 'spread_window_years', label: 'Spread window (years, applies on retrain)', step: 1 },
] as const

function toFormValues(cfg: SeverityConfigModel): FormValues {
  return { ...DEFAULTS, ...cfg, weights: { ...DEFAULTS.weights, ...cfg.weights } }
}

function firstError(errors: unknown[]): string | null {
  const e = errors[0] as { message?: string } | string | undefined
  if (!e) return null
  return typeof e === 'string' ? e : (e.message ?? null)
}

/**
 * Severity weights & priors (TanStack Form + Zod). Saving re-weights the stored components of
 * the current model on the server — no retraining — and the change is recorded in the bulletin
 * methods footnote.
 */
export function SeverityConfigForm({
  taxonKey,
  config,
}: {
  taxonKey: number
  config: SeverityConfigModel
}) {
  const mutation = useUpdateSeverityConfig(taxonKey)
  const form = useForm({
    defaultValues: toFormValues(config),
    validators: { onChange: severityConfigSchema },
    onSubmit: async ({ value }) => {
      await mutation.mutateAsync(value)
    },
  })

  return (
    <Card>
      <CardHeader>
        <CardTitle>Severity weights &amp; priors</CardTitle>
        <CardDescription>
          Weights are normalised to sum to 1. Changes re-score the current model instantly and are
          documented in the bulletin's methods footnote.
        </CardDescription>
      </CardHeader>
      <CardContent>
        <form
          className="space-y-6"
          onSubmit={(e) => {
            e.preventDefault()
            void form.handleSubmit()
          }}
        >
          <form.Subscribe selector={(s) => s.values.weights}>
            {(weights) => {
              const norm = normalizedWeights(weights)
              return (
                <div className="space-y-4">
                  {SEVERITY_COMPONENTS.map((c) => (
                    <form.Field key={c.key} name={`weights.${c.key}`}>
                      {(field) => (
                        <div className="grid grid-cols-[140px_1fr_90px] items-center gap-3">
                          <Label htmlFor={field.name} className="flex items-center gap-2">
                            <span className="size-2.5 rounded-sm" style={{ background: c.color }} />
                            {c.label}
                          </Label>
                          <Slider
                            id={field.name}
                            min={0}
                            max={1}
                            step={0.05}
                            value={[field.state.value]}
                            onValueChange={([v]) => v !== undefined && field.handleChange(v)}
                          />
                          <span className="text-right text-xs tabular-nums text-muted-foreground">
                            {field.state.value.toFixed(2)} → {(norm[c.key] * 100).toFixed(0)}%
                          </span>
                        </div>
                      )}
                    </form.Field>
                  ))}
                </div>
              )
            }}
          </form.Subscribe>

          <div className="grid gap-4 sm:grid-cols-2">
            <form.Field name="impact_prior">
              {(field) => (
                <div className="space-y-2">
                  <Label htmlFor={field.name}>
                    Ecological impact prior <span className="text-muted-foreground">(0–1)</span>
                  </Label>
                  <div className="flex items-center gap-3">
                    <Slider
                      id={field.name}
                      min={0}
                      max={1}
                      step={0.05}
                      value={[field.state.value]}
                      onValueChange={([v]) => v !== undefined && field.handleChange(v)}
                    />
                    <span className="w-10 text-right text-xs tabular-nums">{field.state.value.toFixed(2)}</span>
                  </div>
                </div>
              )}
            </form.Field>
            <form.Field name="impact_prior_source">
              {(field) => (
                <div className="space-y-2">
                  <Label htmlFor={field.name}>Impact prior source</Label>
                  <Input
                    id={field.name}
                    value={field.state.value}
                    onChange={(e) => field.handleChange(e.target.value)}
                    onBlur={field.handleBlur}
                    aria-invalid={field.state.meta.errors.length > 0}
                    placeholder="e.g. GISD 100 worst, EICAT MR"
                  />
                  {firstError(field.state.meta.errors) && (
                    <p className="text-xs text-destructive">{firstError(field.state.meta.errors)}</p>
                  )}
                </div>
              )}
            </form.Field>
            {NUMERIC_FIELDS.map(({ name, label, step }) => (
              <form.Field key={name} name={name}>
                {(field) => (
                  <div className="space-y-2">
                    <Label htmlFor={field.name}>{label}</Label>
                    <Input
                      id={field.name}
                      type="number"
                      step={step}
                      value={Number.isFinite(field.state.value) ? field.state.value : ''}
                      onChange={(e) => field.handleChange(e.target.valueAsNumber)}
                      onBlur={field.handleBlur}
                      aria-invalid={field.state.meta.errors.length > 0}
                    />
                    {firstError(field.state.meta.errors) && (
                      <p className="text-xs text-destructive">{firstError(field.state.meta.errors)}</p>
                    )}
                  </div>
                )}
              </form.Field>
            ))}
          </div>

          <form.Subscribe selector={(s) => [s.canSubmit, s.isSubmitting, s.errors] as const}>
            {([canSubmit, isSubmitting, errors]) => (
              <div className="flex items-center gap-2">
                <Button type="submit" size="sm" disabled={!canSubmit || isSubmitting}>
                  {isSubmitting ? <Loader2 className="animate-spin" /> : <Save />} Save &amp; re-score
                </Button>
                <Button type="button" variant="ghost" size="sm" onClick={() => form.reset(DEFAULTS)}>
                  <RotateCcw /> Defaults
                </Button>
                {mutation.isError && <span className="text-xs text-destructive">{mutation.error.message}</span>}
                {mutation.isSuccess && !isSubmitting && (
                  <span className="text-xs text-success">Saved — severity re-scored.</span>
                )}
                {errors.length > 0 && (
                  <span className="text-xs text-destructive">{firstError(errors as unknown[])}</span>
                )}
              </div>
            )}
          </form.Subscribe>
        </form>
      </CardContent>
    </Card>
  )
}
