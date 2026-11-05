import { Crosshair, Download, GitCompare, Layers3, Sigma, Target } from 'lucide-react'

import { absoluteApiUrl } from '@/api/client'
import { useExports } from '@/api/queries'
import type { ModelVersionDetail } from '@/api/types'
import { ResponseCurvesChart } from '@/components/charts/ResponseCurvesChart'
import { Badge } from '@/components/ui/badge'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table'
import { formatNumber } from '@/lib/utils'
import { mapActions } from '@/store/mapStore'

const pct = (v: number | null | undefined) => (v == null ? '–' : `${(v * 100).toFixed(0)}%`)
const num = (v: number | null | undefined, d = 3) => (v == null ? '–' : v.toFixed(d))

/** Every threshold rule with its training sensitivity / background specificity (U5). */
export function ThresholdsCard({ model }: { model: ModelVersionDetail }) {
  const rules = Object.entries(model.thresholds ?? {})
  if (!rules.length) return null
  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Target className="size-4" /> Thresholds and binary maps
        </CardTitle>
        <CardDescription>
          The zones map uses <strong>{model.threshold_rule}</strong> ({num(model.threshold)}).
          Background points stand in for absences, so specificity is relative. Show any rule as
          a yes/no map; the extrapolation overlay still applies.
        </CardDescription>
      </CardHeader>
      <CardContent className="px-2">
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>Rule</TableHead>
              <TableHead className="text-right">Value</TableHead>
              <TableHead className="text-right">Sensitivity</TableHead>
              <TableHead className="text-right">Specificity (bg)</TableHead>
              <TableHead />
            </TableRow>
          </TableHeader>
          <TableBody>
            {rules.map(([rule, t]) => (
              <TableRow key={rule}>
                <TableCell className="text-xs">
                  {t.label}
                  {t.value === model.threshold && <Badge className="ml-2">used</Badge>}
                </TableCell>
                <TableCell className="text-right font-mono text-xs">{num(t.value)}</TableCell>
                <TableCell className="text-right text-xs">{pct(t.sensitivity)}</TableCell>
                <TableCell className="text-right text-xs">{pct(t.specificity_background)}</TableCell>
                <TableCell className="text-right">
                  <button
                    type="button"
                    className="text-xs text-primary hover:underline"
                    onClick={() => mapActions.setProjection(`product:binary:${rule}`)}
                  >
                    Show map
                  </button>
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </CardContent>
    </Card>
  )
}

/** U3 / §4.4: what the ensemble or integrated model is made of. */
export function AlgorithmCard({ model }: { model: ModelVersionDetail }) {
  if (model.algorithm === 'ensemble' && model.ensemble) {
    const e = model.ensemble
    return (
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Layers3 className="size-4" /> Ensemble members
          </CardTitle>
          <CardDescription>
            {e.weighting}. Every member is tested on the same spatial blocks against held-out{' '}
            {e.cv_test_set}. The ensemble-agreement map shows where they disagree.
          </CardDescription>
        </CardHeader>
        <CardContent className="px-2">
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Algorithm</TableHead>
                <TableHead className="text-right">Weight</TableHead>
                <TableHead className="text-right">CV AUC</TableHead>
                <TableHead className="text-right">CV CBI</TableHead>
                <TableHead className="text-right">CV TSS</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {e.members.map((m) => (
                <TableRow key={m.algorithm} className={m.included ? undefined : 'opacity-60'}>
                  <TableCell className="text-xs">
                    {m.label}
                    {m.note && <div className="text-[11px] text-warning">{m.note}</div>}
                  </TableCell>
                  <TableCell className="text-right text-xs">{m.included ? pct(m.weight) : 'left out'}</TableCell>
                  <TableCell className="text-right font-mono text-xs">{num(m.auc_mean, 2)}</TableCell>
                  <TableCell className="text-right font-mono text-xs">{num(m.cbi_mean, 2)}</TableCell>
                  <TableCell className="text-right font-mono text-xs">{num(m.tss_mean, 2)}</TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
          <p className="px-2 pt-2 text-[11px] text-muted-foreground">{e.reference}</p>
        </CardContent>
      </Card>
    )
  }
  if (model.algorithm === 'integrated' && model.integrated) {
    const m = model.integrated
    return (
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Sigma className="size-4" /> Integrated model <Badge variant="warning">experimental</Badge>
          </CardTitle>
          <CardDescription>{m.method}. {m.prediction}.</CardDescription>
        </CardHeader>
        <CardContent className="space-y-2 text-xs">
          <div>
            {formatNumber(m.n_presence_only)} presence-only records · {formatNumber(m.n_pa_present)} survey
            presences · {formatNumber(m.n_pa_absent)} survey absences ·{' '}
            {m.converged ? 'converged' : 'did not converge'}
          </div>
          {Object.keys(m.bias_coefficients).length > 0 && (
            <div>
              Estimated observer bias:{' '}
              {Object.entries(m.bias_coefficients).map(([k, v]) => (
                <span key={k} className="font-mono">
                  {k} {v.toFixed(2)}
                </span>
              ))}{' '}
              <span className="text-muted-foreground">(dropped for prediction)</span>
            </div>
          )}
          <div>
            Survey AUC on held-out blocks: <span className="font-mono">{num(m.cv_survey_auc_mean, 2)}</span>
          </div>
          <p className="text-[11px] text-muted-foreground">{m.reference}</p>
        </CardContent>
      </Card>
    )
  }
  if (model.algorithm === 'ppm' && model.point_process) {
    const m = model.point_process
    const t = m.offset_test
    return (
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Sigma className="size-4" /> Point process with effort offset{' '}
            <Badge variant="warning">experimental</Badge>
          </CardTitle>
          <CardDescription>
            {m.method}. Bias mode <strong>{m.bias_mode}</strong>: offset {m.offset}. {m.prediction}.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-2 text-xs">
          <div>
            {formatNumber(m.n_presences)} presences · {formatNumber(m.n_quadrature)} quadrature
            points · {m.converged ? 'converged' : 'did not converge'}
          </div>
          {Object.keys(m.free_bias_coefficients).length > 0 && (
            <div>
              Estimated bias coefficients:{' '}
              {Object.entries(m.free_bias_coefficients).map(([k, v]) => (
                <span key={k} className="mr-2 font-mono">
                  {k} {v.toFixed(3)}
                </span>
              ))}
            </div>
          )}
          {t && (
            <div className={t.p_value < 0.05 ? 'text-warning' : undefined}>
              Offset test ({t.column}): assumed {t.assumed_coefficient}, estimated{' '}
              <span className="font-mono">{t.free_coefficient.toFixed(3)}</span>, LR{' '}
              {t.lr_statistic.toFixed(2)}, p = {t.p_value.toFixed(3)} — {t.interpretation}.
            </div>
          )}
          <p className="text-[11px] text-muted-foreground">{m.reference}</p>
        </CardContent>
      </Card>
    )
  }
  return null
}

/** U6 + U12 + §3.2: what changed, where to survey next, inferred-absence check. */
export function MonitoringProductsCard({ model }: { model: ModelVersionDetail }) {
  const ch = model.change_previous
  const pr = model.survey_priority
  const ia = model.inferred_absences
  const lim = model.projection.limiting_factor
  if (!ch && !pr && !ia && !lim) return null
  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <GitCompare className="size-4" /> Change, limiting factors and survey priority
        </CardTitle>
        <CardDescription>Open any of these as a map from “More maps” on the projection map.</CardDescription>
      </CardHeader>
      <CardContent className="space-y-4 text-xs">
        {ch && (
          <section>
            <h4 className="font-medium">
              Since v{ch.from_version}
              {ch.status === 'available' && (
                <button
                  type="button"
                  className="ml-2 font-normal text-primary hover:underline"
                  onClick={() => mapActions.setProjection('product:change_previous')}
                >
                  map
                </button>
              )}
            </h4>
            {ch.status === 'available' ? (
              <p>
                <span className="text-success">+{formatNumber(ch.gain_km2 ?? 0)} km² gained</span> ·{' '}
                <span className="text-destructive">−{formatNumber(ch.loss_km2 ?? 0)} km² lost</span> · net{' '}
                {pct(ch.net_change_fraction)} of the previous suitable area
              </p>
            ) : (
              <p className="text-muted-foreground">Not available: {ch.reason}</p>
            )}
          </section>
        )}
        {lim && lim.land.length > 0 && (
          <section>
            <h4 className="font-medium">Limiting factor (share of land)</h4>
            <p className="text-muted-foreground">{lim.rule}</p>
            <ul className="mt-1 flex flex-wrap gap-1">
              {lim.land.slice(0, 6).map((e) => (
                <Badge key={e.variable} variant="secondary">
                  {e.variable} {pct(e.fraction)}
                </Badge>
              ))}
            </ul>
          </section>
        )}
        {pr && (
          <section>
            <h4 className="flex items-center gap-1 font-medium">
              <Crosshair className="size-3" /> Where to look next
            </h4>
            {pr.status === 'available' ? (
              <>
                <p className="text-muted-foreground">{pr.formula}</p>
                <ol className="mt-1 space-y-0.5 font-mono text-[11px]">
                  {(pr.top_cells ?? []).slice(0, 5).map((c) => (
                    <li key={c.rank}>
                      <button
                        type="button"
                        className="hover:text-primary"
                        onClick={() => {
                          mapActions.setProjection('product:priority')
                          mapActions.setViewState({ longitude: c.longitude, latitude: c.latitude, zoom: 6, pitch: 0, bearing: 0 })
                        }}
                      >
                        #{c.rank} {c.latitude.toFixed(2)}, {c.longitude.toFixed(2)} · priority {c.priority.toFixed(2)} ·
                        suitability {c.suitability.toFixed(2)} · effort {c.effort.toFixed(2)}
                      </button>
                    </li>
                  ))}
                </ol>
              </>
            ) : (
              <p className="text-muted-foreground">Not available: {pr.reason}</p>
            )}
          </section>
        )}
        {ia && (
          <section>
            <h4 className="font-medium">Inferred absences</h4>
            {ia.status === 'evaluated' ? (
              <p>
                {formatNumber(ia.n ?? 0)} well-surveyed native-range cells without the species ·
                specificity {pct(ia.specificity)} · AUC {num(ia.auc_training_presence_vs_inferred, 2)}
              </p>
            ) : (
              <p className="text-muted-foreground">
                {ia.status}
                {ia.reason ? `: ${ia.reason}` : ''}
              </p>
            )}
            {ia.note && <p className="text-[11px] text-muted-foreground">{ia.note}</p>}
          </section>
        )}
      </CardContent>
    </Card>
  )
}

export function ResponseCurvesCard({
  model,
  descriptions,
}: {
  model: ModelVersionDetail
  descriptions?: Record<string, string>
}) {
  if (!model.response_curves) return null
  return (
    <Card>
      <CardHeader>
        <CardTitle>Response curves</CardTitle>
        <CardDescription>
          {model.response_curves.method}. Judge whether the responses are biologically plausible;
          the shaded band is where the presences lie.
        </CardDescription>
      </CardHeader>
      <CardContent>
        <ResponseCurvesChart curves={model.response_curves} descriptions={descriptions} />
      </CardContent>
    </Card>
  )
}

/** U8: downloads offered by the backend for one version. */
export function ExportsCard({ taxonKey, version }: { taxonKey: number; version: number }) {
  const exports = useExports(taxonKey, version)
  const items = exports.data?.items ?? []
  const main = items.filter((i) => !i.id.includes(':') || i.id.startsWith('binary:p10'))
  const more = items.filter((i) => !main.includes(i))
  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Download className="size-4" /> Downloads and standards
        </CardTitle>
        <CardDescription>
          Cloud-optimised GeoTIFFs, the records as GeoPackage, an ODMAP protocol, a STAC item and
          the citation bundle (GBIF DOI) for v{version}.
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-3 text-xs">
        {exports.isError && <p className="text-destructive">{exports.error.message}</p>}
        <ul className="grid gap-1 sm:grid-cols-2">
          {main.map((i) => (
            <li key={i.id}>
              <a className="text-primary hover:underline" href={absoluteApiUrl(i.url)} download>
                {i.label}
              </a>
              <div className="text-[11px] text-muted-foreground">{i.description}</div>
            </li>
          ))}
        </ul>
        {more.length > 0 && (
          <details>
            <summary className="cursor-pointer text-muted-foreground">
              {more.length} more (other thresholds, scenarios)
            </summary>
            <ul className="mt-1 space-y-0.5">
              {more.map((i) => (
                <li key={i.id}>
                  <a className="text-primary hover:underline" href={absoluteApiUrl(i.url)} download>
                    {i.label}
                  </a>
                </li>
              ))}
            </ul>
          </details>
        )}
      </CardContent>
    </Card>
  )
}
