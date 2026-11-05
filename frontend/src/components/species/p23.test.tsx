import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { render, screen } from '@testing-library/react'
import type { ReactNode } from 'react'
import { describe, expect, it, vi } from 'vitest'

import type { LayerSet, ModelVersionDetail, RasterLayer } from '@/api/types'
import { productLayers, resolveProjection } from '@/components/maps/DualMapView'
import { TooltipProvider } from '@/components/ui/tooltip'

import { AlgorithmCard, MonitoringProductsCard, ThresholdsCard } from './ModelInsightsCard'
import { EffortClasses } from './MonitoringCards'
import { studyRegionSchema, subscriptionSchema } from './monitoringSchema'
import { ANNOTATION_ACTIONS, annotationSummary } from './OccurrenceTable'

// The generated API client is mocked: unit tests never call the real backend.
vi.mock('@/api/client', async (orig) => ({
  ...(await orig<typeof import('@/api/client')>()),
  api: { GET: vi.fn(), POST: vi.fn(), PUT: vi.fn(), DELETE: vi.fn() },
}))

const wrap = (ui: ReactNode) =>
  render(
    <QueryClientProvider client={new QueryClient()}>
      <TooltipProvider>{ui}</TooltipProvider>
    </QueryClientProvider>,
  )

const raster = (id: string): RasterLayer => ({
  id,
  label: id,
  kind: 'categorical',
  tile_url: `http://tiles/${id}/{z}/{x}/{y}.png`,
  legend: [],
  description: id,
})

describe('map products', () => {
  const layers = {
    suitability: raster('suitability'),
    extrapolation: raster('extrapolation'),
    consensus_overlay: raster('consensus-overlay'),
    products: [raster('binary:p10'), raster('limiting'), raster('priority')],
    scenarios: [
      {
        id: 'scenario:ssp585',
        name: 'ssp585',
        label: 'ssp585',
        suitability: raster('suit-ssp585'),
        extrapolation: raster('extra-ssp585'),
        change: raster('change:ssp585'),
      },
    ],
  } as unknown as LayerSet

  it('lists products plus scenario change maps and keeps the extrapolation overlay', () => {
    expect(productLayers(layers).map((l) => l.id)).toEqual([
      'binary:p10',
      'limiting',
      'priority',
      'change:ssp585',
    ])
    const lim = resolveProjection(layers, 'product:limiting')
    expect([lim.base?.id, lim.extrapolation?.id]).toEqual(['limiting', 'extrapolation'])
    expect(resolveProjection(layers, 'product:limiting', 'consensus').extrapolation?.id).toBe(
      'consensus-overlay',
    )
    // a scenario change map carries that scenario's own extrapolation mask
    const ch = resolveProjection(layers, 'product:change:ssp585')
    expect([ch.base?.id, ch.extrapolation?.id]).toEqual(['change:ssp585', 'extra-ssp585'])
    expect(resolveProjection(layers, 'product:missing').base).toBeNull()
  })
})

describe('record annotations', () => {
  it('summarises the effective annotation and offers append-only actions', () => {
    expect(annotationSummary(null)).toEqual([])
    expect(
      annotationSummary({ cultivated: true, verified: false, range_label: 'introduced', n_annotations: 2 }),
    ).toEqual(['cultivated', 'label → introduced'])
    const kinds = new Set(ANNOTATION_ACTIONS.map((a) => a.body.kind))
    expect(kinds).toEqual(new Set(['cultivated', 'exclude', 'verified', 'range_label']))
  })
})

describe('monitoring forms', () => {
  it('validates subscriptions and study regions before sending', () => {
    expect(subscriptionSchema.safeParse({ channel: 'webhook', target: 'http://x.org', allSpecies: false }).success).toBe(false)
    expect(subscriptionSchema.safeParse({ channel: 'webhook', target: 'https://x.org/h', allSpecies: false }).success).toBe(true)
    expect(subscriptionSchema.safeParse({ channel: 'email', target: 'nope', allSpecies: true }).success).toBe(false)
    const ok = { west: '-20', south: '25', east: '50', north: '65', resolution: 'base' }
    expect(studyRegionSchema.parse(ok)).toMatchObject({ west: -20, north: 65 })
    expect(studyRegionSchema.safeParse({ ...ok, east: '-30' }).success).toBe(false)
    expect(studyRegionSchema.safeParse({ ...ok, north: '95' }).success).toBe(false)
  })

  it('shows B10 effort classes', () => {
    wrap(<EffortClasses classes={{ expansion: 2, discovery: 1, unknown: 0 }} />)
    expect(screen.getByText('2 likely expansion')).toBeInTheDocument()
    expect(screen.getByText('1 likely discovery')).toBeInTheDocument()
    expect(screen.queryByText(/effort unknown/)).toBeNull()
  })
})

const model = {
  model_version: 2,
  algorithm: 'ensemble',
  threshold: 0.2,
  threshold_rule: '10th percentile training presence',
  thresholds: {
    p10: { label: '10th percentile training presence', value: 0.2, sensitivity: 0.9, specificity_background: 0.7 },
    max_tss: { label: 'maximum TSS', value: 0.35, sensitivity: 0.8, specificity_background: 0.85 },
  },
  ensemble: {
    weighting: 'weight ∝ (spatial-CV AUC − 0.5)',
    cv_test_set: 'pseudo-absences',
    reference: 'Barbet-Massin et al. 2012',
    members: [
      { algorithm: 'maxent', label: 'MaxEnt', included: true, weight: 0.6, threshold: 0.2, auc_mean: 0.9, n_cv_folds: 3, params: {} },
      { algorithm: 'rf', label: 'Random forest', included: false, weight: 0, threshold: 0.3, auc_mean: 0.6, n_cv_folds: 3, params: {}, note: 'left out' },
    ],
  },
  change_previous: { status: 'available', from_version: 1, to_version: 2, gain_km2: 1200, loss_km2: 300, net_change_fraction: 0.05 },
  survey_priority: { status: 'unavailable', reason: 'no sampling-effort layer', top_cells: [] },
  inferred_absences: { status: 'insufficient', reason: '3 cells' },
  projection: { limiting_factor: { rule: 'Elith', land: [{ variable: 'bio1', area_km2: 10, fraction: 0.6 }], candidate: [] } },
} as unknown as ModelVersionDetail

describe('model insights', () => {
  it('lists every threshold rule and marks the one in use', () => {
    wrap(<ThresholdsCard model={model} />)
    expect(screen.getByText('maximum TSS')).toBeInTheDocument()
    expect(screen.getByText('used')).toBeInTheDocument()
    expect(screen.getAllByText('Show map')).toHaveLength(2)
  })

  it('shows ensemble members with weights and members left out', () => {
    wrap(<AlgorithmCard model={model} />)
    expect(screen.getByText('60%')).toBeInTheDocument()
    expect(screen.getByText('left out', { selector: 'td' })).toBeInTheDocument()
  })

  it('reports change, limiting factors, priority and inferred absences', () => {
    wrap(<MonitoringProductsCard model={model} />)
    expect(screen.getByText(/1,200 km² gained/)).toBeInTheDocument()
    expect(screen.getByText('bio1 60%')).toBeInTheDocument()
    expect(screen.getByText(/no sampling-effort layer/)).toBeInTheDocument()
    expect(screen.getByText(/insufficient: 3 cells/)).toBeInTheDocument()
  })
})
