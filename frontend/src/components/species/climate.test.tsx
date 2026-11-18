import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { render, screen, within } from '@testing-library/react'
import type { ReactNode } from 'react'
import { describe, expect, it, vi } from 'vitest'

import type {
  LayerSet,
  ModelVersionDetail,
  PredictorInfo,
  PredictorSetResultOut,
  SpeciesDetail,
} from '@/api/types'
import { TooltipProvider } from '@/components/ui/tooltip'

import { environmentSchema, groupPredictors, presentOnly } from './environmentSchema'
import { PredictorComparisonCard } from './PredictorComparisonCard'
import { ScenariosCard } from './ScenariosCard'

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

const pred = (
  name: string,
  group_label: string,
  future_available: boolean | null = true,
): PredictorInfo => ({
  name,
  description: name,
  group: group_label.toLowerCase(),
  group_label,
  kind: 'climate_band',
  future_available,
})

describe('climate data + predictor choice', () => {
  const offered = [
    pred('bio1', 'Temperature'),
    pred('bio12', 'Precipitation'),
    pred('hurs_mean', 'Humidity', false),
    pred('bio4', 'Temperature'),
  ]

  it('groups predictors by process in the version’s order', () => {
    expect(groupPredictors(offered).map(([g, ps]) => [g, ps.map((p) => p.name)])).toEqual([
      ['Temperature', ['bio1', 'bio4']],
      ['Precipitation', ['bio12']],
      ['Humidity', ['hurs_mean']],
    ])
  })

  it('flags chosen predictors without future data', () => {
    expect(presentOnly(['bio1', 'hurs_mean'], offered)).toEqual(['hurs_mean'])
    expect(presentOnly(['bio1', 'bio12'], offered)).toEqual([])
  })

  it('needs two predictors for a custom choice, none for automatic', () => {
    const base = { climateVersion: '', vifScreen: true }
    expect(environmentSchema.safeParse({ ...base, mode: 'custom', predictors: ['bio1'] }).success).toBe(false)
    expect(
      environmentSchema.safeParse({ ...base, mode: 'custom', predictors: ['bio1', 'bio12'] }).success,
    ).toBe(true)
    expect(environmentSchema.safeParse({ ...base, mode: 'auto', predictors: [] }).success).toBe(true)
  })
})

describe('predictor-set comparison', () => {
  const set = (name: string, rank: number, cbi: number, extra: Partial<PredictorSetResultOut> = {}) =>
    ({
      name,
      label: name,
      status: 'fitted',
      rank,
      requested: ['bio1', 'bio12'],
      predictors: ['bio1', 'bio12'],
      dropped_by_vif: [],
      present_only: [],
      cbi_mean: cbi,
      cbi_std: 0.05,
      auc_mean: 0.8,
      tss_mean: 0.5,
      ...extra,
    }) as PredictorSetResultOut
  const sp = {
    taxon_key: 1,
    climate_version: null,
    effective_climate_version: 'chelsa_v2',
    predictor_config: { mode: 'auto', predictors: [] },
  } as unknown as SpeciesDetail
  const model = {
    model_version: 3,
    predictor_comparison: {
      model_version: 3,
      bioclim_version: 'chelsa_v2',
      climate_family: 'chelsa',
      resolution: '2.5m',
      created_ts: '2026-11-18T10:00:00Z',
      best: 'humid',
      similar_to_best: ['temperature'],
      ranking_metric: 'spatial block CV CBI (then AUC)',
      vif_screen: true,
      note: 'Relative comparison on the same training data.',
      sets: [
        set('humid', 1, 0.71, { predictors: ['bio1', 'hurs_mean'], present_only: ['hurs_mean'] }),
        set('temperature', 2, 0.7, { dropped_by_vif: ['bio5'] }),
        { name: 'soil', label: 'soil', status: 'skipped', reason: 'too few presences', requested: [] },
      ],
    },
  } as unknown as ModelVersionDetail

  it('ranks sets with CBI ± SD and marks best, ties, present-only and VIF drops', () => {
    wrap(<PredictorComparisonCard sp={sp} model={model} jobActive={false} />)
    const rows = screen.getAllByRole('row').slice(1)
    expect(within(rows[0]!).getByText('best')).toBeInTheDocument()
    expect(within(rows[0]!).getByText('present only')).toBeInTheDocument()
    expect(within(rows[0]!).getByText('0.710 ± 0.050')).toBeInTheDocument()
    expect(within(rows[1]!).getByText('≈ best')).toBeInTheDocument()
    expect(within(rows[1]!).getByText(/VIF dropped bio5/)).toBeInTheDocument()
    expect(screen.getByText(/Skipped: soil \(too few presences\)/)).toBeInTheDocument()
  })

  it('invites a first run when no comparison exists', () => {
    wrap(
      <PredictorComparisonCard
        sp={sp}
        model={{ ...model, predictor_comparison: null } as ModelVersionDetail}
        jobActive={false}
      />,
    )
    expect(screen.getByText('Not run for this version yet.')).toBeInTheDocument()
  })
})

describe('scenarios a model cannot use', () => {
  it('shows the present-only reason instead of offering the projection', () => {
    const layers = {
      scenarios: [],
      scenarios_available: ['MPI-ESM1-2-HR_ssp585_2041-2070'],
      scenarios_unavailable: {
        'MPI-ESM1-2-HR_ssp585_2041-2070':
          'uses hurs_mean, which the climate source does not provide for this future (present-climate only)',
      },
    } as unknown as LayerSet
    wrap(<ScenariosCard taxonKey={1} layers={layers} jobActive={false} />)
    expect(screen.getByText('not projectable')).toBeInTheDocument()
    expect(screen.getByText(/uses hurs_mean/)).toBeInTheDocument()
    expect(screen.queryByRole('button', { name: /Project 1 scenario/ })).not.toBeInTheDocument()
  })
})
