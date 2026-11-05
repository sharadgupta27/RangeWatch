import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { render, screen, within } from '@testing-library/react'
import type { ReactNode } from 'react'
import { describe, expect, it, vi } from 'vitest'

import type {
  LayerSet,
  ModelCardOut,
  ModelVersionDetail,
  RasterLayer,
  SpeciesListOut,
  TargetGroupComparisonOut,
} from '@/api/types'
import { productLayers, resolveProjection } from '@/components/maps/DualMapView'
import { TooltipProvider } from '@/components/ui/tooltip'

import { independentSummary, TargetGroupTable } from './BiasStudiesCard'
import { parseSpeciesEntries, speciesListSchema, toSpeciesListIn } from './listSchema'
import { ModelCardView } from './ModelCardPanel'
import { AlgorithmCard } from './ModelInsightsCard'
import { listSummary } from './SpeciesListsCard'

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

describe('multi-GCM ensembles (U7)', () => {
  const layers = {
    suitability: raster('suitability'),
    extrapolation: raster('extrapolation'),
    products: [raster('limiting')],
    scenarios: [
      {
        id: 'scenario:ensemble_ssp245_2041-2060',
        name: 'ensemble_ssp245_2041-2060',
        label: 'GCM ensemble ssp245 2041-2060 (2 GCMs)',
        suitability: raster('suit-ens'),
        extrapolation: raster('extra-ens'),
        change: raster('change:ensemble_ssp245_2041-2060'),
        gcm_ensemble: { n_gcms: 2, agreement: raster('gcm-agreement:ensemble_ssp245_2041-2060') },
      },
    ],
  } as unknown as LayerSet

  it('offers the agreement map with the ensemble’s own extrapolation mask', () => {
    expect(productLayers(layers).map((l) => l.id)).toEqual([
      'limiting',
      'change:ensemble_ssp245_2041-2060',
      'gcm-agreement:ensemble_ssp245_2041-2060',
    ])
    const agree = resolveProjection(layers, 'product:gcm-agreement:ensemble_ssp245_2041-2060')
    expect([agree.base?.id, agree.extrapolation?.id]).toEqual([
      'gcm-agreement:ensemble_ssp245_2041-2060',
      'extra-ens',
    ])
    expect(resolveProjection(layers, 'product:limiting').extrapolation?.id).toBe('extrapolation')
  })
})

describe('species lists (U9)', () => {
  it('parses names and GBIF keys, dropping blanks and duplicates', () => {
    expect(parseSpeciesEntries('Lantana camara\n\n2891932; lantana  camara, Carpobrotus edulis')).toEqual({
      names: ['Lantana camara', 'Carpobrotus edulis'],
      taxon_keys: [2891932],
    })
    expect(toSpeciesListIn({ name: ' EU list ', source: '', entries: '123' })).toEqual({
      name: 'EU list',
      source: null,
      names: [],
      taxon_keys: [123],
    })
  })

  it('validates the form before sending', () => {
    expect(speciesListSchema.safeParse({ name: '', source: '', entries: 'a b' }).success).toBe(false)
    expect(speciesListSchema.safeParse({ name: 'x', source: '', entries: ' \n ' }).success).toBe(false)
    const many = Array.from({ length: 201 }, (_, i) => String(i + 1)).join('\n')
    expect(speciesListSchema.safeParse({ name: 'x', source: '', entries: many }).success).toBe(false)
    expect(speciesListSchema.safeParse({ name: 'x', source: '', entries: 'Lantana camara' }).success).toBe(true)
  })

  it('summarises a list', () => {
    const l = { n_species: 3, n_registered: 2, n_trained: 1, n_unresolved: 1 } as SpeciesListOut
    expect(listSummary(l)).toBe('3 species · 2 tracked · 1 with a model · 1 not matched in GBIF')
  })
})

describe('model card (U13)', () => {
  const card: ModelCardOut = {
    taxon_key: 1,
    scientific_name: 'Testus invasivus',
    model_version: 2,
    generated_ts: '2026-11-02T00:00:00Z',
    confidence_label: 'lower',
    model_type: 'A',
    headline: 'Testus invasivus: lower-confidence climate suitability model, version 2',
    transferability_caveat: 'Native-range-only models transfer moderately.',
    caveats: ['Native-range-only models transfer moderately.', 'Climate only.'],
    sections: [
      { id: 'what', title: 'What the map shows', paragraphs: ['Scores 0–1.'], facts: [{ label: 'Method', value: 'MaxEnt' }] },
    ],
    citation: 'GBIF.org (2026)',
    source: 'Generated from stored metrics.',
  }

  it('shows the confidence label and caveats before the sections', () => {
    wrap(<ModelCardView card={card} />)
    expect(screen.getByText('lower confidence')).toBeInTheDocument()
    const note = screen.getByRole('note')
    expect(within(note).getByText('Climate only.')).toBeInTheDocument()
    const section = screen.getByText('What the map shows')
    expect(note.compareDocumentPosition(section) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy()
    expect(screen.getByText('MaxEnt')).toBeInTheDocument()
  })
})

describe('bias studies (B1, B8, B9)', () => {
  const report = {
    comparison_id: 'c1',
    taxon_key: 1,
    created_ts: '2026-11-02T00:00:00Z',
    bioclim_version: 'v1',
    configured_rank: 'order',
    best_by_cv_cbi: 'class',
    criterion: 'highest CV CBI',
    note: 'nothing published',
    reference: 'Phillips et al. 2009',
    options: [
      { option: 'order', rank: 'order', match: false, background_method: 'target_group_order', cv_cbi_mean: 0.4, cv_auc_mean: 0.8, independent: [] },
      {
        option: 'class',
        rank: 'class',
        match: false,
        background_method: 'target_group_class',
        cv_cbi_mean: 0.6,
        cv_auc_mean: 0.82,
        independent: [{ name: 'Survey', metrics: { auc_presence_absence: 0.71, cbi: 0.5 } }],
      },
      { option: 'phylum', rank: 'phylum', match: false, error: 'ValueError: too few presences' },
    ],
  } as unknown as TargetGroupComparisonOut

  it('compares target groups with CV and independent scores', () => {
    wrap(<TargetGroupTable report={report} />)
    expect(screen.getByText('best CV')).toBeInTheDocument()
    expect(screen.getByText('ValueError: too few presences')).toBeInTheDocument()
    expect(independentSummary(report.options[1]!)).toBe('Survey: AUC 0.71 · CBI 0.50')
    expect(independentSummary(report.options[0]!)).toBe('no independent data')
  })

  it('reports the point-process offset test', () => {
    const model = {
      algorithm: 'ppm',
      point_process: {
        method: 'Poisson point process',
        reference: 'Renner et al. 2015',
        features: 'LQ',
        bias_mode: 'offset',
        l2_penalty: 1,
        converged: true,
        iterations: 30,
        neg_log_likelihood: 1,
        n_presences: 120,
        n_quadrature: 1500,
        coefficients: {},
        free_bias_coefficients: { bias_travel_time: -0.4 },
        fixed_bias_coefficients: { bias_target_group_effort: 1 },
        offset: '1·bias_target_group_effort',
        offset_test: {
          column: 'bias_target_group_effort',
          assumed_coefficient: 1,
          free_coefficient: 0.42,
          lr_statistic: 12.3,
          p_value: 0.0004,
          interpretation: 'the data reject the assumed effort coefficient',
        },
        prediction: 'offset dropped',
        presence_weights_used: false,
      },
    } as unknown as ModelVersionDetail
    wrap(<AlgorithmCard model={model} />)
    expect(screen.getByText(/Point process with effort offset/)).toBeInTheDocument()
    expect(screen.getByText('0.420')).toBeInTheDocument()
    expect(screen.getByText(/bias_travel_time -0.400/)).toBeInTheDocument()
  })
})
