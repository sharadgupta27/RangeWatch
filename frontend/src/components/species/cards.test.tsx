import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import type { ReactNode } from 'react'
import { describe, expect, it, vi } from 'vitest'

import type {
  BioclimInfo,
  ClimateCrossCheckOut,
  ExtrapolationSummaryOut,
  LayerSet,
  ModelVersionDetail,
} from '@/api/types'
import { TooltipProvider } from '@/components/ui/tooltip'
import { mapActions, mapStore } from '@/store/mapStore'

import { CrossCheckCard } from './CrossCheckCard'
import { ExtrapolationCard } from './ExtrapolationCard'
import { estimateCells, HiresCard } from './HiresCard'

// The generated API client is mocked: unit tests never call the real backend.
vi.mock('@/api/client', async (orig) => ({
  ...(await orig<typeof import('@/api/client')>()),
  api: { GET: vi.fn(), POST: vi.fn() },
}))

const wrap = (ui: ReactNode) =>
  render(<QueryClientProvider client={new QueryClient()}>{ui}</QueryClientProvider>)

const raster = (id: string) => ({
  id,
  label: id,
  kind: 'continuous' as const,
  tile_url: `http://t/${id}/{z}/{x}/{y}.png`,
  legend: [],
  description: '',
})

const baseLayers: LayerSet = {
  taxon_key: 1,
  model_version: 1,
  occurrences: { id: 'occurrences', label: 'o', tile_url: 'x', source_layer: 'occurrences' },
  native_range: { id: 'native_range', label: 'n', tile_url: 'x', source_layer: 'native_range' },
  scenarios: [],
  scenarios_available: [],
}

describe('high-resolution card', () => {
  it('estimates the cell count of a view at a resolution', () => {
    expect(estimateCells([0, 0, 10, 5], '30s')).toBe(1200 * 600)
    expect(estimateCells([0, 0, 10, 5], '10m')).toBe(60 * 30)
    expect(estimateCells([0, 0, 10, 5], 'custom')).toBeNull()
  })

  it('explains how to install a stack when none is registered', () => {
    wrap(<HiresCard taxonKey={1} layers={baseLayers} jobActive={false} />)
    expect(screen.getByText('bioclim_store add-hires')).toBeInTheDocument()
    expect(screen.queryByRole('button', { name: /current view/ })).toBeNull()
  })

  it('blocks views that are too large and shows the current region', () => {
    mapActions.setViewBounds([-180, -60, 180, 80])
    wrap(
      <HiresCard
        taxonKey={1}
        layers={{
          ...baseLayers,
          hires_available: '30s',
          hires_max_cells: 16_000_000,
          hires: {
            resolution: '30s',
            bbox: [5, 38, 25, 52],
            suitability: raster('hires'),
            extrapolation: raster('extrapolation:hires'),
            extrapolated_land_fraction: 0.4,
            created_ts: new Date().toISOString(),
          },
        }}
        jobActive={false}
      />,
    )
    expect(screen.getByRole('button', { name: /Re-project 30″ for current view/ })).toBeDisabled()
    expect(
      screen.getByText(new RegExp(`zoom in to at most ${(16_000_000).toLocaleString()}`)),
    ).toBeInTheDocument()
    expect(screen.getByText('40% of land extrapolated')).toBeInTheDocument()
    expect(screen.getByText('5.0°E–25.0°E, 38.0°N–52.0°N')).toBeInTheDocument()

    mapActions.setViewBounds([5, 38, 25, 52])
    cleanup()
    wrap(
      <HiresCard
        taxonKey={1}
        layers={{ ...baseLayers, hires_available: '30s', hires_max_cells: 16_000_000 }}
        jobActive={false}
      />,
    )
    expect(screen.getByRole('button', { name: /Project 30″ for current view/ })).toBeEnabled()
  })
})

const crosscheck: ClimateCrossCheckOut = {
  alt_bioclim_version: 'chelsa_v1',
  alt_source: 'CHELSA V2.1',
  primary_bioclim_version: 'bioclim_v2',
  created_ts: new Date().toISOString(),
  n_presence: 250,
  n_background: 9800,
  n_dropped: 3,
  feature_classes: 'LQH',
  beta_multiplier: 2,
  predictors: [
    { predictor: 'bio1', pearson_r: 0.98, mean_diff: 0.3, mean_abs_diff: 0.6, scale_ratio: 1.02, units_suspect: false },
    { predictor: 'bio15', pearson_r: 0.4, mean_diff: 12, mean_abs_diff: 20, scale_ratio: 3.1, units_suspect: true },
  ],
  primary: { auc_mean: 0.81, cbi_mean: 0.72, tss_mean: 0.5 },
  alternative: { auc_mean: 0.79, cbi_mean: 0.65, tss_mean: 0.48, threshold: 0.2 },
  suitability_rank_correlation: 0.71,
  classification_agreement: 0.83,
  classification_kappa: 0.55,
  verdict: 'moderate',
  interpretation: 'Range edges are sensitive to the climate data.',
}

const model = (crosschecks: ClimateCrossCheckOut[]) =>
  ({ model_version: 3, crosschecks }) as unknown as ModelVersionDetail

describe('climate cross-check card', () => {
  it('shows the verdict, both sets of CV metrics and flags suspect units', () => {
    const bioclim = { crosscheck_version: 'chelsa_v1', crosscheck_source: 'CHELSA V2.1' } as BioclimInfo
    wrap(<CrossCheckCard taxonKey={1} model={model([crosscheck])} bioclim={bioclim} jobActive={false} />)
    expect(screen.getByText('moderate')).toBeInTheDocument()
    expect(screen.getByText(crosscheck.interpretation)).toBeInTheDocument()
    expect(screen.getByText('0.810')).toBeInTheDocument()
    expect(screen.getByText('0.790')).toBeInTheDocument()
    expect(screen.getByText('0.710')).toBeInTheDocument()
    expect(screen.getAllByText('units?')).toHaveLength(1)
    expect(screen.getByRole('button', { name: /Re-run cross-check against chelsa_v1/ })).toBeEnabled()
  })

  it('explains how to enable it when no cross-check stack is installed', () => {
    wrap(<CrossCheckCard taxonKey={1} model={model([])} bioclim={{} as BioclimInfo} jobActive={false} />)
    expect(screen.getByText('Not cross-checked yet.')).toBeInTheDocument()
    expect(screen.getByText('bioclim_store build-chelsa')).toBeInTheDocument()
    expect(screen.queryByRole('button')).toBeNull()
  })
})

const diag = (
  id: 'mess' | 'exdet' | 'mop' | 'shape' | 'aoa',
  label: string,
  threshold: number,
  candidate: number,
) => ({
  id,
  label,
  name: `${label} long name`,
  reference: 'Someone et al.',
  flag_rule: `${label} rule`,
  threshold,
  threshold_rule: 'outlier-trimmed maximum',
  land_area_km2: 1000,
  land_fraction: 0.2,
  candidate_flagged_fraction: candidate,
})

const extrapolation: ExtrapolationSummaryOut = {
  reference_set: 'training presences + background (as for MESS)',
  n_reference: 10250,
  n_cv_folds: 4,
  mop_percentage: 1,
  mop_k: 103,
  aoa_weights: { bio1: 1, bio12: 0.4 },
  diagnostics: [
    diag('mess', 'MESS', 0, 0.1),
    diag('exdet', 'exDet', 1, 0.15),
    diag('mop', 'MOP', 1.23, 0.3),
    diag('shape', 'Shape', 87.5, 0.25),
    diag('aoa', 'AOA', 0.412, 0.35),
  ],
  consensus_land_area_km2: [10, 5, 4, 3, 2, 1],
  candidate_consensus_ok_fraction: 0.6,
  candidate_consensus_majority_fraction: 0.3,
  exdet_univariate_area_km2: 500,
  exdet_combinatorial_area_km2: 1200,
  top_combinatorial_variables: [{ variable: 'bio12', area_km2: 900 }],
}

describe('extrapolation diagnostics card', () => {
  it('lists every diagnostic with its rule, threshold and flagged shares', () => {
    wrap(
      <TooltipProvider>
        <ExtrapolationCard summary={extrapolation} />
      </TooltipProvider>,
    )
    for (const label of ['MESS', 'exDet', 'MOP', 'Shape', 'AOA']) {
      expect(screen.getByText(`${label} rule`)).toBeInTheDocument()
    }
    expect(screen.getByText('1.23')).toBeInTheDocument()
    expect(screen.getByText('0.412')).toBeInTheDocument()
    expect(screen.getByText('60.0% of candidate zone flagged by none')).toBeInTheDocument()
    expect(screen.getByText('30.0% flagged by ≥ 3 of 5')).toBeInTheDocument()
    expect(screen.getByText(/BIO1 1.00 · BIO12 0.40/)).toBeInTheDocument()
    expect(screen.getByText(/driven mainly by BIO12/)).toBeInTheDocument()
  })

  it('opens a diagnostic on the projection map', () => {
    wrap(
      <TooltipProvider>
        <ExtrapolationCard summary={extrapolation} />
      </TooltipProvider>,
    )
    fireEvent.click(screen.getByRole('button', { name: 'Show AOA on the map' }))
    expect(mapStore.state.projection).toBe('aoa')
    mapActions.resetForSpecies()
  })
})
