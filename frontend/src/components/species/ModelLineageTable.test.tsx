import { render, screen, within } from '@testing-library/react'
import { describe, expect, it } from 'vitest'

import type { ModelVersionOut } from '@/api/types'

import { lineageColumns } from './lineageColumns'
import { ModelLineageTable } from './ModelLineageTable'

function version(v: number, overrides: Partial<ModelVersionOut> = {}): ModelVersionOut {
  return {
    model_version: v,
    model_type: 'B',
    confidence_label: 'standard',
    trained_ts: `2026-0${v}-01T10:00:00Z`,
    trigger: v === 1 ? 'initial training' : 'delta 40 records (11.0% > 5%)',
    auc_mean: 0.8 + v / 100,
    auc_std: 0.05,
    cbi_mean: 0.7,
    cbi_std: 0.1,
    tss_mean: 0.5,
    tss_std: 0.08,
    n_presence: 250,
    n_background: 10000,
    n_cv_folds: 4,
    feature_classes: 'LQH',
    beta_multiplier: 1,
    severity_score: 55,
    mlflow_run_id: `run-${v}`,
    reproducibility: {
      bioclim_version_used: 'bioclim_v1',
      bioclim_sha256: 'abc123',
      bioclim_source: 'WorldClim v2.1',
      gbif_download_key: '0001234-260101',
      gbif_download_doi: '10.15468/dl.xyz',
      gbif_citation: 'GBIF.org (01 January 2026) GBIF Occurrence Download https://doi.org/10.15468/dl.xyz',
      package_versions: { elapid: '1.0.4', pygbif: '0.6.6', pyinaturalist: '0.21.1', rasterio: '1.5.0' },
      random_seeds: { background_sampling: 43, cv_fold_assignment: 42 },
      training_config: {},
      native_range_source: 'user_edit',
    },
    ...overrides,
  }
}

describe('model lineage table', () => {
  it('defines columns for every reproducibility field', () => {
    const ids = lineageColumns.map((c) => c.id ?? ('accessorKey' in c ? c.accessorKey : undefined))
    expect(ids).toEqual(
      expect.arrayContaining([
        'model_version',
        'bioclim_version_used',
        'gbif_download',
        'training_data',
        'package_versions',
        'random_seeds',
        'cbi_mean',
        'auc_mean',
        'tss_mean',
      ]),
    )
  })

  it('renders reproducibility metadata verbatim, newest version first', () => {
    render(<ModelLineageTable versions={[version(1), version(2, { model_type: 'A' })]} />)
    const rows = screen.getAllByRole('row').slice(1)
    expect(within(rows[0]!).getByText('v2')).toBeInTheDocument()
    expect(within(rows[0]!).getByText('A · native only')).toBeInTheDocument()
    expect(within(rows[1]!).getByText('v1')).toBeInTheDocument()

    expect(screen.getAllByText('doi:10.15468/dl.xyz')[0]).toHaveAttribute(
      'href',
      'https://doi.org/10.15468/dl.xyz',
    )
    expect(screen.getAllByText('1.0.4')).toHaveLength(2) // elapid version, not summarised
    expect(screen.getAllByText('43')).toHaveLength(2) // background seed
    expect(screen.getAllByText('bioclim_v1')).toHaveLength(2)
    expect(screen.getByText('run-2')).toBeInTheDocument()
  })

  it('renders the background-sampling record verbatim, including fallbacks', () => {
    const base = version(1).reproducibility
    render(
      <ModelLineageTable
        versions={[
          version(1, { reproducibility: { ...base } }),
          version(2, {
            reproducibility: {
              ...base,
              background: {
                method: 'target_group_order',
                requested_method: 'target_group',
                buffer_km: 500,
                target_group: {
                  rank: 'order',
                  taxon_key: 408,
                  name: 'Lamiales',
                  n_records: 36011809,
                  n_pixels: 70730,
                  map_zoom: 2,
                  pixel_deg: 0.087890625,
                  n_tiles: 4,
                  query: 'https://api.gbif.org/v2/map/occurrence/density/2/{x}/{y}.mvt?taxonKey=408',
                  fetched_ts: '2026-10-03T14:00:00+00:00',
                },
              },
            },
          }),
          version(3, {
            reproducibility: {
              ...base,
              background: {
                method: 'buffer_500km',
                requested_method: 'target_group',
                buffer_km: 500,
                fallback_reason: 'only 120 cells with order Lamiales records',
              },
            },
          }),
        ]}
      />,
    )
    expect(screen.getByText('target_group_order')).toBeInTheDocument()
    expect(screen.getByText('target group order Lamiales (taxonKey 408)')).toBeInTheDocument()
    expect(screen.getByText(/density\/2\/\{x\}\/\{y\}\.mvt\?taxonKey=408/)).toBeInTheDocument()
    expect(screen.getByText('fallback: only 120 cells with order Lamiales records')).toBeInTheDocument()
    expect(screen.getByText('not recorded (trained before background logging)')).toBeInTheDocument()
  })

  it('shows an empty state', () => {
    render(<ModelLineageTable versions={[]} />)
    expect(screen.getByText('No model versions yet.')).toBeInTheDocument()
  })
})
