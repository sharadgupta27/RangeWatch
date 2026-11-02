import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { render, screen } from '@testing-library/react'
import type { ReactNode } from 'react'
import { describe, expect, it, vi } from 'vitest'

import { datasetFormData } from '@/api/queries'
import type {
  DataQualityOut,
  IndependentMetricsOut,
  ReferenceDatasetOut,
  SamplingEffortOut,
} from '@/api/types'
import { TooltipProvider } from '@/components/ui/tooltip'

import { DataQualityCard, SamplingEffortCard } from './DataQualityCard'
import { datasetUploadSchema, EMPTY_UPLOAD } from './datasetSchema'
import { EvaluationTable } from './ReferenceDatasetsPanel'

// The generated API client is mocked: unit tests never call the real backend.
vi.mock('@/api/client', async (orig) => ({
  ...(await orig<typeof import('@/api/client')>()),
  api: { GET: vi.fn(), POST: vi.fn(), DELETE: vi.fn() },
}))

const wrap = (ui: ReactNode) =>
  render(
    <QueryClientProvider client={new QueryClient()}>
      <TooltipProvider>{ui}</TooltipProvider>
    </QueryClientProvider>,
  )

const quality: DataQualityOut = {
  version: 'quality_v1',
  config: { exclude: ['capital', 'institution'] },
  reference: null,
  n_records: 1200,
  n_excluded: 12,
  n_flagged_kept: 40,
  n_outside_climate_grid: 3,
  tiers: { A: 100, B: 800, C: 288, excluded: 12 },
  tier_rules: { A: 'specimens', B: 'others', C: 'flags', excluded: 'error tests' },
  tests: [
    {
      id: 'institution',
      label: 'Biodiversity institution',
      severity: 'error',
      description: 'near a herbarium',
      excluded: true,
      status: 'skipped',
      reason: "reference table 'institutions' not installed",
      n_flagged: 0,
    },
    {
      id: 'spatial_isolate',
      label: 'Spatially isolated',
      severity: 'warning',
      description: 'no neighbour',
      excluded: false,
      status: 'run',
      n_flagged: 7,
    },
  ],
  issue_counts: {},
  n_issues_unknown: 0,
  n_undated: 0,
  grid_cell_m: 18_553,
}

describe('data-quality card', () => {
  it('shows skipped tests with their reason instead of a pass', () => {
    wrap(<DataQualityCard quality={quality} />)
    expect(screen.getByText(/Skipped: reference table 'institutions' not installed/)).toBeInTheDocument()
    expect(screen.getByText('flags only')).toBeInTheDocument()
    expect(screen.getByText('12 excluded from training')).toBeInTheDocument()
    expect(screen.getByText(/no reference gazetteers installed/)).toBeInTheDocument()
  })

  it('explains versions trained before the tests existed', () => {
    wrap(<DataQualityCard quality={undefined} />)
    expect(screen.getByText(/trained before the data-quality tests existed/)).toBeInTheDocument()
  })

  it('reports per-zone unsampled shares, or why effort is unavailable', () => {
    const effort: SamplingEffortOut = {
      status: 'available',
      target_group: { rank: 'order', name: 'Fabales' },
      n_records: 5_000_000,
      pixel_deg: 0.088,
      map_zoom: 2,
      land_unsampled_fraction: 0.4,
      candidate_unsampled_fraction: 0.625,
    }
    const { unmount } = wrap(<SamplingEffortCard effort={effort} />)
    expect(screen.getByText('62.5%')).toBeInTheDocument()
    expect(screen.getByText(/order Fabales/)).toBeInTheDocument()
    unmount()
    wrap(<SamplingEffortCard effort={{ status: 'unavailable', reason: 'map API stalled' }} />)
    expect(screen.getByText(/map API stalled/)).toBeInTheDocument()
  })
})

const metrics = (): IndependentMetricsOut => ({
  threshold: 0.3,
  n_presence: 40,
  n_absence: 0,
  n_outside_grid: 2,
  sensitivity: 0.75,
  cbi: 0.61,
  sensitivity_analog: 0.9,
  sensitivity_extrapolated: 0.2,
})

describe('independent evaluation table', () => {
  it('lists every model version with presence metrics only when there are no absences', () => {
    const ds: ReferenceDatasetOut = {
      taxon_key: 1,
      dataset_id: 'upload-1',
      kind: 'upload',
      role: 'evaluation',
      name: 'INPA',
      n_present: 45,
      n_absent: 0,
      created_ts: '2026-10-01T00:00:00Z',
      updated_ts: '2026-10-01T00:00:00Z',
      evaluations: [
        { model_version: 2, model_type: 'B', evaluated_ts: 'x', n_records: 45, n_excluded_overlap: 3, metrics: metrics() },
        {
          model_version: 1,
          model_type: 'A',
          evaluated_ts: 'x',
          n_records: 45,
          n_excluded_overlap: 3,
          metrics: { ...metrics(), sensitivity: 0.5 },
        },
      ],
    }
    wrap(<EvaluationTable ds={ds} />)
    expect(screen.getByText('v2')).toBeInTheDocument()
    expect(screen.getByText('75%')).toBeInTheDocument()
    expect(screen.getByText('50%')).toBeInTheDocument()
    expect(screen.getAllByText('90% / 20%')).toHaveLength(2)
    expect(screen.queryByText('Specificity')).toBeNull()
    expect(screen.getAllByText('5')).toHaveLength(2) // 3 overlapping + 2 outside the grid
  })
})

describe('dataset upload', () => {
  it('validates the file and the name before sending', () => {
    const empty = datasetUploadSchema.safeParse(EMPTY_UPLOAD)
    expect(empty.success).toBe(false)
    const bad = new File(['x'], 'data.xlsx')
    expect(datasetUploadSchema.safeParse({ ...EMPTY_UPLOAD, file: bad, name: 'x' }).success).toBe(false)
    const ok = new File(['lon,lat\n1,2'], 'survey.csv')
    expect(datasetUploadSchema.safeParse({ ...EMPTY_UPLOAD, file: ok, name: 'Survey' }).success).toBe(true)
  })

  it('sends the file plus only the filled-in fields as multipart', () => {
    const file = new File(['lon,lat\n1,2'], 'survey.csv')
    const fd = datasetFormData(file, { name: ' Survey ', citation: '', latitude_column: 'Y' })
    expect((fd.get('file') as File).name).toBe('survey.csv')
    expect(fd.get('name')).toBe('Survey')
    expect(fd.get('latitude_column')).toBe('Y')
    expect(fd.has('citation')).toBe(false)
  })
})
