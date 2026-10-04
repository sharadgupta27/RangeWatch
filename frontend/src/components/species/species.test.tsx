import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { render, screen, waitFor } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import { describeStatus } from './StatusBadge'
import { normalizedWeights, severityConfigSchema } from './severitySchema'

// The generated API client is mocked: unit tests never call the real backend.
const GET = vi.fn()
vi.mock('@/api/client', async (orig) => ({
  ...(await orig<typeof import('@/api/client')>()),
  api: { GET: (...args: unknown[]) => GET(...args) },
}))

describe('status badge (presentation only)', () => {
  it('describes pending records and retrain dates', () => {
    expect(describeStatus({ status: 'minor_update_pending', pendingRecords: 12 }).label).toBe(
      '12 new records pending',
    )
    expect(describeStatus({ status: 'minor_update_pending', pendingRecords: 1 }).label).toBe(
      '1 new record pending',
    )
    expect(describeStatus({ status: 'up_to_date', trainedTs: '2026-03-01T00:00:00Z' }).label).toMatch(
      /^Up to date · retrained/,
    )
    expect(describeStatus({ status: 'awaiting_native_range_review' }).variant).toBe('warning')
    expect(describeStatus({ status: 'failed' }).variant).toBe('destructive')
    expect(describeStatus({ status: 'up_to_date', jobActive: true }).label).toBe(
      'Checking for updates…',
    )
  })
})

describe('severity config schema', () => {
  const valid = {
    weights: { suitability: 0.35, climate_analogy: 0.25, spread_rate: 0.2, ecological_impact_prior: 0.2 },
    impact_prior: 0.5,
    impact_prior_source: 'GISD',
    suitability_saturation: 0.1,
    spread_half_saturation: 5,
    spread_window_years: 10,
    confidence_low_extrapolated: 0.25,
    confidence_moderate_extrapolated: 0.1,
    confidence_min_analog: 0.6,
    confidence_max_mess_extrapolated: 0.25,
    confidence_min_cbi: 0.2,
  }

  it('accepts valid configs and rejects invalid ones', () => {
    expect(severityConfigSchema.safeParse(valid).success).toBe(true)
    expect(
      severityConfigSchema.safeParse({
        ...valid,
        weights: { suitability: 0, climate_analogy: 0, spread_rate: 0, ecological_impact_prior: 0 },
      }).success,
    ).toBe(false)
    expect(severityConfigSchema.safeParse({ ...valid, impact_prior: 1.2 }).success).toBe(false)
    expect(severityConfigSchema.safeParse({ ...valid, impact_prior_source: ' ' }).success).toBe(false)
    expect(
      severityConfigSchema.safeParse({ ...valid, weights: { ...valid.weights, spread_rate: -1 } })
        .success,
    ).toBe(false)
  })

  it('keeps the confidence thresholds ordered (moderate ≤ low)', () => {
    const bad = severityConfigSchema.safeParse({ ...valid, confidence_moderate_extrapolated: 0.4 })
    expect(bad.success).toBe(false)
    expect(bad.error?.issues[0]?.path).toEqual(['confidence_moderate_extrapolated'])
    expect(severityConfigSchema.safeParse({ ...valid, confidence_min_analog: 1.5 }).success).toBe(false)
    expect(severityConfigSchema.safeParse({ ...valid, confidence_min_cbi: -0.5 }).success).toBe(true)
    expect(severityConfigSchema.safeParse({ ...valid, confidence_min_cbi: 1.5 }).success).toBe(false)
  })

  it('normalises weights for display', () => {
    const n = normalizedWeights({ suitability: 2, climate_analogy: 1, spread_rate: 1, ecological_impact_prior: 0 })
    expect(n.suitability).toBeCloseTo(0.5)
    expect(Object.values(n).reduce((a, b) => a + b, 0)).toBeCloseTo(1)
  })
})

describe('occurrence table', () => {
  beforeEach(() => GET.mockReset())

  it('requests server-side pages through the typed client', async () => {
    GET.mockResolvedValue({
      data: {
        items: [
          {
            source: 'GBIF',
            record_id: 99,
            longitude: 10.5,
            latitude: 45.25,
            event_date: '2020-05-01',
            coordinate_uncertainty_m: 30,
            basis_of_record: 'HUMAN_OBSERVATION',
            country_code: 'IT',
            range_label: 'introduced',
            ingested_ts: '2026-01-01T00:00:00Z',
          },
        ],
        total: 120,
        offset: 0,
        limit: 25,
      },
      response: new Response(null, { status: 200 }),
    })
    const { OccurrenceTable } = await import('./OccurrenceTable')
    render(
      <QueryClientProvider client={new QueryClient()}>
        <OccurrenceTable taxonKey={7} />
      </QueryClientProvider>,
    )
    await waitFor(() => expect(screen.getByText('#99')).toBeInTheDocument())
    expect(screen.getByText('introduced')).toBeInTheDocument()
    expect(screen.getByText('1–25 of 120')).toBeInTheDocument()
    expect(GET).toHaveBeenCalledWith('/species/{taxon_key}/occurrences', {
      params: {
        path: { taxon_key: 7 },
        query: { offset: 0, limit: 25, source: undefined, range_label: undefined },
      },
    })
  })
})
