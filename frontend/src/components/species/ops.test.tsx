import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { render, screen } from '@testing-library/react'
import type { ReactNode } from 'react'
import { describe, expect, it, vi } from 'vitest'

import type { JobOut } from '@/api/types'

import { confirmsName } from './DeleteSpeciesButton'
import { JobStatusBar } from './JobStatusBar'

// The generated API client is mocked: unit tests never call the real backend.
vi.mock('@/api/client', async (orig) => ({
  ...(await orig<typeof import('@/api/client')>()),
  api: { GET: vi.fn(), POST: vi.fn(), PUT: vi.fn(), DELETE: vi.fn() },
}))

const wrap = (ui: ReactNode) =>
  render(<QueryClientProvider client={new QueryClient()}>{ui}</QueryClientProvider>)

const job = (over: Partial<JobOut>): JobOut =>
  ({
    job_id: '00000000-0000-0000-0000-000000000001',
    kind: 'pipeline',
    status: 'running',
    stage: 'raw_occurrences',
    progress: 0.4,
    message: null,
    created_ts: new Date().toISOString(),
    updated_ts: new Date().toISOString(),
    ...over,
  }) as JobOut

describe('delete species confirmation', () => {
  it('requires the exact species name, ignoring case and spacing', () => {
    expect(confirmsName('impatiens  glandulifera ', 'Impatiens glandulifera')).toBe(true)
    expect(confirmsName('Impatiens', 'Impatiens glandulifera')).toBe(false)
    expect(confirmsName('', 'Impatiens glandulifera')).toBe(false)
  })
})

describe('discarded runs', () => {
  it('shows a discarded job as discarded, not failed', () => {
    wrap(<JobStatusBar job={job({ status: 'cancelled', message: 'Discarded by user' })} />)
    expect(screen.getByText(/Pipeline run · discarded/)).toBeInTheDocument()
    expect(screen.getByText('Discarded by user')).not.toHaveClass('text-destructive')
  })

  it('offers no discard button without run permission', () => {
    wrap(<JobStatusBar job={job({})} />)
    expect(screen.queryByRole('button', { name: /Discard run/ })).not.toBeInTheDocument()
  })
})
