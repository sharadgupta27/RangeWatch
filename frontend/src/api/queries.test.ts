import { describe, expect, it } from 'vitest'

import { jobJustFinished } from './queries'
import type { JobOut } from './types'

const job = (id: string, status: JobOut['status']): JobOut => ({
  job_id: id,
  taxon_key: 1,
  kind: 'pipeline',
  status,
  stage: null,
  progress: status === 'succeeded' ? 1 : 0.5,
  message: null,
  result: null,
  created_ts: '2026-09-30T10:00:00Z',
  updated_ts: '2026-09-30T10:00:00Z',
})

describe('jobJustFinished', () => {
  it('fires when the job seen running completes', () => {
    expect(jobJustFinished(job('a', 'running'), job('a', 'succeeded'))).toBe(true)
    expect(jobJustFinished(job('a', 'queued'), job('a', 'failed'))).toBe(true)
  })

  it('fires for a new job that started and finished between two polls', () => {
    expect(jobJustFinished(job('a', 'succeeded'), job('b', 'succeeded'))).toBe(true)
  })

  it('stays quiet while running, on first load and without changes', () => {
    expect(jobJustFinished(job('a', 'succeeded'), job('b', 'running'))).toBe(false)
    expect(jobJustFinished(undefined, job('a', 'succeeded'))).toBe(false)
    expect(jobJustFinished(job('a', 'succeeded'), job('a', 'succeeded'))).toBe(false)
  })
})
