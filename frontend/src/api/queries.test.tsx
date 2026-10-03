import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { act, renderHook } from '@testing-library/react'
import type { ReactNode } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { jobEventsUrl, jobJustFinished, jobPollInterval, qk, useJobStream } from './queries'
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

describe('job polling vs live stream', () => {
  it('stops polling while the SSE stream is connected', () => {
    expect(jobPollInterval(true, [job('a', 'running')])).toBe(false)
    expect(jobPollInterval(false, [job('a', 'running')])).toBe(2_000)
    expect(jobPollInterval(false, [job('a', 'succeeded')])).toBe(30_000)
    expect(jobPollInterval(false, undefined)).toBe(30_000)
  })
})

/** Minimal EventSource stand-in: the test emits named events by hand. */
class FakeEventSource {
  static instances: FakeEventSource[] = []
  closed = false
  private listeners = new Map<string, Array<(e: MessageEvent<string>) => void>>()
  constructor(readonly url: string) {
    FakeEventSource.instances.push(this)
  }
  addEventListener(type: string, fn: (e: MessageEvent<string>) => void) {
    this.listeners.set(type, [...(this.listeners.get(type) ?? []), fn])
  }
  emit(type: string, data = '') {
    for (const fn of this.listeners.get(type) ?? []) fn(new MessageEvent(type, { data }))
  }
  close() {
    this.closed = true
  }
}

describe('useJobStream', () => {
  beforeEach(() => {
    FakeEventSource.instances = []
    vi.stubGlobal('EventSource', FakeEventSource)
  })
  afterEach(() => vi.unstubAllGlobals())

  it('writes streamed job lists into the jobs cache and reports the connection', () => {
    const qc = new QueryClient()
    const wrapper = ({ children }: { children: ReactNode }) => (
      <QueryClientProvider client={qc}>{children}</QueryClientProvider>
    )
    const { result, unmount } = renderHook(() => useJobStream(42), { wrapper })
    const source = FakeEventSource.instances[0]!
    expect(source.url).toBe(jobEventsUrl(42))
    expect(source.url).toContain('/jobs/events?taxon_key=42')
    expect(result.current).toBe(false)

    const jobs = [job('a', 'running')]
    act(() => source.emit('jobs', JSON.stringify(jobs)))
    expect(result.current).toBe(true)
    expect(qc.getQueryData(qk.jobs(42))).toEqual(jobs)

    act(() => source.emit('error'))
    expect(result.current).toBe(false) // polling takes over until EventSource reconnects

    unmount()
    expect(source.closed).toBe(true)
  })
})
