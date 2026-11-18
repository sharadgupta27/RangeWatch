import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'

import type { OccurrenceRow } from '@/api/types'
import { LOCATE_DEFAULT_ZOOM, locateZoom } from '@/components/maps/layers'

import { locateView, osmUrl, recordUrl } from './OccurrenceLocator'

// The generated API client is mocked: unit tests never call the real backend.
const GET = vi.fn()
vi.mock('@/api/client', async (orig) => ({
  ...(await orig<typeof import('@/api/client')>()),
  api: { GET: (...args: unknown[]) => GET(...args), POST: vi.fn() },
}))
// jsdom has no WebGL: stand in for the map and expose the camera it was given.
vi.mock('@/components/maps/DeckMap', () => ({
  DeckMap: ({ viewState }: { viewState: { longitude: number; latitude: number; zoom: number } }) => (
    <div data-testid="map">{`${viewState.latitude},${viewState.longitude}@${viewState.zoom.toFixed(1)}`}</div>
  ),
}))

const row = (over: Partial<OccurrenceRow>): OccurrenceRow =>
  ({
    source: 'iNaturalist',
    record_id: 27603760,
    latitude: 30.242,
    longitude: -87.789,
    coordinate_uncertainty_m: 7,
    event_date: '2019-06-24',
    basis_of_record: 'HUMAN_OBSERVATION',
    country_code: null,
    range_label: 'unknown',
    ingested_ts: '2026-09-29T00:38:00Z',
    ...over,
  }) as OccurrenceRow

describe('locating a record', () => {
  it('frames the coordinate uncertainty: tighter records zoom in further', () => {
    expect(locateZoom(30, null)).toBe(LOCATE_DEFAULT_ZOOM)
    expect(locateZoom(30, 0)).toBe(LOCATE_DEFAULT_ZOOM)
    expect(locateZoom(30, 7)).toBe(14) // clamped at street level
    expect(locateZoom(30, 50_000)).toBeLessThan(locateZoom(30, 500))
    expect(locateZoom(30, 5_000_000)).toBe(4) // clamped: still shows the region
  })

  it('links to the original record and to OpenStreetMap', () => {
    expect(recordUrl({ source: 'GBIF', record_id: 1 })).toBe('https://www.gbif.org/occurrence/1')
    expect(recordUrl({ source: 'iNaturalist', record_id: 2 })).toBe('https://www.inaturalist.org/observations/2')
    expect(osmUrl(34.248, -6.678, 11)).toBe(
      'https://www.openstreetmap.org/?mlat=34.24800&mlon=-6.67800#map=12/34.24800/-6.67800',
    )
    expect(locateView(row({})).longitude).toBe(-87.789)
  })

  it('opens the map on a row, steps through records and closes', async () => {
    const page = {
      data: {
        items: [row({}), row({ record_id: 335345549, latitude: 34.248, longitude: -6.678, coordinate_uncertainty_m: null })],
        total: 2,
        offset: 0,
        limit: 25,
      },
      response: new Response(null, { status: 200 }),
    }
    const me = {
      data: { user: 'curator', role: 'curator', permissions: { run: true, curate: true } },
      response: new Response(null, { status: 200 }),
    }
    GET.mockImplementation(async (path: string) => (path === '/me' ? me : page))
    const { OccurrenceTable } = await import('./OccurrenceTable')
    render(
      <QueryClientProvider client={new QueryClient()}>
        <OccurrenceTable taxonKey={7} />
      </QueryClientProvider>,
    )
    await waitFor(() => expect(screen.getByText('#27603760')).toBeInTheDocument())
    expect(screen.queryByTestId('map')).not.toBeInTheDocument()

    fireEvent.click(screen.getByRole('button', { name: 'Locate record 27603760 on the map' }))
    expect(await screen.findByTestId('map')).toHaveTextContent('30.242,-87.789@14.0')
    expect(screen.getByText('1 / 2')).toBeInTheDocument()
    // The annotation control sits next to the map.
    expect(await screen.findAllByRole('combobox', { name: 'Annotate record 27603760' })).toHaveLength(2)
    expect(screen.getByRole('link', { name: /Open record on iNaturalist/ })).toHaveAttribute(
      'href',
      'https://www.inaturalist.org/observations/27603760',
    )

    fireEvent.click(screen.getByRole('button', { name: 'Next record' }))
    expect(screen.getByTestId('map')).toHaveTextContent(`34.248,-6.678@${LOCATE_DEFAULT_ZOOM.toFixed(1)}`)
    expect(screen.getByRole('button', { name: 'Next record' })).toBeDisabled()

    fireEvent.click(screen.getByRole('button', { name: 'Close map' }))
    expect(screen.queryByTestId('map')).not.toBeInTheDocument()
  })
})
