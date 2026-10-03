import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import {
  createMemoryHistory,
  createRootRoute,
  createRoute,
  createRouter,
  RouterProvider,
} from '@tanstack/react-router'
import { act, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { beforeAll, beforeEach, describe, expect, it, vi } from 'vitest'
import { z } from 'zod'

import { SpeciesSearch } from './SpeciesSearch'

// The generated API client is mocked: unit tests never call the real backend.
const GET = vi.fn()
vi.mock('@/api/client', async (orig) => ({
  ...(await orig<typeof import('@/api/client')>()),
  api: { GET: (...args: unknown[]) => GET(...args) },
}))

beforeAll(() => {
  // jsdom lacks these; Radix popover / cmdk call them.
  globalThis.ResizeObserver ??= class {
    observe() {}
    unobserve() {}
    disconnect() {}
  }
  Element.prototype.scrollIntoView ??= () => {}
})

beforeEach(() => {
  GET.mockReset()
  GET.mockResolvedValue({ data: [], response: new Response(null, { status: 200 }) })
})

/** A router shaped like the app's: `q` is a root search param shared by every route. */
function renderAt(url: string) {
  const root = createRootRoute({
    validateSearch: (s) => z.object({ q: z.string().optional() }).parse(s),
    component: () => <SpeciesSearch />,
  })
  const index = createRoute({ getParentRoute: () => root, path: '/', component: () => null })
  const router = createRouter({
    routeTree: root.addChildren([index]),
    history: createMemoryHistory({ initialEntries: [url] }),
  })
  render(
    <QueryClientProvider client={new QueryClient()}>
      <RouterProvider router={router} />
    </QueryClientProvider>,
  )
  return router
}

const input = () => screen.findByPlaceholderText(/Search a species/)

describe('species search text in the URL', () => {
  it('restores the search box from ?q= and runs the search', async () => {
    renderAt('/?q=Lantana')
    expect(await input()).toHaveValue('Lantana')
    await waitFor(() =>
      expect(GET).toHaveBeenCalledWith('/species/search', {
        params: { query: { q: 'Lantana', limit: 10 } },
      }),
    )
  })

  it('writes typed text to ?q= once typing settles, without pushing history entries', async () => {
    const router = renderAt('/')
    fireEvent.change(await input(), { target: { value: 'Vespa velutina' } })
    await waitFor(() => expect(router.state.location.search).toEqual({ q: 'Vespa velutina' }))
    expect(router.history.length).toBe(1)

    fireEvent.change(await input(), { target: { value: '' } })
    await waitFor(() => expect(router.state.location.search).toEqual({}))
  })

  it('follows URL changes that did not come from typing (back/forward, links)', async () => {
    const router = renderAt('/?q=Acacia')
    expect(await input()).toHaveValue('Acacia')
    act(() => void router.navigate({ to: '/', search: { q: 'Pueraria' } }))
    await waitFor(() =>
      expect(screen.getByPlaceholderText(/Search a species/)).toHaveValue('Pueraria'),
    )
  })
})
