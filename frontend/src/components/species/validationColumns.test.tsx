import {
  createMemoryHistory,
  createRootRoute,
  createRoute,
  createRouter,
  RouterProvider,
} from '@tanstack/react-router'
import { useTable } from '@tanstack/react-table'
import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'

import type { ValidationSpecies } from '@/api/types'

import { validationColumns, validationFeatures } from './validationColumns'

const species: ValidationSpecies[] = [
  {
    taxon_key: 1,
    scientific_name: 'Linepithema humile',
    native_range_reference: 'Paraná basin',
    status: 'evaluated',
    passed: true,
    model_version: 4,
    model_type: 'B',
    bioclim_version: 'bioclim_v2',
    auc_mean: 0.84,
    cbi_mean: 0.87,
    n_occurrences: 21946,
    regions: [],
    invaded_captured: 4,
    invaded_total: 4,
    controls_correct: 2,
    controls_total: 2,
    transferability: { auc: 0.74, n_introduced_test: 900, literature_mean_auc: 0.7 },
  },
  {
    taxon_key: 2,
    scientific_name: 'Vespa velutina',
    native_range_reference: 'SE Asia',
    status: 'error',
    passed: false,
    error: 'HTTPError',
    regions: [],
  },
]

function Grid() {
  const table = useTable({ features: validationFeatures, columns: validationColumns, data: species })
  return (
    <table>
      <tbody>
        {table.getRowModel().rows.map((r) => (
          <tr key={r.id}>
            {r.getAllCells().map((c) => (
              <td key={c.id}>
                <table.FlexRender cell={c} />
              </td>
            ))}
          </tr>
        ))}
      </tbody>
    </table>
  )
}

describe('validation table columns', () => {
  it('shows pass/error status, region capture and the independent transfer AUC', async () => {
    const root = createRootRoute({ component: Grid })
    const species$ = createRoute({ getParentRoute: () => root, path: '/species/$taxonKey' })
    const router = createRouter({
      routeTree: root.addChildren([species$]),
      history: createMemoryHistory({ initialEntries: ['/'] }),
    })
    render(<RouterProvider router={router} />)
    expect(await screen.findByText('pass')).toBeInTheDocument()
    expect(screen.getByText('error')).toBeInTheDocument()
    expect(screen.getByText('4/4')).toBeInTheDocument()
    expect(screen.getByText('0.740')).toHaveClass('text-success') // ≥ literature 0.7
    expect(screen.getByText('v4 · B · bioclim_v2')).toBeInTheDocument()
  })
})
