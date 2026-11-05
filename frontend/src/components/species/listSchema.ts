import { z } from 'zod'

import type { SpeciesListIn } from '@/api/types'

/** Most names / keys per list request (the server enforces SDM_LIST_MAX_NAMES too). */
export const MAX_LIST_ENTRIES = 200

/**
 * One species per line (or separated by ";" / ","): a GBIF taxon key (digits only) or a
 * scientific name. Blank lines and duplicates are dropped; names keep their spelling, the
 * server matches them against the GBIF backbone.
 */
export function parseSpeciesEntries(text: string): Pick<SpeciesListIn, 'names' | 'taxon_keys'> {
  const names: string[] = []
  const keys: number[] = []
  const seen = new Set<string>()
  for (const raw of text.split(/[\n;,]+/)) {
    const entry = raw.trim().replace(/\s+/g, ' ')
    if (!entry || seen.has(entry.toLowerCase())) continue
    seen.add(entry.toLowerCase())
    if (/^\d+$/.test(entry)) keys.push(Number(entry))
    else names.push(entry)
  }
  return { names, taxon_keys: keys }
}

export const speciesListSchema = z
  .object({
    name: z.string().trim().min(1, 'Name the list').max(200),
    source: z.string().trim().max(500),
    entries: z.string(),
  })
  .superRefine((v, ctx) => {
    const { names = [], taxon_keys = [] } = parseSpeciesEntries(v.entries)
    const n = names.length + taxon_keys.length
    if (n === 0) ctx.addIssue({ code: 'custom', path: ['entries'], message: 'Add at least one species' })
    if (n > MAX_LIST_ENTRIES)
      ctx.addIssue({ code: 'custom', path: ['entries'], message: `At most ${MAX_LIST_ENTRIES} species` })
  })

export type SpeciesListForm = z.input<typeof speciesListSchema>
export const EMPTY_SPECIES_LIST: SpeciesListForm = { name: '', source: '', entries: '' }

export function toSpeciesListIn(v: SpeciesListForm): SpeciesListIn {
  return { name: v.name.trim(), source: v.source.trim() || null, ...parseSpeciesEntries(v.entries) }
}
