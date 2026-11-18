import { z } from 'zod'

import type { PredictorInfo } from '@/api/types'

/** Climate data + predictor choice for one species (the server re-validates everything). */
export const environmentSchema = z
  .object({
    /** '' = the deployment default climate version. */
    climateVersion: z.string(),
    mode: z.enum(['auto', 'custom']),
    predictors: z.array(z.string()),
    vifScreen: z.boolean(),
  })
  .refine((v) => v.mode === 'auto' || v.predictors.length >= 2, {
    path: ['predictors'],
    message: 'Choose at least two predictors',
  })

export type EnvironmentForm = z.input<typeof environmentSchema>

/** Predictors grouped by process, in the order the climate version lists them. */
export function groupPredictors(predictors: PredictorInfo[]): [string, PredictorInfo[]][] {
  const groups = new Map<string, PredictorInfo[]>()
  for (const p of predictors) {
    const list = groups.get(p.group_label) ?? []
    list.push(p)
    groups.set(p.group_label, list)
  }
  return [...groups.entries()]
}

/** Chosen predictors without future data (the model then cannot be projected onto futures). */
export function presentOnly(chosen: string[], predictors: PredictorInfo[]): string[] {
  const offered = new Map(predictors.map((p) => [p.name, p]))
  return chosen.filter((n) => offered.get(n)?.future_available === false)
}
