import { z } from 'zod'

const weight = z.number().min(0, 'Weights must be ≥ 0').max(10)

/** Client-side validation mirroring the backend's SeverityConfig rules. */
export const severityConfigSchema = z
  .object({
    weights: z.object({
      suitability: weight,
      climate_analogy: weight,
      spread_rate: weight,
      ecological_impact_prior: weight,
    }),
    impact_prior: z.number().min(0).max(1),
    impact_prior_source: z.string().trim().min(1, 'Cite where the impact prior comes from').max(200),
    suitability_saturation: z.number().gt(0).max(1),
    spread_half_saturation: z.number().gt(0).max(1000),
    spread_window_years: z.number().int().min(2).max(50),
  })
  .refine((v) => Object.values(v.weights).some((w) => w > 0), {
    message: 'At least one weight must be positive',
    path: ['weights'],
  })

export type SeverityConfigForm = z.infer<typeof severityConfigSchema>

export function normalizedWeights(w: SeverityConfigForm['weights']) {
  const total = Object.values(w).reduce((a, b) => a + b, 0) || 1
  return Object.fromEntries(Object.entries(w).map(([k, v]) => [k, v / total])) as SeverityConfigForm['weights']
}
