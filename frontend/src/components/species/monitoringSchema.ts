import { z } from 'zod'

/** Alert subscription (the server re-validates: https, public host, e-mail format). */
export const subscriptionSchema = z
  .object({
    channel: z.enum(['webhook', 'email']),
    target: z.string().trim().min(3, 'Enter a URL or e-mail address').max(500),
    allSpecies: z.boolean(),
  })
  .superRefine((v, ctx) => {
    if (v.channel === 'webhook' && !/^https:\/\/[^\s/]+/i.test(v.target))
      ctx.addIssue({ code: 'custom', path: ['target'], message: 'Webhooks must use https://' })
    if (v.channel === 'email' && !/^[^@\s]+@[^@\s]+\.[^@\s]+$/.test(v.target))
      ctx.addIssue({ code: 'custom', path: ['target'], message: 'Not an e-mail address' })
  })

export type SubscriptionForm = z.input<typeof subscriptionSchema>
export const EMPTY_SUBSCRIPTION: SubscriptionForm = { channel: 'webhook', target: '', allSpecies: false }

const lon = z.coerce.number().min(-180, '≥ −180').max(180, '≤ 180')
const lat = z.coerce.number().min(-90, '≥ −90').max(90, '≤ 90')

/** Regional study area [west, south, east, north] (U2). */
export const studyRegionSchema = z
  .object({ west: lon, south: lat, east: lon, north: lat, resolution: z.string() })
  .refine((v) => v.west < v.east, { path: ['east'], message: 'East must exceed west' })
  .refine((v) => v.south < v.north, { path: ['north'], message: 'North must exceed south' })

export type StudyRegionForm = z.input<typeof studyRegionSchema>
