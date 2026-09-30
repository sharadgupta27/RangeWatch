import { AlertTriangle, Info } from 'lucide-react'

import type { ModelSummary } from '@/api/types'
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert'

/**
 * Mandatory transferability caveat (CLAUDE.md constraint 1). Model A (native-only) gets the
 * stronger, warning-level treatment; Model B still carries the caveat.
 */
export function CaveatBanner({ model }: { model: ModelSummary }) {
  if (model.model_type === 'A') {
    return (
      <Alert variant="warning">
        <AlertTriangle />
        <AlertTitle>Lower-confidence projection — native-range-only model (Model A)</AlertTitle>
        <AlertDescription>
          No documented introduced-range records were available, so the invasion projection is an
          extrapolation from the native range. {model.transferability_caveat}
        </AlertDescription>
      </Alert>
    )
  }
  return (
    <Alert variant="info">
      <Info />
      <AlertTitle>Combined native + invaded-range model (Model B)</AlertTitle>
      <AlertDescription>{model.transferability_caveat}</AlertDescription>
    </Alert>
  )
}
