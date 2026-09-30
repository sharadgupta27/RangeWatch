import { AlertCircle, CheckCircle2, Clock, Loader2, PenLine, RefreshCw } from 'lucide-react'

import type { SpeciesStatus } from '@/api/types'
import { Badge, type BadgeProps } from '@/components/ui/badge'
import { formatDate } from '@/lib/utils'

interface StatusBadgeProps {
  status: SpeciesStatus
  pendingRecords?: number
  trainedTs?: string | null
  jobActive?: boolean
}

/** Pure presentation of backend state — the retrain decision itself is server-side only. */
export function describeStatus({
  status,
  pendingRecords = 0,
  trainedTs,
  jobActive,
}: StatusBadgeProps): { label: string; variant: BadgeProps['variant']; icon: typeof CheckCircle2 } {
  if (jobActive && status !== 'training' && status !== 'ingesting') {
    return { label: 'Checking for updates…', variant: 'default', icon: Loader2 }
  }
  switch (status) {
    case 'up_to_date':
      return {
        label: trainedTs ? `Up to date · retrained ${formatDate(trainedTs)}` : 'Up to date',
        variant: 'success',
        icon: CheckCircle2,
      }
    case 'minor_update_pending':
      return {
        label: `${pendingRecords.toLocaleString()} new record${pendingRecords === 1 ? '' : 's'} pending`,
        variant: 'warning',
        icon: Clock,
      }
    case 'retrain_pending':
      return { label: 'Retrain pending', variant: 'warning', icon: RefreshCw }
    case 'awaiting_native_range_review':
      return { label: 'Native range review needed', variant: 'warning', icon: PenLine }
    case 'ingesting':
      return { label: 'Ingesting occurrences…', variant: 'default', icon: Loader2 }
    case 'training':
      return { label: 'Training model…', variant: 'default', icon: Loader2 }
    case 'failed':
      return { label: 'Last run failed', variant: 'destructive', icon: AlertCircle }
    default:
      return { label: 'Registered', variant: 'secondary', icon: Clock }
  }
}

export function StatusBadge(props: StatusBadgeProps) {
  const { label, variant, icon: Icon } = describeStatus(props)
  const spinning = Icon === Loader2
  return (
    <Badge variant={variant}>
      <Icon className={spinning ? 'animate-spin' : undefined} />
      {label}
    </Badge>
  )
}
