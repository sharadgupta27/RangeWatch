import { Info } from 'lucide-react'
import type { ReactNode } from 'react'

import type { ModelVersionDetail } from '@/api/types'
import { Card } from '@/components/ui/card'
import { Tooltip, TooltipContent, TooltipTrigger } from '@/components/ui/tooltip'
import { formatCompact, formatNumber } from '@/lib/utils'

function Stat({
  label,
  value,
  sub,
  hint,
}: {
  label: string
  value: ReactNode
  sub?: ReactNode
  hint?: string
}) {
  return (
    <Card className="px-4 py-3">
      <div className="flex items-center gap-1.5 text-xs text-muted-foreground">
        {label}
        {hint && (
          <Tooltip>
            <TooltipTrigger aria-label={`About ${label}`}>
              <Info className="size-3" />
            </TooltipTrigger>
            <TooltipContent>{hint}</TooltipContent>
          </Tooltip>
        )}
      </div>
      <div className="mt-1 text-xl font-semibold tabular-nums tracking-tight">{value}</div>
      {sub && <div className="text-[11px] text-muted-foreground">{sub}</div>}
    </Card>
  )
}

const pm = (m?: number | null, s?: number | null) =>
  m == null ? '–' : (
    <>
      {m.toFixed(3)}
      {s != null && <span className="ml-1 text-sm font-normal text-muted-foreground">±{s.toFixed(2)}</span>}
    </>
  )

/** Spatial-CV diagnostics — CBI is always shown next to AUC/TSS (CLAUDE.md constraint 5). */
export function MetricsCards({ model }: { model: ModelVersionDetail }) {
  const p = model.projection
  return (
    <div className="grid grid-cols-2 gap-3 md:grid-cols-3 xl:grid-cols-6">
      <Stat
        label="CBI (spatial CV)"
        value={pm(model.cbi_mean, model.cbi_std)}
        hint="Continuous Boyce Index: presence-only-appropriate; > 0 means better than random."
      />
      <Stat
        label="AUC (spatial CV)"
        value={pm(model.auc_mean, model.auc_std)}
        hint="Background treated as pseudo-absence; spatial blocks avoid the inflation of random CV."
      />
      <Stat label="TSS (spatial CV)" value={pm(model.tss_mean, model.tss_std)} />
      <Stat
        label="Candidate invasion zone"
        value={`${formatCompact(p.candidate_area_km2)} km²`}
        sub={`${(p.candidate_fraction_of_nonnative_land * 100).toFixed(1)}% of non-native land`}
      />
      <Stat
        label="Within training climate"
        value={`${(p.candidate_mess_ok_fraction * 100).toFixed(0)}%`}
        sub="of candidate area has MESS ≥ 0"
        hint="Share of the candidate zone where no predictor is outside the training range."
      />
      <Stat
        label="Presences / background"
        value={formatNumber(model.n_presence)}
        sub={`${formatNumber(model.n_background)} background · ${model.n_cv_folds} folds`}
      />
    </div>
  )
}
