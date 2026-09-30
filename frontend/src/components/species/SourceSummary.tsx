import { Database } from 'lucide-react'

import type { SpeciesDetail } from '@/api/types'
import { Tooltip, TooltipContent, TooltipTrigger } from '@/components/ui/tooltip'
import { formatCompact, formatRelative } from '@/lib/utils'

/**
 * How the merged occurrence set was assembled: GBIF records (some of which are iNaturalist
 * observations republished to GBIF) plus research-grade iNaturalist observations not yet in
 * GBIF. Records are de-duplicated across the two sources by iNaturalist observation id.
 */
export function SourceSummary({ sp }: { sp: SpeciesDetail }) {
  const s = sp.sources
  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <span className="inline-flex cursor-help items-center gap-1 text-xs">
          <Database className="size-3" />
          {formatCompact(sp.n_occurrences_total)} records ={' '}
          <span className="text-foreground">{formatCompact(s.gbif)} GBIF</span> +{' '}
          <span className="text-foreground">{formatCompact(s.inaturalist_direct)} iNaturalist</span>
          <span className="text-muted-foreground/70">
            {' '}· GBIF checked {formatRelative(sp.last_gbif_fetch_ts)} · iNaturalist checked{' '}
            {formatRelative(s.last_inat_fetch_ts)}
          </span>
        </span>
      </TooltipTrigger>
      <TooltipContent className="max-w-sm text-xs leading-relaxed">
        <p>
          <b>{s.gbif.toLocaleString()}</b> records from GBIF, of which{' '}
          <b>{s.gbif_from_inaturalist.toLocaleString()}</b> are iNaturalist research-grade observations
          republished to GBIF.
        </p>
        <p className="mt-1">
          <b>{s.inaturalist_direct.toLocaleString()}</b> further research-grade observations fetched
          directly from iNaturalist (not yet in GBIF). Observations present in both are kept once,
          matched by iNaturalist observation id.
        </p>
      </TooltipContent>
    </Tooltip>
  )
}
