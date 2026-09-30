import { Info, ShieldAlert } from 'lucide-react'

import type { SeverityResult } from '@/api/types'
import { SEVERITY_COMPONENTS, SeverityGauge } from '@/components/charts/SeverityGauge'
import { Badge } from '@/components/ui/badge'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Tooltip, TooltipContent, TooltipTrigger } from '@/components/ui/tooltip'

const CONFIDENCE_VARIANT = { high: 'success', moderate: 'warning', low: 'destructive' } as const

/** Severity with its full decomposition: components S, weights w and contributions w·S. */
export function SeverityPanel({ severity }: { severity: SeverityResult }) {
  const conf = severity.confidence
  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between">
          <CardTitle>Invasion severity index</CardTitle>
          <Tooltip>
            <TooltipTrigger asChild>
              <Badge variant={CONFIDENCE_VARIANT[conf.level]} className="cursor-help">
                <ShieldAlert /> {conf.level} confidence
              </Badge>
            </TooltipTrigger>
            <TooltipContent>
              {(conf.mess_ok_fraction * 100).toFixed(0)}% of candidate-zone area lies inside the
              training climate space (MESS ≥ 0); model {conf.model_type}. {conf.note}
            </TooltipContent>
          </Tooltip>
        </div>
        <CardDescription className="font-mono">{severity.formula}</CardDescription>
      </CardHeader>
      <CardContent className="grid gap-5 md:grid-cols-[auto_1fr] md:items-center">
        <div className="flex justify-center">
          <SeverityGauge severity={severity} />
        </div>
        <table className="w-full text-sm">
          <thead>
            <tr className="text-xs text-muted-foreground">
              <th className="pb-2 text-left font-medium">Component</th>
              <th className="pb-2 text-right font-medium">S</th>
              <th className="pb-2 text-right font-medium">w</th>
              <th className="pb-2 text-right font-medium">w·S</th>
            </tr>
          </thead>
          <tbody>
            {SEVERITY_COMPONENTS.map((c) => (
              <tr key={c.key} className="border-t">
                <td className="py-2">
                  <span className="flex items-center gap-2">
                    <span className="size-2.5 rounded-sm" style={{ background: c.color }} />
                    {c.label}
                    <Tooltip>
                      <TooltipTrigger aria-label={`About ${c.label}`}>
                        <Info className="size-3.5 text-muted-foreground" />
                      </TooltipTrigger>
                      <TooltipContent>{severity.descriptions[c.key]}</TooltipContent>
                    </Tooltip>
                  </span>
                </td>
                <td className="py-2 text-right tabular-nums">{severity.components[c.key].toFixed(2)}</td>
                <td className="py-2 text-right tabular-nums text-muted-foreground">
                  {severity.weights[c.key].toFixed(2)}
                </td>
                <td className="py-2 text-right font-medium tabular-nums">
                  {severity.contributions[c.key].toFixed(3)}
                </td>
              </tr>
            ))}
            <tr className="border-t-2 border-foreground/20">
              <td className="py-2 font-semibold">Severity</td>
              <td />
              <td />
              <td className="py-2 text-right font-semibold tabular-nums">{severity.score.toFixed(3)}</td>
            </tr>
          </tbody>
        </table>
      </CardContent>
    </Card>
  )
}
