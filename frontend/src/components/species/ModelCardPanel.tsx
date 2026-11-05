import { AlertTriangle, Download, FileText } from 'lucide-react'

import { absoluteApiUrl } from '@/api/client'
import { useModelCard } from '@/api/queries'
import type { ModelCardOut } from '@/api/types'
import { Badge } from '@/components/ui/badge'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Skeleton } from '@/components/ui/skeleton'

/** The card itself: confidence label and caveats first, then plain-language sections. */
export function ModelCardView({ card }: { card: ModelCardOut }) {
  return (
    <div className="space-y-5 text-sm">
      <div className="space-y-2">
        <div className="flex flex-wrap items-center gap-2">
          <Badge variant={card.confidence_label === 'standard' ? 'success' : 'warning'}>
            {card.confidence_label} confidence
          </Badge>
          <Badge variant="outline">Model {card.model_type}</Badge>
        </div>
        <p className="font-medium">{card.headline}</p>
      </div>
      <div role="note" className="space-y-1.5 rounded-lg border border-warning/40 bg-warning/5 p-3 text-xs">
        <div className="flex items-center gap-1.5 font-medium">
          <AlertTriangle className="size-3.5 text-warning" /> Before you use this map
        </div>
        <ul className="list-disc space-y-1 pl-4">
          {card.caveats.map((c) => (
            <li key={c}>{c}</li>
          ))}
        </ul>
      </div>
      {card.sections.map((s) => (
        <section key={s.id} className="space-y-1.5">
          <h3 className="font-medium">{s.title}</h3>
          {s.paragraphs.map((p) => (
            <p key={p} className="text-xs leading-relaxed text-muted-foreground">
              {p}
            </p>
          ))}
          {!!s.facts?.length && (
            <dl className="grid grid-cols-[auto_1fr] gap-x-4 gap-y-0.5 text-xs">
              {s.facts.map((f) => (
                <div key={f.label} className="contents">
                  <dt className="text-muted-foreground">{f.label}</dt>
                  <dd className="break-all font-mono">{f.value}</dd>
                </div>
              ))}
            </dl>
          )}
        </section>
      ))}
      <section className="space-y-1">
        <h3 className="font-medium">How to cite</h3>
        <p className="break-words text-xs text-muted-foreground">{card.citation}</p>
      </section>
      <p className="text-[11px] text-muted-foreground">{card.source}</p>
    </div>
  )
}

/** U13: plain-language model card for non-modellers. */
export function ModelCardPanel({ taxonKey, version }: { taxonKey: number; version: number }) {
  const card = useModelCard(taxonKey, version)
  const md = absoluteApiUrl(
    `/species/${taxonKey}/models/${version}/exports/model-card?format=md`,
  )
  return (
    <Card>
      <CardHeader className="flex flex-row flex-wrap items-start justify-between gap-3">
        <div className="space-y-1.5">
          <CardTitle className="flex items-center gap-2">
            <FileText className="size-4" /> Model card · v{version}
          </CardTitle>
          <CardDescription className="max-w-3xl">
            What this model can and cannot tell you, in plain language. Generated from the
            metrics and lineage stored with the version.
          </CardDescription>
        </div>
        <a href={md} className="inline-flex items-center gap-1 text-xs text-primary hover:underline">
          <Download className="size-3" /> Markdown
        </a>
      </CardHeader>
      <CardContent>
        {card.isLoading ? (
          <Skeleton className="h-64" />
        ) : card.isError ? (
          <p className="text-sm text-destructive">{card.error.message}</p>
        ) : card.data ? (
          <ModelCardView card={card.data} />
        ) : null}
      </CardContent>
    </Card>
  )
}
