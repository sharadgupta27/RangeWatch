import { FileDown, FileText, Loader2 } from 'lucide-react'

import { absoluteApiUrl } from '@/api/client'
import { useBulletinStatus, useGenerateBulletin } from '@/api/queries'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'

export function BulletinCard({ taxonKey, jobActive }: { taxonKey: number; jobActive: boolean }) {
  const status = useBulletinStatus(taxonKey)
  const generate = useGenerateBulletin(taxonKey)
  const s = status.data
  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <FileText className="size-4" /> Species bulletin
        </CardTitle>
        <CardDescription>
          Server-rendered PDF: maps with MESS overlay, severity breakdown, diagnostics, expansion
          outlook, transferability caveat and full reproducibility metadata.
        </CardDescription>
      </CardHeader>
      <CardContent className="flex flex-wrap items-center gap-2">
        <Button
          size="sm"
          onClick={() => generate.mutate()}
          disabled={!s?.model_version || generate.isPending || jobActive}
        >
          {generate.isPending || jobActive ? <Loader2 className="animate-spin" /> : <FileText />}
          Generate bulletin
        </Button>
        {s?.pdf_url && (
          <Button size="sm" variant="outline" asChild>
            <a href={absoluteApiUrl(s.pdf_url)} target="_blank" rel="noreferrer">
              <FileDown /> PDF (v{s.model_version})
            </a>
          </Button>
        )}
        {s?.html_url && (
          <Button size="sm" variant="ghost" asChild>
            <a href={absoluteApiUrl(s.html_url)} target="_blank" rel="noreferrer">
              HTML preview
            </a>
          </Button>
        )}
        {generate.isError && <span className="text-xs text-destructive">{generate.error.message}</span>}
      </CardContent>
    </Card>
  )
}
