import { useNavigate } from '@tanstack/react-router'
import { ChevronRight, Leaf, Loader2, Search } from 'lucide-react'
import { useEffect, useState } from 'react'

import { useSpeciesSearch } from '@/api/queries'
import { Badge } from '@/components/ui/badge'
import {
  Command,
  CommandEmpty,
  CommandGroup,
  CommandInput,
  CommandItem,
  CommandList,
} from '@/components/ui/command'
import { Popover, PopoverAnchor, PopoverContent } from '@/components/ui/popover'
import { cn } from '@/lib/utils'
import { useDebounced } from '@/lib/useDebounced'

/**
 * Search-as-you-type over the GBIF backbone (via the backend), enriched with iNaturalist
 * thumbnails and common names. Selecting a result routes to /species/:taxonKey.
 */
export function SpeciesSearch({ className, autoFocus }: { className?: string; autoFocus?: boolean }) {
  const [q, setQ] = useState('')
  const [open, setOpen] = useState(false)
  const debounced = useDebounced(q, 250)
  const search = useSpeciesSearch(debounced)
  const navigate = useNavigate()

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'k' && (e.metaKey || e.ctrlKey)) {
        e.preventDefault()
        document.getElementById('species-search-input')?.focus()
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [])

  const results = search.data ?? []

  return (
    <Command shouldFilter={false} className={cn('overflow-visible bg-transparent', className)}>
      <Popover open={open && debounced.trim().length >= 2} onOpenChange={setOpen}>
        <PopoverAnchor asChild>
          <div className="relative flex h-10 items-center gap-2 rounded-lg border bg-card/80 px-3 shadow-sm focus-within:ring-[3px] focus-within:ring-ring/40">
            <Search className="size-4 text-muted-foreground" />
            <CommandInputBare
              value={q}
              autoFocus={autoFocus}
              onValueChange={(v) => {
                setQ(v)
                setOpen(true)
              }}
              onFocus={() => setOpen(true)}
            />
            {search.isFetching ? (
              <Loader2 className="size-4 animate-spin text-muted-foreground" />
            ) : (
              <kbd className="hidden rounded border bg-muted px-1.5 font-mono text-[10px] text-muted-foreground sm:inline">
                Ctrl K
              </kbd>
            )}
          </div>
        </PopoverAnchor>
        <PopoverContent
          className="w-[var(--radix-popover-trigger-width)] min-w-[360px]"
          onOpenAutoFocus={(e) => e.preventDefault()}
        >
          <CommandList>
            {!search.isFetching && <CommandEmpty>No species found.</CommandEmpty>}
            {results.length > 0 && (
              <CommandGroup heading="GBIF backbone matches">
                {results.map((r) => (
                  <CommandItem
                    key={r.taxon_key}
                    value={String(r.taxon_key)}
                    onSelect={() => {
                      setOpen(false)
                      setQ('')
                      void navigate({
                        to: '/species/$taxonKey',
                        params: { taxonKey: r.taxon_key },
                      })
                    }}
                  >
                    {r.thumbnail_url ? (
                      <img src={r.thumbnail_url} alt="" className="size-10 rounded-md object-cover" />
                    ) : (
                      <span className="flex size-10 items-center justify-center rounded-md bg-muted">
                        <Leaf className="size-4 text-muted-foreground" />
                      </span>
                    )}
                    <div className="min-w-0 flex-1">
                      <div className="flex items-center gap-2">
                        <span className="truncate font-medium italic">
                          {r.canonical_name ?? r.scientific_name}
                        </span>
                        {r.in_registry && <Badge variant="default">tracked</Badge>}
                      </div>
                      {r.common_name && (
                        <div className="truncate text-xs text-foreground/80">{r.common_name}</div>
                      )}
                      <div className="flex items-center gap-0.5 truncate text-[11px] text-muted-foreground">
                        {(r.breadcrumb ?? []).map((b, i) => (
                          <span key={b} className="flex items-center gap-0.5">
                            {i > 0 && <ChevronRight className="size-3 opacity-50" />}
                            {b}
                          </span>
                        ))}
                      </div>
                    </div>
                    <span className="font-mono text-[10px] text-muted-foreground">{r.taxon_key}</span>
                  </CommandItem>
                ))}
              </CommandGroup>
            )}
          </CommandList>
        </PopoverContent>
      </Popover>
    </Command>
  )
}

/** cmdk input without the default wrapper (the anchor above provides chrome). */
function CommandInputBare(props: React.ComponentProps<typeof CommandInput>) {
  return (
    <div className="flex-1 [&>div]:border-0 [&>div]:px-0 [&_svg]:hidden">
      <CommandInput
        id="species-search-input"
        placeholder="Search a species — e.g. Lantana camara, Vespa velutina…"
        className="h-9 py-0"
        {...props}
      />
    </div>
  )
}
