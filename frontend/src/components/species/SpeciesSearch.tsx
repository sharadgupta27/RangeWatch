import { useNavigate, useSearch } from '@tanstack/react-router'
import { ChevronRight, Leaf, Loader2, Search } from 'lucide-react'
import { useEffect, useRef, useState } from 'react'

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
 * Keeps the search text in the URL (`?q=`, a root search param) so a search survives reloads,
 * back/forward and shared links. The input stays local state for responsiveness; the URL is
 * written once typing settles, and URL changes (back button) flow back into the input.
 */
function useUrlSearchText(debounced: string, setText: (v: string) => void) {
  const urlQ = useSearch({ strict: false, select: (s) => s.q ?? '' })
  const navigate = useNavigate()
  const written = useRef(urlQ)

  // Input → URL, after the debounce. Replace (not push) so history isn't one entry per word.
  useEffect(() => {
    const next = debounced.trim()
    if (next === written.current) return
    written.current = next
    void navigate({
      to: '.',
      search: (prev: Record<string, unknown>) => ({ ...prev, q: next || undefined }),
      replace: true,
    })
  }, [debounced]) // only typing writes the URL

  // URL → input, for changes that did not come from typing (back/forward, links).
  useEffect(() => {
    if (urlQ === written.current) return
    written.current = urlQ
    setText(urlQ)
  }, [urlQ]) // only URL changes overwrite the input
}

/**
 * Search-as-you-type over the GBIF backbone (via the backend), enriched with iNaturalist
 * thumbnails and common names. Selecting a result routes to /species/:taxonKey.
 */
export function SpeciesSearch({ className, autoFocus }: { className?: string; autoFocus?: boolean }) {
  const initialQ = useSearch({ strict: false, select: (s) => s.q ?? '' })
  const [q, setQ] = useState(initialQ)
  // A search restored from the URL shows its results straight away.
  const [open, setOpen] = useState(initialQ.trim().length >= 2)
  const debounced = useDebounced(q, 250)
  useUrlSearchText(debounced, setQ)
  const search = useSpeciesSearch(debounced)
  const navigate = useNavigate()
  const boxRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'k' && (e.metaKey || e.ctrlKey)) {
        e.preventDefault()
        boxRef.current?.querySelector('input')?.focus()
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
          <div
            ref={boxRef}
            className="relative flex h-10 items-center gap-2 rounded-lg border bg-card/80 px-3 shadow-sm focus-within:ring-[3px] focus-within:ring-ring/40">
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
        placeholder="Search a species — e.g. Lantana camara, Vespa velutina…"
        className="h-9 py-0"
        {...props}
      />
    </div>
  )
}
