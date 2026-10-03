import type { QueryClient } from '@tanstack/react-query'
import { createRootRouteWithContext, Link, Outlet, useRouterState } from '@tanstack/react-router'
import { Moon, RefreshCw, Sun } from 'lucide-react'
import { z } from 'zod'

import { useAppUpdateAvailable } from '@/api/appVersion'
import { useBioclim } from '@/api/queries'
import { SpeciesSearch } from '@/components/species/SpeciesSearch'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { TooltipProvider } from '@/components/ui/tooltip'
import { useTheme } from '@/lib/theme'

/** `?q=` — the species search text, kept in the URL so searches survive reloads and links. */
const rootSearchSchema = z.object({ q: z.string().max(100).optional().catch(undefined) })

export const Route = createRootRouteWithContext<{ queryClient: QueryClient }>()({
  validateSearch: (search) => rootSearchSchema.parse(search),
  component: RootLayout,
  notFoundComponent: () => (
    <div className="mx-auto max-w-md py-24 text-center">
      <h1 className="text-lg font-semibold">Page not found</h1>
      <Link to="/" className="mt-2 inline-block text-sm text-primary hover:underline">
        Back to the registry
      </Link>
    </div>
  ),
})

function Logo() {
  return (
    <svg viewBox="0 0 32 32" className="size-7" aria-hidden>
      <rect width="32" height="32" rx="8" className="fill-primary/15" />
      <circle cx="16" cy="16" r="9" fill="none" className="stroke-primary" strokeWidth="2.5" />
      <circle cx="16" cy="16" r="3.5" fill="var(--introduced)" />
    </svg>
  )
}

function RootLayout() {
  const [theme, setTheme] = useTheme()
  const bioclim = useBioclim()
  const updateAvailable = useAppUpdateAvailable()
  // The registry page has its own prominent search in the hero; avoid a second box there.
  const onRegistry = useRouterState({ select: (s) => s.location.pathname === '/' })
  return (
    <TooltipProvider delayDuration={200}>
      <div className="min-h-full">
        {updateAvailable && (
          <div className="flex items-center justify-center gap-3 bg-primary px-4 py-1.5 text-xs text-primary-foreground">
            A newer version of the dashboard has been deployed.
            <Button size="sm" variant="secondary" className="h-6" onClick={() => window.location.reload()}>
              <RefreshCw /> Reload
            </Button>
          </div>
        )}
        <header className="sticky top-0 z-40 border-b bg-background/80 backdrop-blur-xl">
          <div className="mx-auto flex h-14 max-w-[1600px] items-center gap-4 px-4 lg:px-6">
            <Link to="/" className="flex items-center gap-2.5">
              <Logo />
              <span className="hidden leading-tight sm:block">
                <span className="block text-sm font-semibold tracking-tight">RangeWatch</span>
                <span className="block text-[11px] text-muted-foreground">
                  Species distribution MLOps
                </span>
              </span>
            </Link>
            <nav className="hidden items-center gap-1 text-sm md:flex">
              <Link
                to="/"
                activeOptions={{ exact: true }}
                className="rounded-md px-2.5 py-1.5 text-muted-foreground hover:text-foreground data-[status=active]:text-foreground"
              >
                Registry
              </Link>
              <Link
                to="/validation"
                className="rounded-md px-2.5 py-1.5 text-muted-foreground hover:text-foreground data-[status=active]:text-foreground"
              >
                Validation
              </Link>
            </nav>
            {onRegistry ? (
              <div className="flex-1" />
            ) : (
              <SpeciesSearch className="mx-auto w-full max-w-xl" />
            )}
            <div className="flex items-center gap-2">
              {bioclim.data && (
                <Badge variant="outline" className="hidden font-mono md:inline-flex">
                  {bioclim.data.version} · {bioclim.data.resolution}
                </Badge>
              )}
              <Button
                variant="ghost"
                size="icon"
                aria-label="Toggle theme"
                onClick={() => setTheme(theme === 'dark' ? 'light' : 'dark')}
              >
                {theme === 'dark' ? <Sun /> : <Moon />}
              </Button>
            </div>
          </div>
        </header>
        <main className="mx-auto max-w-[1600px] px-4 py-6 lg:px-6">
          <Outlet />
        </main>
      </div>
    </TooltipProvider>
  )
}
