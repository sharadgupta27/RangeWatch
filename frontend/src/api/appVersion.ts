import { useQuery } from '@tanstack/react-query'

/**
 * Detects that a newer dashboard build has been deployed while this tab stayed open (the SPA
 * would otherwise keep running the old bundle indefinitely). Served by the frontend's own
 * Nginx, not the backend API.
 */
export function useAppUpdateAvailable(): boolean {
  const q = useQuery({
    queryKey: ['app-version'],
    queryFn: async () => {
      const res = await fetch(`/version.json?t=${Date.now()}`, { cache: 'no-store' })
      if (!res.ok) return null
      return (await res.json()) as { build: string }
    },
    enabled: !import.meta.env.DEV,
    refetchInterval: 5 * 60_000,
    refetchOnWindowFocus: true,
    retry: false,
  })
  return !!q.data && q.data.build !== __BUILD_ID__
}
