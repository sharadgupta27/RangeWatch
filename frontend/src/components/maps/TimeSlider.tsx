import { useSelector } from '@tanstack/react-store'
import { Pause, Play } from 'lucide-react'
import { useEffect, useState } from 'react'

import { Button } from '@/components/ui/button'
import { Slider } from '@/components/ui/slider'
import { mapActions, mapStore } from '@/store/mapStore'

/**
 * "Invasion timeline": filters occurrence tiles to records up to the selected year
 * (server-side via the pg_tileserv `max_year` argument) with a play-through animation.
 */
export function TimeSlider({ minYear, maxYear }: { minYear: number; maxYear: number }) {
  const storeYear = useSelector(mapStore, (s) => s.maxYear)
  const playing = useSelector(mapStore, (s) => s.playing)
  const [draft, setDraft] = useState<number>(storeYear ?? maxYear)

  useEffect(() => setDraft(storeYear ?? maxYear), [storeYear, maxYear])

  useEffect(() => {
    if (!playing) return
    const id = window.setInterval(() => {
      const cur = mapStore.state.maxYear ?? minYear
      if (cur >= maxYear) {
        mapActions.setPlaying(false)
        mapActions.setMaxYear(null)
        return
      }
      mapActions.setMaxYear(cur + 1)
    }, 700)
    return () => window.clearInterval(id)
  }, [playing, minYear, maxYear])

  const togglePlay = () => {
    if (!playing && (storeYear === null || storeYear >= maxYear)) mapActions.setMaxYear(minYear)
    mapActions.setPlaying(!playing)
  }

  return (
    <div className="flex items-center gap-3 rounded-lg border bg-card/85 px-3 py-2 shadow-sm backdrop-blur">
      <Button
        size="icon"
        variant="ghost"
        className="size-7"
        onClick={togglePlay}
        aria-label={playing ? 'Pause timeline' : 'Play invasion timeline'}
      >
        {playing ? <Pause /> : <Play />}
      </Button>
      <span className="w-10 text-xs tabular-nums text-muted-foreground">{minYear}</span>
      <Slider
        className="w-56"
        min={minYear}
        max={maxYear}
        step={1}
        value={[draft]}
        onValueChange={([v]) => v !== undefined && setDraft(v)}
        // Only refetch tiles once the user lets go.
        onValueCommit={([v]) => v !== undefined && mapActions.setMaxYear(v >= maxYear ? null : v)}
      />
      <span className="w-20 text-right font-mono text-xs font-medium tabular-nums">
        ≤ {draft}
      </span>
    </div>
  )
}
