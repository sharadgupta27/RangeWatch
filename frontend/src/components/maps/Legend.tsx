import type { RasterLayer } from '@/api/types'
import { cn } from '@/lib/utils'

export function RasterLegend({ layer, className }: { layer: RasterLayer; className?: string }) {
  return (
    <div
      className={cn(
        'rounded-lg border bg-card/85 px-3 py-2 text-[11px] shadow-sm backdrop-blur',
        className,
      )}
    >
      <div className="mb-1.5 font-medium">{layer.label}</div>
      {layer.kind === 'continuous' ? (
        <>
          <div
            className="h-2 w-44 rounded-full"
            style={{
              background: `linear-gradient(to right, ${layer.legend.map((e) => e.color).join(', ')})`,
            }}
          />
          <div className="mt-1 flex w-44 justify-between text-muted-foreground">
            <span>{layer.legend[0]?.label}</span>
            <span>{layer.legend[layer.legend.length - 1]?.label}</span>
          </div>
        </>
      ) : (
        <ul className="space-y-1">
          {layer.legend.map((e) => (
            <li key={e.value} className="flex items-center gap-2">
              <span className="size-2.5 rounded-sm" style={{ background: e.color }} />
              <span className="text-muted-foreground">{e.label}</span>
            </li>
          ))}
        </ul>
      )}
    </div>
  )
}

export function OccurrenceLegend({ className }: { className?: string }) {
  const items = [
    ['var(--native)', 'Native'],
    ['var(--introduced)', 'Introduced'],
    ['var(--unknown)', 'Unlabelled'],
  ] as const
  return (
    <div
      className={cn(
        'rounded-lg border bg-card/85 px-3 py-2 text-[11px] shadow-sm backdrop-blur',
        className,
      )}
    >
      <div className="mb-1.5 font-medium">Occurrences</div>
      <ul className="space-y-1">
        {items.map(([c, l]) => (
          <li key={l} className="flex items-center gap-2">
            <span className="size-2.5 rounded-full" style={{ background: c }} />
            <span className="text-muted-foreground">{l}</span>
          </li>
        ))}
      </ul>
      <p className="mt-1.5 max-w-44 text-muted-foreground">
        Circles are aggregated at low zoom; size ∝ √count.
      </p>
    </div>
  )
}
