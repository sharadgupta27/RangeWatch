import { CartesianGrid, Line, LineChart, ReferenceArea, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'

import type { ResponseCurvesOut } from '@/api/types'

/** Small multiples of partial-dependence curves; the shaded band is the presences' 5–95% range. */
export function ResponseCurvesChart({
  curves,
  descriptions,
}: {
  curves: ResponseCurvesOut
  descriptions?: Record<string, string>
}) {
  return (
    <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-3">
      {curves.curves.map((c) => {
        const data = c.x.map((x, i) => ({ x, y: c.y[i] }))
        const [lo, hi] = c.presence_range
        return (
          <figure key={c.predictor} className="rounded-lg border p-2" aria-label={`Response to ${c.predictor}`}>
            <figcaption className="px-1 text-xs">
              <span className="font-medium uppercase">{c.predictor}</span>{' '}
              <span className="text-muted-foreground">{descriptions?.[c.predictor] ?? ''}</span>
            </figcaption>
            <ResponsiveContainer width="100%" height={130}>
              <LineChart data={data} margin={{ top: 6, right: 8, bottom: 0, left: -18 }}>
                <CartesianGrid stroke="var(--border)" vertical={false} />
                {lo != null && hi != null && (
                  <ReferenceArea x1={lo} x2={hi} fill="var(--primary)" fillOpacity={0.08} />
                )}
                <XAxis
                  dataKey="x"
                  type="number"
                  domain={['dataMin', 'dataMax']}
                  tick={{ fontSize: 10, fill: 'var(--muted-foreground)' }}
                  tickFormatter={(v: number) => (Math.abs(v) >= 1000 ? `${(v / 1000).toFixed(1)}k` : v.toFixed(0))}
                />
                <YAxis domain={[0, 'auto']} tick={{ fontSize: 10, fill: 'var(--muted-foreground)' }} />
                <Tooltip
                  contentStyle={{
                    background: 'var(--popover)',
                    border: '1px solid var(--border)',
                    borderRadius: 8,
                    fontSize: 12,
                  }}
                  formatter={(v) => [Number(v).toFixed(3), 'suitability']}
                  labelFormatter={(v) => `${c.predictor} = ${Number(v).toFixed(1)}`}
                />
                <Line dataKey="y" stroke="var(--primary)" strokeWidth={2} dot={false} isAnimationActive={false} />
              </LineChart>
            </ResponsiveContainer>
          </figure>
        )
      })}
    </div>
  )
}
