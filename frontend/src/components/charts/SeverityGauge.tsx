import { Group } from '@visx/group'
import { Arc } from '@visx/shape'

import type { SeverityResult } from '@/api/types'

export const SEVERITY_COMPONENTS = [
  { key: 'suitability', label: 'Suitability', color: 'var(--sev-suitability)' },
  { key: 'climate_analogy', label: 'Climate analogy', color: 'var(--sev-analogy)' },
  { key: 'spread_rate', label: 'Spread rate', color: 'var(--sev-spread)' },
  { key: 'ecological_impact_prior', label: 'Impact prior', color: 'var(--sev-impact)' },
] as const

const START = -Math.PI / 2
const SWEEP = Math.PI

/**
 * Half-donut gauge whose arc is split into the weighted component contributions (w·S), so the
 * score is never shown as an opaque number.
 */
export function SeverityGauge({ severity, size = 260 }: { severity: SeverityResult; size?: number }) {
  const r = size / 2
  const thickness = size * 0.12
  let angle = START
  const segments = SEVERITY_COMPONENTS.map((c) => {
    const v = severity.contributions[c.key]
    const seg = { ...c, start: angle, end: angle + v * SWEEP }
    angle = seg.end
    return seg
  })
  return (
    <svg
      width={size}
      height={r + 24}
      viewBox={`0 0 ${size} ${r + 24}`}
      role="img"
      aria-label={`Severity ${severity.score_0_100.toFixed(0)} of 100, ${severity.category}`}
    >
      <Group top={r} left={r}>
        <Arc
          startAngle={START}
          endAngle={START + SWEEP}
          outerRadius={r - 4}
          innerRadius={r - 4 - thickness}
          cornerRadius={3}
          fill="var(--muted)"
        />
        {segments.map((s) =>
          s.end - s.start > 0.001 ? (
            <Arc
              key={s.key}
              startAngle={s.start}
              endAngle={s.end}
              outerRadius={r - 4}
              innerRadius={r - 4 - thickness}
              padAngle={0.008}
              fill={s.color}
            />
          ) : null,
        )}
        <text
          textAnchor="middle"
          y={-r * 0.18}
          className="fill-foreground"
          style={{ fontSize: size * 0.2, fontWeight: 700, fontVariantNumeric: 'tabular-nums' }}
        >
          {severity.score_0_100.toFixed(0)}
        </text>
        <text
          textAnchor="middle"
          y={r * 0.02}
          className="fill-muted-foreground uppercase"
          style={{ fontSize: 11, letterSpacing: '0.08em', fontWeight: 600 }}
        >
          {severity.category}
        </text>
        <text x={-r + 6} y={18} className="fill-muted-foreground" style={{ fontSize: 10 }}>
          0
        </text>
        <text x={r - 6} y={18} textAnchor="end" className="fill-muted-foreground" style={{ fontSize: 10 }}>
          100
        </text>
      </Group>
    </svg>
  )
}
