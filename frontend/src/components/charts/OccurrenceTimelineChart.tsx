import {
  Bar,
  CartesianGrid,
  ComposedChart,
  Legend,
  Line,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'

import type { Timeline } from '@/api/types'

export function OccurrenceTimelineChart({
  timeline,
  highlightYear,
}: {
  timeline: Timeline
  highlightYear?: number | null
}) {
  const data = timeline.years.slice(-40)
  return (
    <ResponsiveContainer width="100%" height={220}>
      <ComposedChart data={data} margin={{ top: 8, right: 8, bottom: 0, left: 0 }}>
        <CartesianGrid vertical={false} stroke="var(--border)" />
        <XAxis
          dataKey="year"
          tickLine={false}
          axisLine={false}
          tick={{ fontSize: 11, fill: 'var(--muted-foreground)' }}
          minTickGap={16}
        />
        <YAxis
          yAxisId="n"
          tickLine={false}
          axisLine={false}
          tick={{ fontSize: 11, fill: 'var(--muted-foreground)' }}
          width={44}
        />
        <YAxis yAxisId="cum" orientation="right" hide />
        <Tooltip
          cursor={{ fill: 'var(--muted)', opacity: 0.35 }}
          contentStyle={{
            background: 'var(--popover)',
            border: '1px solid var(--border)',
            borderRadius: 8,
            fontSize: 12,
          }}
        />
        <Legend iconType="circle" iconSize={8} wrapperStyle={{ fontSize: 11 }} />
        <Bar yAxisId="n" dataKey="native" stackId="a" name="Native" fill="var(--native)" />
        <Bar yAxisId="n" dataKey="introduced" stackId="a" name="Introduced" fill="var(--introduced)" />
        <Bar
          yAxisId="n"
          dataKey="unknown"
          stackId="a"
          name="Unlabelled"
          fill="var(--unknown)"
          radius={[3, 3, 0, 0]}
        />
        <Line
          yAxisId="cum"
          type="monotone"
          dataKey="cumulative"
          name="Cumulative"
          stroke="var(--foreground)"
          strokeOpacity={0.6}
          dot={false}
          strokeWidth={1.5}
        />
        {highlightYear != null && (
          <ReferenceLine yAxisId="n" x={highlightYear} stroke="var(--primary)" strokeDasharray="4 3" />
        )}
      </ComposedChart>
    </ResponsiveContainer>
  )
}
