import { Bar, BarChart, CartesianGrid, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'

import type { ModelVersionDetail } from '@/api/types'

/** Per-fold spatial block CV metrics — shows spread, not just the mean. */
export function CvFoldsChart({ folds }: { folds: ModelVersionDetail['folds'] }) {
  const data = folds.map((f) => ({
    fold: `Fold ${f.fold + 1}`,
    AUC: f.auc ?? 0,
    TSS: f.tss ?? 0,
    CBI: f.cbi ?? 0,
  }))
  return (
    <ResponsiveContainer width="100%" height={200}>
      <BarChart data={data} margin={{ top: 8, right: 8, bottom: 0, left: -18 }}>
        <CartesianGrid vertical={false} stroke="var(--border)" />
        <XAxis dataKey="fold" tickLine={false} axisLine={false} tick={{ fontSize: 11, fill: 'var(--muted-foreground)' }} />
        <YAxis domain={[-0.2, 1]} tickLine={false} axisLine={false} tick={{ fontSize: 11, fill: 'var(--muted-foreground)' }} />
        <Tooltip
          cursor={{ fill: 'var(--muted)', opacity: 0.35 }}
          contentStyle={{ background: 'var(--popover)', border: '1px solid var(--border)', borderRadius: 8, fontSize: 12 }}
          formatter={(v) => Number(v).toFixed(3)}
        />
        <Legend iconType="circle" iconSize={8} wrapperStyle={{ fontSize: 11 }} />
        <Bar dataKey="AUC" fill="var(--sev-suitability)" radius={[3, 3, 0, 0]} />
        <Bar dataKey="TSS" fill="var(--sev-spread)" radius={[3, 3, 0, 0]} />
        <Bar dataKey="CBI" fill="var(--sev-analogy)" radius={[3, 3, 0, 0]} />
      </BarChart>
    </ResponsiveContainer>
  )
}
