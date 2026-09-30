import { Bar, BarChart, CartesianGrid, LabelList, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'

export function VariableImportanceChart({
  importance,
  descriptions,
}: {
  importance: Record<string, number>
  descriptions?: Record<string, string>
}) {
  const data = Object.entries(importance)
    .map(([k, v]) => ({ key: k.toUpperCase(), value: v, label: descriptions?.[k] ?? k }))
    .sort((a, b) => b.value - a.value)
  return (
    <ResponsiveContainer width="100%" height={Math.max(120, data.length * 34 + 24)}>
      <BarChart data={data} layout="vertical" margin={{ top: 4, right: 44, bottom: 4, left: 4 }}>
        <CartesianGrid horizontal={false} stroke="var(--border)" />
        <XAxis type="number" domain={[0, 'dataMax']} hide />
        <YAxis
          type="category"
          dataKey="key"
          width={52}
          tickLine={false}
          axisLine={false}
          tick={{ fontSize: 11, fill: 'var(--muted-foreground)' }}
        />
        <Tooltip
          cursor={{ fill: 'var(--muted)', opacity: 0.4 }}
          contentStyle={{
            background: 'var(--popover)',
            border: '1px solid var(--border)',
            borderRadius: 8,
            fontSize: 12,
          }}
          formatter={(v, _n, item) => [`${Number(v).toFixed(1)}%`, item?.payload?.label]}
          labelFormatter={() => ''}
        />
        <Bar dataKey="value" fill="var(--primary)" radius={[0, 4, 4, 0]} barSize={16}>
          <LabelList
            dataKey="value"
            position="right"
            formatter={(v) => `${Number(v).toFixed(0)}%`}
            style={{ fontSize: 11, fill: 'var(--muted-foreground)' }}
          />
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  )
}
