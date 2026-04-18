import { useState } from 'react'
import axios from 'axios'
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from 'recharts'
import { ComposableMap, Geographies, Geography } from 'react-simple-maps'

const GEO_URL = 'https://cdn.jsdelivr.net/npm/world-atlas@2/countries-110m.json'

interface QueryResponse {
  query: string
  sql: string
  columns: string[]
  rows: Record<string, unknown>[]
  row_count: number
}

function gapColor(score: number): string {
  if (score >= 0.7) return '#dc2626'
  if (score >= 0.4) return '#f97316'
  return '#facc15'
}

export default function App() {
  const [query, setQuery] = useState('')
  const [yearFrom, setYearFrom] = useState(2024)
  const [yearTo, setYearTo] = useState(2026)
  const [showSql, setShowSql] = useState(false)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [result, setResult] = useState<QueryResponse | null>(null)

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault()
    if (!query.trim()) return
    setLoading(true)
    setError(null)
    setResult(null)
    try {
      const { data } = await axios.post<QueryResponse>('/query', {
        query, year_from: yearFrom, year_to: yearTo, limit: 50,
      })
      setResult(data)
    } catch (err: unknown) {
      const msg = axios.isAxiosError(err)
        ? err.response?.data?.detail ?? err.message
        : 'Unknown error'
      setError(String(msg))
    } finally {
      setLoading(false)
    }
  }

  const chartRows = result?.rows.filter(r => r.gap_score != null).slice(0, 20) ?? []
  const isoScoreMap: Record<string, number> = {}
  result?.rows.forEach(r => {
    if (r.country_iso3 && r.gap_score != null)
      isoScoreMap[r.country_iso3 as string] = r.gap_score as number
  })

  return (
    <div className="min-h-screen bg-gray-950 text-gray-100 flex flex-col">

      {/* Header */}
      <header className="border-b border-gray-800 px-8 py-4 flex items-center gap-3">
        <span className="text-2xl">🌍</span>
        <div>
          <h1 className="text-lg font-semibold text-white leading-none">Geo-Insight</h1>
          <p className="text-xs text-gray-400 mt-0.5">Which humanitarian crises are most overlooked?</p>
        </div>
        <span className="ml-auto text-xs bg-blue-900 text-blue-300 px-2 py-1 rounded font-mono">
          Databricks AI
        </span>
      </header>

      <main className="flex-1 px-8 py-8 max-w-6xl mx-auto w-full">

        {/* Query form */}
        <form onSubmit={handleSubmit} className="flex flex-col gap-3">
          <textarea
            className="w-full bg-gray-900 border border-gray-700 rounded-lg px-4 py-3 text-gray-100 placeholder-gray-500 focus:outline-none focus:border-blue-500 resize-none text-sm"
            rows={2}
            placeholder="e.g. Which countries have the highest gap between need and funding in 2025?"
            value={query}
            onChange={e => setQuery(e.target.value)}
          />
          <div className="flex items-center gap-4 flex-wrap">
            <label className="text-sm text-gray-400 flex items-center gap-2">
              Year from
              <select className="bg-gray-900 border border-gray-700 rounded px-2 py-1 text-gray-100 text-sm" value={yearFrom} onChange={e => setYearFrom(Number(e.target.value))}>
                {[2024, 2025, 2026].map(y => <option key={y}>{y}</option>)}
              </select>
            </label>
            <label className="text-sm text-gray-400 flex items-center gap-2">
              to
              <select className="bg-gray-900 border border-gray-700 rounded px-2 py-1 text-gray-100 text-sm" value={yearTo} onChange={e => setYearTo(Number(e.target.value))}>
                {[2024, 2025, 2026].map(y => <option key={y}>{y}</option>)}
              </select>
            </label>
            <label className="text-sm text-gray-400 flex items-center gap-2 ml-auto">
              <input type="checkbox" checked={showSql} onChange={e => setShowSql(e.target.checked)} className="accent-blue-500" />
              Show SQL
            </label>
            <button
              type="submit"
              disabled={loading || !query.trim()}
              className="bg-blue-600 hover:bg-blue-500 disabled:opacity-40 disabled:cursor-not-allowed text-white px-6 py-2 rounded-lg text-sm font-medium transition-colors"
            >
              {loading ? 'Analysing…' : 'Analyse'}
            </button>
          </div>
        </form>

        {/* Error */}
        {error && (
          <div className="mt-6 bg-red-950 border border-red-800 text-red-300 rounded-lg px-4 py-3 text-sm">{error}</div>
        )}

        {/* Results */}
        {result && (
          <div className="mt-8 flex flex-col gap-8">

            {showSql && (
              <div>
                <p className="text-xs font-semibold text-gray-400 mb-2 uppercase tracking-wide">Generated SQL</p>
                <pre className="bg-gray-900 border border-gray-700 rounded-lg p-4 text-xs text-green-300 overflow-x-auto whitespace-pre-wrap">{result.sql}</pre>
              </div>
            )}

            <p className="text-sm text-gray-400">{result.row_count} rows returned</p>

            {/* Bar chart */}
            {chartRows.length > 0 && (
              <div>
                <p className="text-xs font-semibold text-gray-400 mb-3 uppercase tracking-wide">Gap Score by Country</p>
                <ResponsiveContainer width="100%" height={260}>
                  <BarChart data={chartRows} margin={{ bottom: 60 }}>
                    <XAxis dataKey="country_iso3" tick={{ fill: '#9ca3af', fontSize: 11 }} angle={-45} textAnchor="end" />
                    <YAxis tick={{ fill: '#9ca3af', fontSize: 11 }} domain={[0, 1]} />
                    <Tooltip contentStyle={{ background: '#1f2937', border: 'none', fontSize: 12 }} />
                    <Bar dataKey="gap_score" radius={[3, 3, 0, 0]}>
                      {chartRows.map((r, i) => <Cell key={i} fill={gapColor(r.gap_score as number)} />)}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>
            )}

            {/* World map */}
            {Object.keys(isoScoreMap).length > 0 && (
              <div>
                <p className="text-xs font-semibold text-gray-400 mb-3 uppercase tracking-wide">World Map — Gap Score</p>
                <ComposableMap projectionConfig={{ scale: 147 }}>
                  <Geographies geography={GEO_URL}>
                    {({ geographies }: { geographies: { rsmKey: string; properties: Record<string, unknown> }[] }) =>
                      geographies.map(geo => {
                        const iso = geo.properties.ISO_A3 as string
                        const score = isoScoreMap[iso]
                        return (
                          <Geography
                            key={geo.rsmKey}
                            geography={geo}
                            fill={score != null ? gapColor(score) : '#1f2937'}
                            stroke="#374151"
                            strokeWidth={0.4}
                            style={{ hover: { fill: '#60a5fa', outline: 'none' } }}
                          />
                        )
                      })
                    }
                  </Geographies>
                </ComposableMap>
                <div className="flex items-center gap-4 mt-1 text-xs text-gray-500">
                  <span className="flex items-center gap-1"><span className="w-3 h-3 rounded-sm bg-yellow-400 inline-block" /> Low</span>
                  <span className="flex items-center gap-1"><span className="w-3 h-3 rounded-sm bg-orange-500 inline-block" /> Medium</span>
                  <span className="flex items-center gap-1"><span className="w-3 h-3 rounded-sm bg-red-600 inline-block" /> High</span>
                  <span className="flex items-center gap-1"><span className="w-3 h-3 rounded-sm bg-gray-800 inline-block" /> No data</span>
                </div>
              </div>
            )}

            {/* Table */}
            <div>
              <p className="text-xs font-semibold text-gray-400 mb-3 uppercase tracking-wide">Results Table</p>
              <div className="overflow-x-auto rounded-lg border border-gray-800">
                <table className="w-full text-sm">
                  <thead className="bg-gray-900 text-gray-400 text-xs uppercase">
                    <tr>{result.columns.map(col => <th key={col} className="px-4 py-2 text-left font-medium whitespace-nowrap">{col}</th>)}</tr>
                  </thead>
                  <tbody>
                    {result.rows.map((row, i) => (
                      <tr key={i} className="border-t border-gray-800 hover:bg-gray-900 transition-colors">
                        {result.columns.map(col => (
                          <td key={col} className="px-4 py-2 text-gray-300 whitespace-nowrap font-mono text-xs">
                            {row[col] == null
                              ? <span className="text-gray-600">—</span>
                              : typeof row[col] === 'number'
                                ? (row[col] as number) > 1000 ? (row[col] as number).toLocaleString() : (row[col] as number).toFixed(3)
                                : String(row[col])}
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>

          </div>
        )}

        {!result && !loading && !error && (
          <div className="mt-20 text-center text-gray-600">
            <p className="text-4xl mb-4">🔍</p>
            <p className="text-sm">Try: <em className="text-gray-400">Which countries have more than 5M people in need but under 30% funded?</em></p>
          </div>
        )}

      </main>
    </div>
  )
}
