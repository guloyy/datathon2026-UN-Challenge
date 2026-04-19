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

interface PrioritizeRow {
  rank: number
  country_iso3?: string
  country_name?: string
  sector?: string
  year?: number
  score: number
}

interface ProCon {
  feature: string
  weight: number
  target_value: number
  contribution?: number
  label: string
}

interface CounterfactualRow {
  country_iso3?: string
  country_name?: string
  sector?: string
  score?: number
}

interface DisplacedRow extends CounterfactualRow {
  default_rank?: number
  new_rank?: number
  score_delta?: number
  default_score?: number
  new_score?: number
}

interface BlockingRow {
  country_iso3?: string
  country_name?: string
  sector?: string
  score_gap: number
  dominant_features: {
    feature: string
    row_value: number
    target_value: number
    contribution_gap?: number
  }[]
}

interface PrioritizeResponse {
  mode: string
  intent: Record<string, unknown>
  weights: Record<string, number>
  weight_deviation_from_default: number
  short_circuited: boolean
  reason?: string
  target: (PrioritizeRow & { rank: number }) | null
  ranking: PrioritizeRow[]
  pros: ProCon[]
  cons: ProCon[]
  counterfactual: {
    default_top: CounterfactualRow | null
    displaced: DisplacedRow[]
    archetypes: Record<string, CounterfactualRow | null>
  } | null
  explanation: string
}

interface PrioritizeError {
  error: string
  reason: string
  blocking_rows: BlockingRow[]
}

function gapColor(score: number): string {
  if (score >= 0.7) return '#dc2626'
  if (score >= 0.4) return '#f97316'
  return '#facc15'
}

type Tab = 'query' | 'prioritize'

export default function App() {
  const [tab, setTab] = useState<Tab>('prioritize')
  return (
    <div className="min-h-screen bg-gray-950 text-gray-100 flex flex-col">
      <header className="border-b border-gray-800 px-8 py-4 flex items-center gap-3">
        <span className="text-2xl">🌍</span>
        <div>
          <h1 className="text-lg font-semibold text-white leading-none">Geo-Insight</h1>
          <p className="text-xs text-gray-400 mt-0.5">Which humanitarian crises are most overlooked?</p>
        </div>
        <nav className="ml-auto flex gap-1 text-xs">
          <button
            onClick={() => setTab('prioritize')}
            className={`px-3 py-1.5 rounded font-medium transition-colors ${
              tab === 'prioritize' ? 'bg-blue-600 text-white' : 'text-gray-400 hover:text-white'
            }`}
          >
            Prioritize
          </button>
          <button
            onClick={() => setTab('query')}
            className={`px-3 py-1.5 rounded font-medium transition-colors ${
              tab === 'query' ? 'bg-blue-600 text-white' : 'text-gray-400 hover:text-white'
            }`}
          >
            Query
          </button>
        </nav>
      </header>
      <main className="flex-1 px-8 py-8 max-w-6xl mx-auto w-full">
        {tab === 'query' ? <QueryPanel /> : <PrioritizePanel />}
      </main>
    </div>
  )
}

// ── Prioritize panel ──────────────────────────────────────────────────────────

function PrioritizePanel() {
  const [query, setQuery] = useState('')
  const [k, setK] = useState(3)
  const [useLLM, setUseLLM] = useState(true)
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState<PrioritizeResponse | null>(null)
  const [error, setError] = useState<PrioritizeError | string | null>(null)

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault()
    if (!query.trim()) return
    setLoading(true)
    setError(null)
    setResult(null)
    try {
      const { data } = await axios.post<PrioritizeResponse>('/prioritize', {
        query,
        k,
        use_llm_prose: useLLM,
      })
      setResult(data)
    } catch (err: unknown) {
      if (axios.isAxiosError(err) && err.response?.data?.detail) {
        const detail = err.response.data.detail
        if (typeof detail === 'object' && detail.error === 'infeasible') {
          setError(detail as PrioritizeError)
        } else {
          setError(String(detail))
        }
      } else {
        setError(String(err))
      }
    } finally {
      setLoading(false)
    }
  }

  const weightEntries = result
    ? Object.entries(result.weights)
        .filter(([, w]) => w > 0.001)
        .sort(([, a], [, b]) => b - a)
    : []

  return (
    <>
      <form onSubmit={handleSubmit} className="flex flex-col gap-3">
        <textarea
          className="w-full bg-gray-900 border border-gray-700 rounded-lg px-4 py-3 text-gray-100 placeholder-gray-500 focus:outline-none focus:border-blue-500 resize-none text-sm"
          rows={2}
          placeholder='e.g. "prioritize Brazil for water supply" or "prioritize Yemen" or "how should we fund the Middle East"'
          value={query}
          onChange={e => setQuery(e.target.value)}
        />
        <div className="flex items-center gap-4 flex-wrap">
          <label className="text-sm text-gray-400 flex items-center gap-2">
            Top-k target
            <select
              className="bg-gray-900 border border-gray-700 rounded px-2 py-1 text-gray-100 text-sm"
              value={k}
              onChange={e => setK(Number(e.target.value))}
            >
              {[1, 2, 3, 5, 10].map(v => (
                <option key={v}>{v}</option>
              ))}
            </select>
          </label>
          <label className="text-sm text-gray-400 flex items-center gap-2">
            <input
              type="checkbox"
              checked={useLLM}
              onChange={e => setUseLLM(e.target.checked)}
              className="accent-blue-500"
            />
            LLM prose
          </label>
          <button
            type="submit"
            disabled={loading || !query.trim()}
            className="ml-auto bg-blue-600 hover:bg-blue-500 disabled:opacity-40 disabled:cursor-not-allowed text-white px-6 py-2 rounded-lg text-sm font-medium transition-colors"
          >
            {loading ? 'Optimizing…' : 'Prioritize'}
          </button>
        </div>
      </form>

      {/* Infeasible error */}
      {error && typeof error === 'object' && 'error' in error && (
        <div className="mt-6 bg-red-950 border border-red-800 rounded-lg p-4">
          <p className="text-red-300 text-sm font-semibold mb-2">Infeasible</p>
          <p className="text-red-200 text-xs mb-3">{error.reason}</p>
          {error.blocking_rows.length > 0 && (
            <div>
              <p className="text-xs font-semibold text-red-400 uppercase tracking-wide mb-2">Blocking rows</p>
              <div className="space-y-2">
                {error.blocking_rows.map((br, i) => (
                  <div key={i} className="bg-red-900/40 rounded px-3 py-2 text-xs">
                    <p className="text-red-200">
                      <span className="font-semibold">{br.country_name ?? br.country_iso3}</span>
                      {br.sector && <span className="text-red-400"> × {br.sector}</span>}
                      <span className="ml-2 font-mono text-red-300">score gap +{br.score_gap.toFixed(3)}</span>
                    </p>
                    <ul className="mt-1 ml-4 text-red-300 list-disc">
                      {br.dominant_features.map((df, j) => (
                        <li key={j}>
                          {df.feature}: row {df.row_value.toFixed(2)} vs target {df.target_value.toFixed(2)}
                        </li>
                      ))}
                    </ul>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}

      {error && typeof error === 'string' && (
        <div className="mt-6 bg-red-950 border border-red-800 text-red-300 rounded-lg px-4 py-3 text-sm">
          {error}
        </div>
      )}

      {result && (
        <div className="mt-8 flex flex-col gap-6">
          {/* Verdict */}
          <div className="bg-gray-900 border border-gray-800 rounded-lg p-5">
            <div className="flex items-start gap-4 flex-wrap">
              <div>
                <p className="text-xs font-semibold text-gray-400 uppercase tracking-wide">Mode</p>
                <p className="text-sm text-gray-100">{result.mode}</p>
              </div>
              {result.target && (
                <div>
                  <p className="text-xs font-semibold text-gray-400 uppercase tracking-wide">Target</p>
                  <p className="text-sm text-gray-100">
                    {result.target.country_name ?? result.target.country_iso3}
                    {result.target.sector && result.target.sector !== 'ALL' && (
                      <span className="text-gray-500"> × {result.target.sector}</span>
                    )}
                    <span className="ml-2 text-blue-400 font-semibold">rank #{result.target.rank}</span>
                  </p>
                </div>
              )}
              <div>
                <p className="text-xs font-semibold text-gray-400 uppercase tracking-wide">L2 deviation</p>
                <p className="text-sm font-mono text-gray-100">
                  {result.weight_deviation_from_default.toFixed(3)}
                </p>
              </div>
              {result.short_circuited && (
                <span className="bg-green-900 text-green-300 text-xs px-2 py-1 rounded">
                  short-circuit (already top-k)
                </span>
              )}
            </div>
            {result.explanation && (
              <p className="mt-4 text-sm text-gray-300 leading-relaxed italic">{result.explanation}</p>
            )}
          </div>

          {/* Weights + Pros/Cons */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <div>
              <p className="text-xs font-semibold text-gray-400 uppercase tracking-wide mb-3">Weights</p>
              <div className="space-y-2">
                {weightEntries.map(([name, w]) => (
                  <div key={name} className="text-xs">
                    <div className="flex justify-between text-gray-300">
                      <span className="font-mono">{name}</span>
                      <span className="font-mono">{w.toFixed(3)}</span>
                    </div>
                    <div className="mt-1 bg-gray-800 rounded-sm h-1.5 overflow-hidden">
                      <div className="h-full bg-blue-500" style={{ width: `${w * 100}%` }} />
                    </div>
                  </div>
                ))}
              </div>
            </div>

            <div>
              <p className="text-xs font-semibold text-green-400 uppercase tracking-wide mb-3">
                Optimized for (pros)
              </p>
              {result.pros.length === 0 ? (
                <p className="text-xs text-gray-500">none</p>
              ) : (
                <ul className="space-y-2 text-xs">
                  {result.pros.map((p, i) => (
                    <li key={i} className="bg-green-950/40 border border-green-900 rounded px-3 py-2">
                      <p className="text-green-200 font-semibold">{p.label}</p>
                      <p className="text-green-400 font-mono mt-1">
                        weight {p.weight.toFixed(3)} × target {p.target_value.toFixed(2)} = {(p.contribution ?? 0).toFixed(3)}
                      </p>
                    </li>
                  ))}
                </ul>
              )}
            </div>

            <div>
              <p className="text-xs font-semibold text-orange-400 uppercase tracking-wide mb-3">
                Ignored despite high target value (cons)
              </p>
              {result.cons.length === 0 ? (
                <p className="text-xs text-gray-500">none</p>
              ) : (
                <ul className="space-y-2 text-xs">
                  {result.cons.map((c, i) => (
                    <li key={i} className="bg-orange-950/40 border border-orange-900 rounded px-3 py-2">
                      <p className="text-orange-200 font-semibold">{c.label}</p>
                      <p className="text-orange-400 font-mono mt-1">
                        weight {c.weight.toFixed(3)} · target value {c.target_value.toFixed(2)}
                      </p>
                    </li>
                  ))}
                </ul>
              )}
            </div>
          </div>

          {/* Counterfactual */}
          {result.counterfactual && (
            <div>
              <p className="text-xs font-semibold text-gray-400 uppercase tracking-wide mb-3">
                Counterfactuals — it could have been…
              </p>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-xs">
                {result.counterfactual.default_top && (
                  <div className="bg-gray-900 border border-gray-800 rounded px-3 py-2">
                    <p className="text-gray-500 uppercase text-[10px] tracking-wide">Default top</p>
                    <p className="text-gray-100 font-semibold">
                      {result.counterfactual.default_top.country_name ?? result.counterfactual.default_top.country_iso3}
                    </p>
                    <p className="text-gray-400 font-mono">
                      score {(result.counterfactual.default_top.score ?? 0).toFixed(3)}
                    </p>
                  </div>
                )}
                {result.counterfactual.archetypes?.severity_max && (
                  <div className="bg-gray-900 border border-gray-800 rounded px-3 py-2">
                    <p className="text-gray-500 uppercase text-[10px] tracking-wide">Severity-max archetype</p>
                    <p className="text-gray-100 font-semibold">
                      {result.counterfactual.archetypes.severity_max.country_name ??
                        result.counterfactual.archetypes.severity_max.country_iso3}
                    </p>
                    <p className="text-gray-400 font-mono">
                      score {(result.counterfactual.archetypes.severity_max.score ?? 0).toFixed(3)}
                    </p>
                  </div>
                )}
                {result.counterfactual.archetypes?.gap_max && (
                  <div className="bg-gray-900 border border-gray-800 rounded px-3 py-2">
                    <p className="text-gray-500 uppercase text-[10px] tracking-wide">Gap-max archetype</p>
                    <p className="text-gray-100 font-semibold">
                      {result.counterfactual.archetypes.gap_max.country_name ??
                        result.counterfactual.archetypes.gap_max.country_iso3}
                    </p>
                    <p className="text-gray-400 font-mono">
                      score {(result.counterfactual.archetypes.gap_max.score ?? 0).toFixed(3)}
                    </p>
                  </div>
                )}
              </div>
              {result.counterfactual.displaced && result.counterfactual.displaced.length > 0 && (
                <div className="mt-3">
                  <p className="text-[10px] text-gray-500 uppercase tracking-wide mb-1">Displaced from top-k under your weights</p>
                  <ul className="text-xs text-gray-400 list-disc ml-5">
                    {result.counterfactual.displaced.map((d, i) => (
                      <li key={i}>
                        <span className="text-gray-200">{d.country_name ?? d.country_iso3}</span>
                        : rank {d.default_rank} → {d.new_rank}, score delta {(d.score_delta ?? 0).toFixed(3)}
                      </li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          )}

          {/* Ranking */}
          <div>
            <p className="text-xs font-semibold text-gray-400 uppercase tracking-wide mb-3">Ranking</p>
            <div className="overflow-x-auto rounded-lg border border-gray-800">
              <table className="w-full text-sm">
                <thead className="bg-gray-900 text-gray-400 text-xs uppercase">
                  <tr>
                    <th className="px-3 py-2 text-left">#</th>
                    <th className="px-3 py-2 text-left">Country</th>
                    <th className="px-3 py-2 text-left">Sector</th>
                    <th className="px-3 py-2 text-left">Year</th>
                    <th className="px-3 py-2 text-right">Score</th>
                  </tr>
                </thead>
                <tbody>
                  {result.ranking.map((r, i) => {
                    const isTarget =
                      result.target != null &&
                      r.country_iso3 === result.target.country_iso3 &&
                      r.sector === result.target.sector
                    return (
                      <tr
                        key={i}
                        className={`border-t border-gray-800 ${
                          isTarget ? 'bg-blue-950/50' : 'hover:bg-gray-900'
                        }`}
                      >
                        <td className="px-3 py-2 font-mono text-gray-300">{r.rank}</td>
                        <td className="px-3 py-2 text-gray-200">
                          {r.country_name ?? r.country_iso3}
                        </td>
                        <td className="px-3 py-2 font-mono text-gray-400 text-xs">{r.sector}</td>
                        <td className="px-3 py-2 font-mono text-gray-400 text-xs">{r.year}</td>
                        <td className="px-3 py-2 font-mono text-gray-200 text-right">
                          {r.score.toFixed(3)}
                        </td>
                      </tr>
                    )
                  })}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}

      {!result && !loading && !error && (
        <div className="mt-20 text-center text-gray-600">
          <p className="text-4xl mb-4">🎯</p>
          <p className="text-sm">
            Try: <em className="text-gray-400">prioritize Brazil for water supply</em>, or{' '}
            <em className="text-gray-400">how should we fund the Middle East</em>
          </p>
        </div>
      )}
    </>
  )
}

// ── Query panel (unchanged from original /query flow) ─────────────────────────

function QueryPanel() {
  const [query, setQuery] = useState('')
  const [yearFrom, setYearFrom] = useState(2019)
  const [yearTo, setYearTo] = useState(2025)
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
        query,
        year_from: yearFrom,
        year_to: yearTo,
        limit: 50,
      })
      setResult(data)
    } catch (err: unknown) {
      const msg = axios.isAxiosError(err) ? err.response?.data?.detail ?? err.message : 'Unknown error'
      setError(String(msg))
    } finally {
      setLoading(false)
    }
  }

  const chartRows = result?.rows.filter(r => r.gap_score != null).slice(0, 20) ?? []
  const isoScoreMap: Record<string, number> = {}
  result?.rows.forEach(r => {
    if (r.country_iso3 && r.gap_score != null) isoScoreMap[r.country_iso3 as string] = r.gap_score as number
  })

  return (
    <>
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
            <select
              className="bg-gray-900 border border-gray-700 rounded px-2 py-1 text-gray-100 text-sm"
              value={yearFrom}
              onChange={e => setYearFrom(Number(e.target.value))}
            >
              {[2019, 2020, 2021, 2022, 2023, 2024, 2025].map(y => (
                <option key={y}>{y}</option>
              ))}
            </select>
          </label>
          <label className="text-sm text-gray-400 flex items-center gap-2">
            to
            <select
              className="bg-gray-900 border border-gray-700 rounded px-2 py-1 text-gray-100 text-sm"
              value={yearTo}
              onChange={e => setYearTo(Number(e.target.value))}
            >
              {[2019, 2020, 2021, 2022, 2023, 2024, 2025].map(y => (
                <option key={y}>{y}</option>
              ))}
            </select>
          </label>
          <label className="text-sm text-gray-400 flex items-center gap-2 ml-auto">
            <input
              type="checkbox"
              checked={showSql}
              onChange={e => setShowSql(e.target.checked)}
              className="accent-blue-500"
            />
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

      {error && (
        <div className="mt-6 bg-red-950 border border-red-800 text-red-300 rounded-lg px-4 py-3 text-sm">{error}</div>
      )}

      {result && (
        <div className="mt-8 flex flex-col gap-8">
          {showSql && (
            <div>
              <p className="text-xs font-semibold text-gray-400 mb-2 uppercase tracking-wide">Generated SQL</p>
              <pre className="bg-gray-900 border border-gray-700 rounded-lg p-4 text-xs text-green-300 overflow-x-auto whitespace-pre-wrap">
                {result.sql}
              </pre>
            </div>
          )}
          <p className="text-sm text-gray-400">{result.row_count} rows returned</p>
          {chartRows.length > 0 && (
            <div>
              <p className="text-xs font-semibold text-gray-400 mb-3 uppercase tracking-wide">Gap Score by Country</p>
              <ResponsiveContainer width="100%" height={260}>
                <BarChart data={chartRows} margin={{ bottom: 60 }}>
                  <XAxis dataKey="country_iso3" tick={{ fill: '#9ca3af', fontSize: 11 }} angle={-45} textAnchor="end" />
                  <YAxis tick={{ fill: '#9ca3af', fontSize: 11 }} domain={[0, 1]} />
                  <Tooltip contentStyle={{ background: '#1f2937', border: 'none', fontSize: 12 }} />
                  <Bar dataKey="gap_score" radius={[3, 3, 0, 0]}>
                    {chartRows.map((r, i) => (
                      <Cell key={i} fill={gapColor(r.gap_score as number)} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          )}
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
            </div>
          )}
          <div>
            <p className="text-xs font-semibold text-gray-400 mb-3 uppercase tracking-wide">Results Table</p>
            <div className="overflow-x-auto rounded-lg border border-gray-800">
              <table className="w-full text-sm">
                <thead className="bg-gray-900 text-gray-400 text-xs uppercase">
                  <tr>
                    {result.columns.map(col => (
                      <th key={col} className="px-4 py-2 text-left font-medium whitespace-nowrap">
                        {col}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {result.rows.map((row, i) => (
                    <tr key={i} className="border-t border-gray-800 hover:bg-gray-900 transition-colors">
                      {result.columns.map(col => (
                        <td key={col} className="px-4 py-2 text-gray-300 whitespace-nowrap font-mono text-xs">
                          {row[col] == null ? (
                            <span className="text-gray-600">—</span>
                          ) : typeof row[col] === 'number' ? (
                            (row[col] as number) > 1000
                              ? (row[col] as number).toLocaleString()
                              : (row[col] as number).toFixed(3)
                          ) : (
                            String(row[col])
                          )}
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
    </>
  )
}
