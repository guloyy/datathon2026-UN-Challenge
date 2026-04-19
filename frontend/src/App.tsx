import { useState, useEffect, useRef, useCallback, type MouseEvent } from 'react'
import axios from 'axios'
import {
  ScatterChart, Scatter, XAxis, YAxis, ZAxis,
  Tooltip as ReTooltip, ResponsiveContainer, ReferenceLine,
  Label,
} from 'recharts'
import { ComposableMap, Geographies, Geography } from 'react-simple-maps'

const GEO_URL = 'https://cdn.jsdelivr.net/npm/world-atlas@2/countries-110m.json'
const YEARS = [2019, 2020, 2021, 2022, 2023, 2024, 2025]

// ── Weights ───────────────────────────────────────────────────────────────────

interface Weights { scale: number; gap: number; structural: number; trend: number }

const DEFAULT_WEIGHTS: Weights = { scale: 0.25, gap: 0.25, structural: 0.25, trend: 0.25 }

// ── CSV export ────────────────────────────────────────────────────────────────

function exportCsv(rows: ScoreRow[], year: number) {
  const cols: (keyof ScoreRow)[] = [
    'rank', 'country_iso3', 'country_name', 'continent', 'region_name',
    'overlooked_score', 'borda_rank', 'robust',
    'pin', 'coverage_ratio', 'requirements_usd', 'fts_funding_usd',
    'years_underfunded', 'n_coverage_years', 'coverage_slope',
    'gap_component', 'structural_multiplier', 'confidence_weight',
    'data_complete', 'explanation',
  ]
  const header = cols.join(',')
  const body = rows.map(r =>
    cols.map(c => {
      const v = r[c]
      if (v == null) return ''
      if (typeof v === 'string') return `"${v.replace(/"/g, '""')}"`
      return String(v)
    }).join(',')
  ).join('\n')
  const blob = new Blob([header + '\n' + body], { type: 'text/csv' })
  const url  = URL.createObjectURL(blob)
  const a    = document.createElement('a')
  a.href = url
  a.download = `geo-insight-${year}.csv`
  a.click()
  URL.revokeObjectURL(url)
}

// ── Types ─────────────────────────────────────────────────────────────────────

interface QueryResponse {
  query: string; sql: string; columns: string[]
  rows: Record<string, unknown>[]; row_count: number
}

interface ScoreRow {
  rank: number
  country_iso3: string; country_name: string
  continent: string; region_name: string
  overlooked_score: number; gap_component: number
  structural_multiplier: number; confidence_weight: number
  borda_rank: number; robust: boolean
  explanation: string; data_complete: boolean
  pin: number | null; coverage_ratio: number | null
  targeted: number | null; requirements_usd: number | null
  fts_funding_usd: number | null
  years_underfunded: number | null; coverage_slope: number | null
  n_coverage_years: number | null
  pc1: number; pc2: number
}

// ── Analyze types ─────────────────────────────────────────────────────────────

interface BetterCountry {
  country_name: string; country_iso3: string; value: number | null; mcda_rank: number
}

interface DimCard {
  dimension: string; label: string; group: string
  weight: number; value: number | null; percentile: number; narrative: string
  better_country?: BetterCountry
}

interface ProConResult {
  country_iso3: string; country_name: string
  continent: string; region_name: string
  mcda_score: number; mcda_rank: number; n_countries: number
  why_fund: string; why_not: string
  pros: DimCard[]; cons: DimCard[]; neutral: DimCard[]
  pin: number | null; coverage_ratio: number | null
  requirements_usd: number | null; years_underfunded: number | null
  dim_scores: Record<string, number | null>
}

interface RankedRow {
  mcda_rank: number; country_iso3: string; country_name: string
  continent: string; region_name: string; mcda_score: number
  rank_delta: number | null; has_vulnerability: boolean
  [key: string]: unknown
}

interface DimLeader {
  dimension: string; label: string; group: string; weight: number
  country_iso3: string; country_name: string
  value: number; percentile: number; mcda_rank: number
}

interface AnalyzeResponse {
  prompt: string; year: number; sector: string
  weights: Record<string, number>
  importance_scores: Record<string, number>   // 1–10
  interpretation: string
  ranked: RankedRow[]
  pro_con: ProConResult | { error: string } | null
  dimension_leaders: DimLeader[]
  full_rank_map: Record<string, number>
}

// ── Dimension metadata (mirrors mcda_analyzer.py DIMENSIONS) ─────────────────

const DIM_META: Record<string, { label: string; group: string; desc: string; source: string }> = {
  need_scale:          { label: 'People in Need', group: 'Humanitarian Need',
    desc: 'Absolute number of people requiring humanitarian assistance, log-normalised so a 10M-person crisis scores proportionally higher than a 1M one.',
    source: 'OCHA Humanitarian Needs Overview (HNO)' },
  funding_gap:         { label: 'Funding Gap', group: 'Funding',
    desc: '1 − coverage ratio: the share of humanitarian requirements that remain unfunded this year. 1.0 = fully unfunded, 0 = fully funded.',
    source: 'OCHA Financial Tracking Service (FTS)' },
  structural_neglect:  { label: 'Structural Neglect', group: 'Funding',
    desc: 'Mean historical funding gap across all recorded years. Captures chronic underfunding, not just the current year.',
    source: 'FTS (multi-year average)' },
  trend_worsening:     { label: 'Worsening Trend', group: 'Funding',
    desc: 'Rate at which funding coverage is declining year-over-year (negative slope of coverage ratio). Higher = faster deterioration.',
    source: 'FTS (linear regression over available years)' },
  targeting_gap:       { label: 'Targeting Gap', group: 'Humanitarian Need',
    desc: 'Share of people in need not targeted by any humanitarian programme. High = large population missed by the response.',
    source: 'OCHA HNO (targeted vs. PIN)' },
  water_stress:        { label: 'Water Scarcity', group: 'Vulnerability',
    desc: 'Baseline water stress: total water demand relative to renewable freshwater supply. Scores near 1 indicate demand regularly exceeds supply.',
    source: 'WRI Aqueduct Water Risk Atlas' },
  food_insecurity_risk:{ label: 'Food Insecurity Risk', group: 'Vulnerability',
    desc: 'Structural risk of food insecurity driven by production shortfalls, import dependency, and market fragility — independent of current funding.',
    source: 'ND-GAIN / INFORM Risk Index' },
  displacement_risk:   { label: 'Displacement Risk', group: 'Vulnerability',
    desc: 'Risk from internal displacement and cross-border refugee flows that concentrate humanitarian needs and strain host systems.',
    source: 'INFORM Risk Index — displacement sub-index' },
  health_fragility:    { label: 'Health System Fragility', group: 'Vulnerability',
    desc: 'Weakness of national health infrastructure, workforce, and supply chains. Fragile systems cannot absorb the disease burden of a crisis.',
    source: 'INFORM Risk Index — health sub-index' },
  climate_vulnerability:{ label: 'Climate Vulnerability', group: 'Vulnerability',
    desc: 'Exposure and sensitivity to climate shocks, adjusted for adaptive capacity (ND-GAIN index). High = extreme weather directly amplifies needs.',
    source: 'ND-GAIN Country Index' },
  governance_fragility:{ label: 'Governance Fragility', group: 'Vulnerability',
    desc: 'Weakness of state institutions, rule of law, and capacity to respond. Affects humanitarian access, coordination, and durable solutions.',
    source: 'INFORM Risk Index — governance sub-index' },
  disaster_risk:       { label: 'Natural Disaster Exposure', group: 'Vulnerability',
    desc: 'Physical exposure to floods, droughts, earthquakes, and cyclones that repeatedly trigger or deepen humanitarian crises.',
    source: 'INFORM Risk Index — hazard sub-index' },
}

const SECTOR_OPTIONS: { value: string; label: string }[] = [
  { value: 'ALL',  label: 'All Sectors' },
  { value: 'WSH',  label: 'WASH' },
  { value: 'FSC',  label: 'Food Security' },
  { value: 'HEA',  label: 'Health' },
  { value: 'NUT',  label: 'Nutrition' },
  { value: 'PRO',  label: 'Protection' },
  { value: 'SHL',  label: 'Shelter' },
  { value: 'EDU',  label: 'Education' },
]

// ── Score response ─────────────────────────────────────────────────────────────

interface ScoreResponse {
  year: number; rows: ScoreRow[]; row_count: number
  meta: {
    robust_count: number; total_years_in_data: number
    methodology: Record<string, string>
    pca_explained_var: Record<string, number>
    pca_loadings: Record<string, Record<string, number>>
  }
}

// ── Color helpers ─────────────────────────────────────────────────────────────

function lerp(a: number, b: number, t: number) { return a + (b - a) * Math.max(0, Math.min(1, t)) }

function scoreToColor(score: number, robust = false): string {
  if (robust) return '#7c3aed'
  // 0 → steel blue, 0.4 → amber, 0.7 → orange, 1 → deep red
  if (score < 0.4) {
    const t = score / 0.4
    return `rgb(${Math.round(lerp(56,251,t))},${Math.round(lerp(132,146,t))},${Math.round(lerp(200,22,t))})`
  } else if (score < 0.7) {
    const t = (score - 0.4) / 0.3
    return `rgb(${Math.round(lerp(251,239,t))},${Math.round(lerp(146,68,t))},${Math.round(lerp(22,11,t))})`
  } else {
    const t = (score - 0.7) / 0.3
    return `rgb(${Math.round(lerp(239,153,t))},${Math.round(lerp(68,27,t))},${Math.round(lerp(11,27,t))})`
  }
}

function pct(v: number | null | undefined) {
  return v != null ? `${(v * 100).toFixed(0)}%` : '—'
}
function millions(v: number | null | undefined) {
  return v != null ? `${(v / 1e6).toFixed(1)}M` : '—'
}

// ── Mini score bar ─────────────────────────────────────────────────────────────

function ScoreBar({ row, maxScore }: { row: ScoreRow; maxScore: number }) {
  const gapW  = (row.gap_component / maxScore) * 100
  const bonusW = ((row.overlooked_score - row.gap_component) / maxScore) * 100
  return (
    <div className="flex items-center gap-1.5">
      <div className="flex h-2 w-28 rounded overflow-hidden bg-gray-800">
        <div style={{ width: `${gapW}%`, background: scoreToColor(row.overlooked_score) }} />
        <div style={{ width: `${Math.max(0, bonusW)}%` }} className="bg-purple-500 opacity-70" />
      </div>
      <span className="text-xs font-mono" style={{ color: scoreToColor(row.overlooked_score) }}>
        {row.overlooked_score.toFixed(3)}
      </span>
    </div>
  )
}

// ── Bubble chart custom dot ───────────────────────────────────────────────────

// eslint-disable-next-line @typescript-eslint/no-explicit-any
function PcaDot(props: any) {
  const { cx, cy, payload } = props
  if (!cx || !cy) return null
  return (
    <g>
      <circle cx={cx} cy={cy} r={payload.robust ? 7 : 4.5}
        fill={scoreToColor(payload.overlooked_score, payload.robust)}
        fillOpacity={0.85}
        stroke={payload.robust ? '#a78bfa' : 'none'} strokeWidth={1.5} />
      {payload.rank <= 10 && (
        <text x={cx + 8} y={cy + 4} fontSize={9} fill="#9ca3af">{payload.country_iso3}</text>
      )}
    </g>
  )
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
function BubbleDot(props: any) {
  const { cx, cy, payload } = props
  // ZAxis passes area in px²; derive radius from it
  const area  = props.size ?? props.r ?? 200
  const r     = Math.sqrt(area / Math.PI)
  const color = scoreToColor(payload.overlooked_score, payload.robust)
  if (!cx || !cy) return null
  return (
    <g>
      <circle cx={cx} cy={cy} r={r} fill={color} fillOpacity={0.80}
        stroke={payload.robust ? '#a78bfa' : 'none'} strokeWidth={payload.robust ? 2 : 0} />
      {payload.rank <= 8 && (
        <text x={cx} y={cy - r - 4} textAnchor="middle" fontSize={9} fill="#d1d5db">
          {payload.country_iso3}
        </text>
      )}
    </g>
  )
}

// ── Bubble tooltip ────────────────────────────────────────────────────────────

function BubbleTooltip({ active, payload }: { active?: boolean; payload?: { payload: ScoreRow }[] }) {
  if (!active || !payload?.length) return null
  const r = payload[0].payload
  return (
    <div className="bg-gray-900 border border-gray-700 rounded-lg p-3 text-xs shadow-xl max-w-xs">
      <div className="font-semibold text-white flex items-center gap-1.5 mb-1">
        {r.country_name}
        {r.robust && <span className="bg-purple-900 text-purple-300 px-1.5 py-0.5 rounded text-xs">Robust</span>}
      </div>
      <div className="text-gray-400 space-y-0.5">
        <div>Score: <span className="text-white font-mono">{r.overlooked_score.toFixed(3)}</span>
          <span className="text-gray-600 ml-2">Borda #{r.borda_rank}</span></div>
        <div>People in need: <span className="text-orange-300">{millions(r.pin)}</span></div>
        <div>Funded: <span className="text-blue-300">{pct(r.coverage_ratio)}</span></div>
        <div>Requirements: <span className="text-gray-300">${millions(r.requirements_usd)}</span></div>
        {(r.years_underfunded ?? 0) > 0 &&
          <div>Underfunded {r.years_underfunded}/{r.n_coverage_years} years</div>}
      </div>
      <div className="mt-2 text-gray-500 leading-relaxed">{r.explanation}</div>
    </div>
  )
}

// ── PCA tooltip ───────────────────────────────────────────────────────────────

function PcaTooltip({ active, payload }: { active?: boolean; payload?: { payload: ScoreRow }[] }) {
  if (!active || !payload?.length) return null
  const r = payload[0].payload
  return (
    <div className="bg-gray-900 border border-gray-700 rounded-lg p-3 text-xs shadow-xl max-w-xs">
      <div className="font-semibold text-white mb-1">
        {r.country_name} {r.robust ? '★' : ''}
      </div>
      <div className="text-gray-400 space-y-0.5">
        <div>Score: <span className="text-white font-mono">{r.overlooked_score.toFixed(3)}</span></div>
        <div>Borda rank: <span className="text-white">#{r.borda_rank}</span></div>
        <div className="text-gray-500 mt-1">{r.explanation.split(';')[0]}</div>
      </div>
    </div>
  )
}

// ── Main app ─────────────────────────────────────────────────────────────────

export default function App() {
  const [tab, setTab] = useState<'score' | 'query' | 'analyze'>('score')

  // Score state
  const [scoreYear, setScoreYear]     = useState(2024)
  const [weights]                      = useState<Weights>(DEFAULT_WEIGHTS)
  const [sLoading, setSLoading]       = useState(false)
  const [sError, setSError]           = useState<string | null>(null)
  const [sResult, setSResult]         = useState<ScoreResponse | null>(null)
  const [playing, setPlaying]         = useState(false)
  const [animYear, setAnimYear]       = useState(2019)
  const playRef = useRef<ReturnType<typeof setInterval> | null>(null)

  // Map tooltip — fixed position relative to viewport
  const [tooltipPos, setTooltipPos]         = useState({ x: 0, y: 0 })
  const [hoveredCountry, setHoveredCountry] = useState<ScoreRow | null>(null)

  // Analyze state
  const [aPrompt, setAPrompt]         = useState('')
  const [aRefine, setARefine]         = useState('')
  const [aFocusRaw, setAFocusRaw]     = useState('')   // user-typed country (name or ISO3)
  const [aCountry, setACountry]       = useState('')   // resolved ISO3
  const [aYear, setAYear]             = useState(2024)
  const [aLoading, setALoading]       = useState(false)
  const [aError, setAError]           = useState<string | null>(null)
  const [aResult, setAResult]         = useState<AnalyzeResponse | null>(null)
  const [aSector, setASector]         = useState('ALL')
  const [aSortBy, setASortBy]         = useState<string>('mcda_rank')
  const [aSortDir, setASortDir]       = useState<1|-1>(1)
  const [aCompare, setACompare]       = useState<Set<string>>(new Set())
  const [aSearch, setASearch]         = useState('')
  const [dataSource, setDataSource]   = useState<'databricks' | 'parquet' | 'unknown'>('unknown')
  const [showGlossary, setShowGlossary] = useState(false)

  const aBase = (countryIso3: string | null = null) => ({
    year: aYear, sector: aSector,
    country_iso3: countryIso3,
    top_n: 20,
  })

  // Resolve a user-typed country name or ISO3 to an ISO3 code using available data
  function resolveIso3(raw: string, result: AnalyzeResponse): string | null {
    const q = raw.trim().toUpperCase()
    if (!q) return null
    if (result.full_rank_map[q] !== undefined) return q   // exact ISO3 match
    const nameMatch = result.ranked.find(r =>
      r.country_name.toUpperCase().includes(q) || r.country_iso3 === q)
    if (nameMatch) return nameMatch.country_iso3
    return q.length === 3 ? q : null   // fallback: try as ISO3 (may fail at backend)
  }

  async function handleAnalyze(e: React.FormEvent) {
    e.preventDefault()
    if (!aPrompt.trim()) return
    setALoading(true); setAError(null); setAResult(null); setARefine(''); setACompare(new Set()); setACountry('')
    try {
      // First call: get rankings + importance scores (no country focus yet)
      const { data } = await axios.post<AnalyzeResponse>('/analyze', { ...aBase(), prompt: aPrompt })

      // If user specified a focus country, resolve it then fetch pro/con in a second call
      if (aFocusRaw.trim()) {
        const iso3 = resolveIso3(aFocusRaw, data)
        if (iso3) {
          setACountry(iso3)
          const { data: data2 } = await axios.post<AnalyzeResponse>('/analyze', {
            ...aBase(iso3), prompt: '', force_scores: data.importance_scores,
          })
          setAResult(data2)
          return
        }
      }
      setAResult(data)
    } catch (err: unknown) {
      setAError(axios.isAxiosError(err) ? String(err.response?.data?.detail ?? err.message) : 'Unknown error')
    } finally { setALoading(false) }
  }

  // Focus on a specific country after results are already loaded (no LLM re-run)
  async function handleFocusCountry(iso3: string) {
    if (!aResult) return
    const upper = iso3.toUpperCase()
    setACountry(upper); setAFocusRaw(upper)
    setALoading(true); setAError(null)
    try {
      const { data } = await axios.post<AnalyzeResponse>('/analyze', {
        ...aBase(upper), prompt: '', force_scores: aResult.importance_scores,
      })
      setAResult(data)
    } catch (err: unknown) {
      setAError(axios.isAxiosError(err) ? String(err.response?.data?.detail ?? err.message) : 'Unknown error')
    } finally { setALoading(false) }
  }

  async function handleAdjustScore(dim: string, delta: number) {
    if (!aResult) return
    const updated = { ...aResult.importance_scores, [dim]: Math.max(1, Math.min(10, aResult.importance_scores[dim] + delta)) }
    setALoading(true); setAError(null)
    try {
      const { data } = await axios.post<AnalyzeResponse>('/analyze', {
        ...aBase(aCountry || null), prompt: '', force_scores: updated,
      })
      setAResult(data)
    } catch (err: unknown) {
      setAError(axios.isAxiosError(err) ? String(err.response?.data?.detail ?? err.message) : 'Unknown error')
    } finally { setALoading(false) }
  }

  async function handleRefine(e: React.FormEvent) {
    e.preventDefault()
    if (!aRefine.trim() || !aResult) return
    setALoading(true); setAError(null)
    try {
      const { data } = await axios.post<AnalyzeResponse>('/analyze', {
        ...aBase(aCountry || null), prompt: aRefine, previous_scores: aResult.importance_scores,
      })
      setAResult(data); setARefine('')
    } catch (err: unknown) {
      setAError(axios.isAxiosError(err) ? String(err.response?.data?.detail ?? err.message) : 'Unknown error')
    } finally { setALoading(false) }
  }

  // Query state
  const [query, setQuery]       = useState('')
  const [yearFrom, setYearFrom] = useState(2019)
  const [yearTo, setYearTo]     = useState(2025)
  const [showSql, setShowSql]   = useState(false)
  const [qLoading, setQLoading] = useState(false)
  const [qError, setQError]     = useState<string | null>(null)
  const [qResult, setQResult]   = useState<QueryResponse | null>(null)

  // ── Scoring ───────────────────────────────────────────────────────────────

  const runScore = useCallback(async (year: number, w: Weights) => {
    setSLoading(true); setSError(null)
    try {
      const { data } = await axios.post<ScoreResponse>('/score', { year, weights: w, top_n: 58 })
      setSResult(data)
    } catch (err: unknown) {
      setSError(axios.isAxiosError(err)
        ? String(err.response?.data?.detail ?? err.message) : 'Unknown error')
    } finally {
      setSLoading(false)
    }
  }, [])

  // Auto-load on mount
  useEffect(() => {
    runScore(2024, DEFAULT_WEIGHTS)
    axios.get<{ data_source: string }>('/health').then(r => {
      const src = r.data.data_source as 'databricks' | 'parquet' | 'unknown'
      setDataSource(src)
    }).catch(() => {})
  }, [runScore])

  // Animation
  useEffect(() => {
    if (playing) {
      playRef.current = setInterval(() => {
        setAnimYear(y => {
          const next = y >= 2025 ? 2019 : y + 1
          runScore(next, weights)
          return next
        })
      }, 1800)
    } else {
      if (playRef.current) clearInterval(playRef.current)
    }
    return () => { if (playRef.current) clearInterval(playRef.current) }
  }, [playing, weights, runScore])

  async function handleQuery(e: React.FormEvent) {
    e.preventDefault()
    if (!query.trim()) return
    setQLoading(true); setQError(null); setQResult(null)
    try {
      const { data } = await axios.post<QueryResponse>('/query', {
        query, year_from: yearFrom, year_to: yearTo, limit: 50,
      })
      setQResult(data)
    } catch (err: unknown) {
      setQError(axios.isAxiosError(err)
        ? String(err.response?.data?.detail ?? err.message) : 'Unknown error')
    } finally {
      setQLoading(false)
    }
  }

  // ── Derived ───────────────────────────────────────────────────────────────

  const rows = sResult?.rows ?? []
  const maxScore = Math.max(...rows.map(r => r.overlooked_score), 0.01)

  const sIsoMap = Object.fromEntries(rows.map(r => [r.country_iso3, r]))

  const bubbleData = rows
    .filter(r => r.pin != null && r.coverage_ratio != null)
    .map(r => ({
      ...r,
      x: +(r.coverage_ratio! * 100).toFixed(1),
      y: +Math.log10(r.pin! / 1e6 + 0.01).toFixed(3),
      _z: Math.sqrt((r.requirements_usd ?? 1e6) / 1e6),
    }))

  const qIsoMap: Record<string, number> = {}
  qResult?.rows.forEach(r => {
    if (r.country_iso3 && r.gap_score != null)
      qIsoMap[r.country_iso3 as string] = r.gap_score as number
  })

  const displayYear = playing ? animYear : scoreYear

  // ── Render ────────────────────────────────────────────────────────────────

  return (
    <div className="min-h-screen bg-gray-950 text-gray-100 flex flex-col">

      {/* Header */}
      <header className="border-b border-gray-800 px-8 py-4 flex items-center gap-3">
        <div>
          <h1 className="text-lg font-bold text-white leading-none tracking-tight">Geo-Insight</h1>
          <p className="text-xs text-gray-500 mt-0.5">Which humanitarian crises are most overlooked?</p>
        </div>
        <div className="ml-auto flex items-center gap-2 text-xs text-gray-500">
          <span className="bg-gray-900 border border-gray-800 px-2 py-1 rounded font-mono">Llama 4 Maverick</span>
          {dataSource === 'databricks' ? (
            <span className="bg-green-950 border border-green-800 text-green-400 px-2 py-1 rounded font-mono flex items-center gap-1">
              <span className="w-1.5 h-1.5 rounded-full bg-green-400 inline-block" />
              Live · Databricks
            </span>
          ) : dataSource === 'parquet' ? (
            <span className="bg-amber-950 border border-amber-800 text-amber-400 px-2 py-1 rounded font-mono flex items-center gap-1"
              title="Databricks unreachable — using local parquet snapshots">
              <span className="w-1.5 h-1.5 rounded-full bg-amber-400 inline-block" />
              Parquet snapshot
            </span>
          ) : (
            <span className="bg-gray-900 border border-gray-800 px-2 py-1 rounded font-mono">Databricks</span>
          )}
          <button
            onClick={() => axios.post('/refresh').then(r => setDataSource((r.data as {data_source: string}).data_source as 'databricks'|'parquet'|'unknown')).catch(()=>{})}
            className="bg-gray-900 border border-gray-800 hover:border-gray-600 px-2 py-1 rounded font-mono hover:text-gray-300 transition-colors"
            title="Reload data from Databricks">
            ↺
          </button>
        </div>
      </header>

      {/* Tabs */}
      <div className="border-b border-gray-800 px-8 flex gap-1 pt-2">
        {([
          ['score',   '📊 Gap Scoring'],
          ['analyze', '🎯 Prioritize'],
          ['query',   '🔍 SQL Query'],
        ] as const).map(([t, label]) => (
          <button key={t} onClick={() => setTab(t)}
            className={`px-4 py-2 text-sm font-medium rounded-t-lg transition-colors ${
              tab === t ? 'bg-gray-900 text-white border border-b-gray-900 border-gray-700'
                        : 'text-gray-500 hover:text-gray-300'}`}>
            {label}
          </button>
        ))}
      </div>

      <main className="flex-1 px-6 py-6 max-w-7xl mx-auto w-full">

        {/* ═══ SCORE TAB ═══════════════════════════════════════════════════ */}
        {tab === 'score' && (
          <div className="flex flex-col gap-8">

            {/* Controls */}
            <div className="bg-gray-900 border border-gray-800 rounded-2xl p-5 flex flex-wrap items-end gap-6">

              {/* Year selector + play */}
              <div className="flex flex-col gap-1.5">
                <span className="text-xs text-gray-500 font-medium uppercase tracking-wide">Year</span>
                <div className="flex items-center gap-2">
                  <div className="flex rounded-lg overflow-hidden border border-gray-700">
                    {YEARS.map(y => (
                      <button key={y} onClick={() => { setPlaying(false); setScoreYear(y); runScore(y, weights) }}
                        className={`px-3 py-1.5 text-xs font-mono transition-colors ${
                          displayYear === y
                            ? 'bg-blue-600 text-white'
                            : 'bg-gray-800 text-gray-400 hover:text-gray-200'}`}>
                        {y}
                      </button>
                    ))}
                  </div>
                  <button
                    onClick={() => { setAnimYear(scoreYear); setPlaying(p => !p) }}
                    className={`px-3 py-1.5 rounded-lg text-xs font-medium border transition-colors ${
                      playing ? 'border-red-700 text-red-400 bg-red-950' : 'border-gray-700 text-gray-300 hover:border-blue-600 hover:text-blue-400'}`}>
                    {playing ? '⏹ Stop' : '▶ Animate'}
                  </button>
                </div>
              </div>

              {/* Weight summary */}
              <div className="flex flex-col gap-1.5 flex-1 min-w-52">
                <span className="text-xs text-gray-500 font-medium uppercase tracking-wide">Active weights</span>
                <div className="flex gap-3 text-xs font-mono">
                  {(['scale','gap','structural','trend'] as (keyof Weights)[]).map(k => (
                    <span key={k} className="flex flex-col items-center">
                      <span className="text-blue-400">{weights[k].toFixed(2)}</span>
                      <span className="text-gray-600">{k}</span>
                    </span>
                  ))}
                </div>
              </div>

              <div className="flex gap-2">
                <button onClick={() => runScore(scoreYear, weights)} disabled={sLoading}
                  className="bg-blue-600 hover:bg-blue-500 disabled:opacity-40 text-white px-5 py-2 rounded-lg text-sm font-medium transition-colors">
                  {sLoading ? '…' : 'Recalculate'}
                </button>
                {sResult && (
                  <button onClick={() => exportCsv(sResult.rows, sResult.year)}
                    className="bg-gray-800 hover:bg-gray-700 border border-gray-700 text-gray-300 px-4 py-2 rounded-lg text-sm font-medium transition-colors flex items-center gap-1.5">
                    ↓ CSV
                  </button>
                )}
              </div>

              {/* Formula */}
              <div className="w-full text-xs text-gray-600 font-mono bg-gray-800/50 rounded-lg px-4 py-2 leading-relaxed">
                <span className="text-gray-300">overlooked</span> = (
                <span className="text-orange-400">w_scale·scale</span> +
                <span className="text-blue-400"> w_gap·gap</span> +
                <span className="text-purple-400"> w_structural·structural</span> +
                <span className="text-yellow-600"> w_trend·trend</span>) ×
                <span className="text-green-600"> confidence</span>
                <span className="ml-3 text-gray-700">· validated by Borda ensemble of 4 rankings</span>
              </div>
            </div>

            {sError && (
              <div className="bg-red-950 border border-red-800 text-red-300 rounded-xl px-4 py-3 text-sm">{sError}</div>
            )}

            {sLoading && !sResult && (
              <div className="flex items-center justify-center py-24 text-gray-600">
                <div className="text-sm">Loading {displayYear} data…</div>
              </div>
            )}

            {sResult && (
              <>
                {/* Stat cards */}
                <div className="grid grid-cols-4 gap-3">
                  {[
                    { label: 'Countries assessed', value: sResult.row_count, color: 'text-white' },
                    { label: 'Robustly overlooked', value: sResult.meta.robust_count, color: 'text-purple-400',
                      sub: 'top quartile in all 4 rankings' },
                    { label: '#1 most overlooked', value: rows[0]?.country_name ?? '—', color: 'text-red-400', small: true },
                    { label: 'PC1+PC2 variance', color: 'text-blue-400',
                      value: (Object.values(sResult.meta.pca_explained_var).slice(0,2).reduce((a,b)=>a+b,0)*100).toFixed(0)+'%' },
                  ].map(c => (
                    <div key={c.label} className="bg-gray-900 border border-gray-800 rounded-xl p-4">
                      <div className={`${c.small ? 'text-lg' : 'text-2xl'} font-bold ${c.color}`}>{c.value}</div>
                      <div className="text-xs text-gray-500 mt-1">{c.label}</div>
                      {c.sub && <div className="text-xs text-gray-700 mt-0.5">{c.sub}</div>}
                    </div>
                  ))}
                </div>

                {/* ── BUBBLE CHART ── */}
                <div className="bg-gray-900 border border-gray-800 rounded-2xl p-6">
                  <div className="flex items-center justify-between mb-1">
                    <p className="text-sm font-semibold text-white">Need vs. Funding Space</p>
                    <span className="text-xs text-gray-600">bubble size = requirements · top-8 labelled</span>
                  </div>
                  <p className="text-xs text-gray-600 mb-4">
                    Crises in the top-left corner have the largest unmet need — high people in need, low funding coverage.
                  </p>
                  <ResponsiveContainer width="100%" height={380}>
                    <ScatterChart margin={{ top: 20, right: 30, bottom: 40, left: 30 }}>
                      <XAxis type="number" dataKey="x" domain={[0, 105]} tick={{ fill: '#6b7280', fontSize: 11 }}>
                        <Label value="% Funded (coverage ratio)" position="insideBottom" offset={-25} fill="#6b7280" fontSize={11} />
                      </XAxis>
                      <YAxis type="number" dataKey="y" domain={[-2, 2]} tick={{ fill: '#6b7280', fontSize: 11 }}
                        tickFormatter={v => `${Math.pow(10,v).toFixed(1)}M`}>
                        <Label value="People in Need (log scale)" angle={-90} position="insideLeft" fill="#6b7280" fontSize={11} offset={10} />
                      </YAxis>
                      <ZAxis type="number" dataKey="_z" range={[40, 900]} />
                      <ReferenceLine x={30} stroke="#374151" strokeDasharray="4 4"
                        label={{ value: '30% threshold', fill: '#4b5563', fontSize: 10, position: 'insideTopRight' }} />
                      <ReTooltip content={<BubbleTooltip />} cursor={{ strokeDasharray: '3 3', stroke: '#374151' }} />
                      <Scatter data={bubbleData} shape={<BubbleDot />} />
                    </ScatterChart>
                  </ResponsiveContainer>
                  <div className="flex items-center gap-6 mt-2 text-xs text-gray-600 justify-center">
                    {[['Steel blue','low gap'],['Amber','medium gap'],['Red','high gap'],['Purple','robustly overlooked']].map(([c,l])=>(
                      <span key={l} className="flex items-center gap-1.5">
                        <span className="w-2.5 h-2.5 rounded-full inline-block" style={{background: c==='Steel blue'?'#3884c8':c==='Amber'?'#f97316':c==='Red'?'#dc2626':'#7c3aed'}}/>
                        {l}
                      </span>
                    ))}
                  </div>
                </div>

                {/* ── WORLD MAP ── */}
                <div className="bg-gray-900 border border-gray-800 rounded-2xl p-6">
                  <p className="text-sm font-semibold text-white mb-1">World Map — Overlooked Score {displayYear}</p>
                  <p className="text-xs text-gray-600 mb-4">Hover over a country for details. Purple = robustly overlooked across all ranking methods.</p>

                  <div className="relative">
                    <ComposableMap projectionConfig={{ scale: 153 }} style={{ background: 'transparent' }}>
                      <Geographies geography={GEO_URL}>
                        {({ geographies }: { geographies: { rsmKey: string; properties: Record<string,unknown> }[] }) =>
                          geographies.map(geo => {
                            const iso  = geo.properties.ISO_A3 as string
                            const row  = sIsoMap[iso]
                            const fill = row ? scoreToColor(row.overlooked_score, row.robust) : '#1f2937'
                            return (
                              <Geography
                                key={geo.rsmKey} geography={geo}
                                fill={fill} stroke="#111827" strokeWidth={0.4}
                                onMouseEnter={(e: MouseEvent) => {
                                  if (!row) return
                                  setHoveredCountry(row)
                                  setTooltipPos({ x: e.clientX, y: e.clientY })
                                }}
                                onMouseMove={(e: MouseEvent) => {
                                  setTooltipPos({ x: e.clientX, y: e.clientY })
                                }}
                                onMouseLeave={() => setHoveredCountry(null)}
                                style={{
                                  default:  { outline: 'none' },
                                  hover:    { fill: row ? '#60a5fa' : '#374151', outline: 'none', cursor: row ? 'pointer' : 'default' },
                                  pressed:  { outline: 'none' },
                                }}
                              />
                            )
                          })
                        }
                      </Geographies>
                    </ComposableMap>
                  </div>

                  {/* Tooltip — fixed to viewport so it's never clipped */}
                  {hoveredCountry && (
                    <div
                      className="pointer-events-none fixed z-50 bg-gray-950 border border-gray-700 rounded-xl p-3 shadow-2xl text-xs w-64"
                      style={{
                        left: tooltipPos.x + 14,
                        top:  tooltipPos.y - 10,
                        transform: tooltipPos.x > window.innerWidth - 280 ? 'translateX(-110%)' : undefined,
                      }}
                    >
                      <div className="flex items-center gap-2 mb-2">
                        <span className="font-semibold text-sm text-white">{hoveredCountry.country_name}</span>
                        {hoveredCountry.robust && (
                          <span className="bg-purple-900 text-purple-300 px-1.5 py-0.5 rounded text-xs">Robust</span>
                        )}
                      </div>
                      <div className="space-y-1 text-gray-400">
                        <div className="flex justify-between">
                          <span>Overlooked score</span>
                          <span className="font-mono" style={{ color: scoreToColor(hoveredCountry.overlooked_score, hoveredCountry.robust) }}>
                            {hoveredCountry.overlooked_score.toFixed(3)}
                          </span>
                        </div>
                        <div className="flex justify-between">
                          <span>Borda rank</span>
                          <span className="text-white font-mono">#{hoveredCountry.borda_rank}</span>
                        </div>
                        <div className="flex justify-between">
                          <span>People in need</span>
                          <span className="text-orange-300">{millions(hoveredCountry.pin)}</span>
                        </div>
                        <div className="flex justify-between">
                          <span>Funded</span>
                          <span className="text-blue-300">{pct(hoveredCountry.coverage_ratio)}</span>
                        </div>
                        {(hoveredCountry.years_underfunded ?? 0) > 0 && (
                          <div className="flex justify-between">
                            <span>Yrs &lt;30%</span>
                            <span className="text-purple-300">{hoveredCountry.years_underfunded}/{hoveredCountry.n_coverage_years}</span>
                          </div>
                        )}
                      </div>
                      <div className="mt-2 pt-2 border-t border-gray-800 text-gray-500 leading-relaxed">
                        {hoveredCountry.explanation}
                      </div>
                    </div>
                  )}

                  {/* Gradient legend */}
                  <div className="flex items-center gap-3 mt-3 justify-center">
                    <span className="text-xs text-gray-600">Low gap</span>
                    <div className="h-2.5 w-48 rounded-full" style={{
                      background: 'linear-gradient(to right, #3884c8, #fb9216, #dc2626)'
                    }} />
                    <span className="text-xs text-gray-600">High gap</span>
                    <span className="text-xs text-gray-600 ml-4 flex items-center gap-1.5">
                      <span className="w-3 h-3 rounded-sm inline-block bg-purple-600" /> Robustly overlooked
                    </span>
                    <span className="text-xs text-gray-600 flex items-center gap-1.5">
                      <span className="w-3 h-3 rounded-sm inline-block bg-gray-800" /> No crisis data
                    </span>
                  </div>
                </div>

                {/* ── PCA SCATTER ── */}
                <div className="bg-gray-900 border border-gray-800 rounded-2xl p-6">
                  <p className="text-sm font-semibold text-white mb-1">Country Similarity Map (PCA)</p>
                  <p className="text-xs text-gray-600 mb-4">
                    Countries near each other share similar crisis profiles across all dimensions.
                    PC1 captures overall severity + gap · PC2 separates structural from acute neglect.
                  </p>
                  <ResponsiveContainer width="100%" height={320}>
                    <ScatterChart margin={{ top: 10, right: 20, bottom: 30, left: 20 }}>
                      <XAxis type="number" dataKey="pc1" tick={{ fill: '#4b5563', fontSize: 10 }}>
                        <Label value="PC1 — severity + funding gap" position="insideBottom" offset={-20} fill="#4b5563" fontSize={10} />
                      </XAxis>
                      <YAxis type="number" dataKey="pc2" tick={{ fill: '#4b5563', fontSize: 10 }}>
                        <Label value="PC2 — structural vs acute" angle={-90} position="insideLeft" fill="#4b5563" fontSize={10} />
                      </YAxis>
                      <ZAxis range={[40, 40]} />
                      <ReTooltip content={<PcaTooltip />} cursor={{ strokeDasharray: '3 3', stroke: '#374151' }} />
                      <Scatter data={rows} shape={<PcaDot />} />
                    </ScatterChart>
                  </ResponsiveContainer>
                </div>

                {/* ── RANKINGS TABLE ── */}
                <div className="bg-gray-900 border border-gray-800 rounded-2xl overflow-hidden">
                  <div className="px-6 py-4 border-b border-gray-800">
                    <p className="text-sm font-semibold text-white">Full Rankings — {displayYear}</p>
                    <p className="text-xs text-gray-600 mt-0.5">
                      Bar = gap component (colour) + structural bonus (purple). Borda validates across 4 independent methods.
                    </p>
                  </div>
                  <div className="overflow-x-auto">
                    <table className="w-full text-sm">
                      <thead className="text-gray-500 text-xs uppercase border-b border-gray-800">
                        <tr>
                          <th className="px-4 py-3 text-left w-10">#</th>
                          <th className="px-4 py-3 text-left">Country</th>
                          <th className="px-4 py-3 text-left min-w-[160px]">Score</th>
                          <th className="px-4 py-3 text-right">Borda</th>
                          <th className="px-4 py-3 text-right">PIN</th>
                          <th className="px-4 py-3 text-right">Funded</th>
                          <th className="px-4 py-3 text-right">Yrs &lt;30%</th>
                          <th className="px-4 py-3 text-left min-w-[280px]">Notes</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-gray-800/60">
                        {rows.map((r, i) => (
                          <tr key={r.country_iso3}
                            className={`transition-colors hover:bg-gray-800/50 ${
                              r.robust ? 'bg-purple-950/20' : i < 3 ? 'bg-red-950/10' : ''}`}>
                            <td className="px-4 py-3 font-mono text-xs text-gray-500">
                              {i < 3
                                ? <span className="text-base">{['🥇','🥈','🥉'][i]}</span>
                                : r.rank}
                            </td>
                            <td className="px-4 py-3">
                              <div className="flex items-center gap-2 flex-wrap">
                                <span className="text-gray-100 text-xs font-medium">{r.country_name}</span>
                                {r.robust && <span className="text-xs bg-purple-900/70 text-purple-300 px-1.5 py-0.5 rounded-full">★ Robust</span>}
                                {!r.data_complete && <span className="text-xs text-yellow-700 border border-yellow-900 px-1.5 py-0.5 rounded-full">Partial</span>}
                              </div>
                              <div className="text-xs text-gray-600 mt-0.5">{r.region_name}</div>
                            </td>
                            <td className="px-4 py-3">
                              <ScoreBar row={r} maxScore={maxScore} />
                            </td>
                            <td className="px-4 py-3 text-right font-mono text-xs text-gray-400">#{r.borda_rank}</td>
                            <td className="px-4 py-3 text-right font-mono text-xs text-gray-300">{millions(r.pin)}</td>
                            <td className="px-4 py-3 text-right font-mono text-xs">
                              <span style={{ color: r.coverage_ratio != null ? scoreToColor(1 - r.coverage_ratio) : '#6b7280' }}>
                                {pct(r.coverage_ratio)}
                              </span>
                            </td>
                            <td className="px-4 py-3 text-right font-mono text-xs text-gray-400">
                              {r.years_underfunded != null ? `${r.years_underfunded}/${r.n_coverage_years}` : '—'}
                            </td>
                            <td className="px-4 py-3 text-xs text-gray-500 leading-relaxed max-w-xs">{r.explanation}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              </>
            )}
          </div>
        )}

        {/* ═══ ANALYZE TAB ════════════════════════════════════════════════ */}
        {tab === 'analyze' && (() => {
          const pc = aResult?.pro_con && !('error' in aResult.pro_con)
            ? aResult.pro_con as ProConResult : null
          const topDims = aResult
            ? Object.entries(aResult.importance_scores)
                .filter(([, s]) => s >= 5)
                .sort((a, b) => b[1] - a[1])
                .slice(0, 4)
                .map(([k]) => k)
            : []
          const dimLabel = (k: string) =>
            k.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase())

          const DIM_COLORS: Record<string, string> = {
            need_scale: '#f97316', funding_gap: '#ef4444', structural_neglect: '#8b5cf6',
            trend_worsening: '#ec4899', targeting_gap: '#f59e0b', water_stress: '#06b6d4',
            food_insecurity_risk: '#10b981', displacement_risk: '#3b82f6',
            health_fragility: '#e11d48', climate_vulnerability: '#84cc16',
            governance_fragility: '#a855f7', disaster_risk: '#14b8a6',
          }

          const sortedRanked = aResult
            ? [...aResult.ranked].sort((a, b) => {
                const av = (a[aSortBy] as number) ?? 0
                const bv = (b[aSortBy] as number) ?? 0
                return aSortDir * (av < bv ? -1 : av > bv ? 1 : 0)
              })
            : []

          const thSort = (key: string, label: string) => (
            <th key={key}
              className="px-3 py-2.5 text-right cursor-pointer select-none hover:text-gray-300 transition-colors font-normal"
              onClick={() => {
                if (aSortBy === key) setASortDir(d => (d === 1 ? -1 : 1) as 1|-1)
                else { setASortBy(key); setASortDir(-1) }
              }}>
              {label.split(' ').slice(0,2).join('\u00a0')}
              <span className="ml-0.5 text-gray-600">{aSortBy === key ? (aSortDir === -1 ? '↓' : '↑') : '↕'}</span>
            </th>
          )

          return (
          <div className="flex flex-col gap-5 max-w-5xl mx-auto w-full">

            {/* ── Prompt card ─────────────────────────────────────────────── */}
            <div className="bg-gray-900 border border-gray-800 rounded-2xl p-5 flex flex-col gap-4">
              <form onSubmit={handleAnalyze} className="flex gap-3 items-start">
                <textarea
                  className="flex-1 bg-gray-800 border border-gray-700 rounded-xl px-4 py-3 text-gray-100 placeholder-gray-500 focus:outline-none focus:border-blue-500 resize-none text-sm leading-relaxed"
                  rows={2}
                  placeholder='Describe your mandate — e.g. "We focus on water scarcity in arid regions with chronic underfunding"'
                  value={aPrompt}
                  onChange={e => setAPrompt(e.target.value)}
                />
                <div className="flex flex-col gap-2 shrink-0">
                  <div className="flex gap-2">
                    <select className="bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-gray-100 text-sm flex-1"
                      value={aYear} onChange={e => setAYear(Number(e.target.value))}>
                      {YEARS.map(y => <option key={y}>{y}</option>)}
                    </select>
                    <select className="bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-gray-100 text-sm flex-1"
                      value={aSector} onChange={e => setASector(e.target.value)}>
                      {SECTOR_OPTIONS.map(s => <option key={s.value} value={s.value}>{s.label}</option>)}
                    </select>
                  </div>
                  <input
                    className="bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-gray-100 text-sm placeholder-gray-600 focus:outline-none focus:border-blue-500"
                    placeholder="Focus country, e.g. Brazil or BRA (optional)"
                    value={aFocusRaw}
                    onChange={e => setAFocusRaw(e.target.value)}
                  />
                  <button type="submit" disabled={aLoading || !aPrompt.trim()}
                    className="bg-blue-600 hover:bg-blue-500 disabled:opacity-40 text-white px-4 py-2 rounded-lg text-sm font-medium transition-colors">
                    {aLoading ? '…' : 'Analyze →'}
                  </button>
                </div>
              </form>

              {/* Example chips */}
              <div className="flex flex-wrap gap-2">
                {[
                  "Water scarcity in chronically neglected conflict zones",
                  "Food insecurity at massive scale regardless of funding trend",
                  "Structural neglect and governance fragility",
                  "Climate-vulnerable displacement crises",
                ].map(ex => (
                  <button key={ex} onClick={() => setAPrompt(ex)}
                    className="text-xs text-gray-500 bg-gray-800/60 border border-gray-700 hover:border-gray-500 hover:text-gray-300 px-3 py-1 rounded-full transition-colors">
                    {ex}
                  </button>
                ))}
              </div>
            </div>

            {aError && <div className="bg-red-950 border border-red-800 text-red-300 rounded-xl px-4 py-3 text-sm">{aError}</div>}
            {aLoading && <div className="flex items-center justify-center py-16 text-gray-600 text-sm">Scoring all countries against your priorities…</div>}

            {aResult && (
              <>
                {/* ── Priority tags + refine ───────────────────────────── */}
                <div className="bg-gray-900 border border-gray-800 rounded-2xl p-5">
                  <div className="flex items-center justify-between mb-3">
                    <p className="text-xs text-gray-500 uppercase tracking-wide font-medium">How I understood your priorities</p>
                    <span className="text-xs text-gray-600 font-mono">score = Σ(weight × dim) &nbsp;·&nbsp; weight = imp²/Σ(imp²)</span>
                  </div>
                  <div className="flex flex-wrap gap-2 mb-4">
                    {Object.entries(aResult.importance_scores)
                      .sort((a, b) => b[1] - a[1])
                      .map(([k, s]) => {
                        const color = s >= 8 ? 'bg-red-900/60 border-red-800 text-red-300'
                          : s >= 6 ? 'bg-orange-900/50 border-orange-800 text-orange-300'
                          : s >= 4 ? 'bg-blue-900/40 border-blue-800 text-blue-300'
                          : 'bg-gray-800 border-gray-700 text-gray-500'
                        const meta = DIM_META[k]
                        return (
                          <span key={k}
                            title={meta ? `${meta.label} — ${meta.desc}\nSource: ${meta.source}` : k}
                            className={`inline-flex items-center gap-1 pl-2.5 pr-1 py-0.5 rounded-full border text-xs font-medium cursor-help ${color}`}>
                            {dimLabel(k)}
                            <span className="font-mono font-bold mx-1">{s}/10</span>
                            <span className="inline-flex gap-0.5">
                              <button
                                disabled={aLoading || s >= 10}
                                onClick={() => handleAdjustScore(k, 1)}
                                className="w-5 h-5 flex items-center justify-center rounded-full hover:bg-white/10 disabled:opacity-30 transition-colors font-bold leading-none">
                                +
                              </button>
                              <button
                                disabled={aLoading || s <= 1}
                                onClick={() => handleAdjustScore(k, -1)}
                                className="w-5 h-5 flex items-center justify-center rounded-full hover:bg-white/10 disabled:opacity-30 transition-colors font-bold leading-none">
                                −
                              </button>
                            </span>
                          </span>
                        )
                      })}
                  </div>

                  {/* ── Weight distribution bar ──────────────────────────── */}
                  {(() => {
                    const sorted = Object.entries(aResult.weights).sort((a, b) => b[1] - a[1])
                    const total = sorted.reduce((s, [, w]) => s + w, 0) || 1
                    return (
                      <div className="mb-4 p-3 bg-gray-800/40 rounded-xl">
                        <div className="flex items-center justify-between mb-2">
                          <p className="text-xs text-gray-600">How weight distributes across your 12 dimensions</p>
                          <p className="text-xs text-gray-700 font-mono">weight = imp² / Σ(imp²)</p>
                        </div>
                        <div className="flex h-3 rounded-full overflow-hidden gap-px mb-2">
                          {sorted.map(([k, w]) => (
                            <div key={k}
                              title={`${DIM_META[k]?.label ?? k}: ${(w / total * 100).toFixed(1)}% weight\n${DIM_META[k]?.desc ?? ''}`}
                              style={{ width: `${w / total * 100}%`, background: DIM_COLORS[k] ?? '#6b7280', minWidth: 1 }} />
                          ))}
                        </div>
                        <div className="flex flex-wrap gap-x-3 gap-y-0.5 mb-2">
                          {sorted.filter(([, w]) => w / total >= 0.04).map(([k, w]) => (
                            <span key={k} className="text-xs text-gray-500 flex items-center gap-1"
                              title={DIM_META[k]?.desc}>
                              <span className="w-2 h-2 rounded-sm shrink-0 inline-block"
                                style={{ background: DIM_COLORS[k] ?? '#6b7280' }} />
                              {dimLabel(k)} <span className="font-mono text-gray-600">{(w / total * 100).toFixed(0)}%</span>
                            </span>
                          ))}
                        </div>
                        {/* Math explainer */}
                        <div className="border-t border-gray-700/50 pt-2 mt-1 text-xs text-gray-700 leading-relaxed space-y-0.5">
                          <p><span className="text-gray-500 font-medium">Importance (1–10)</span> — your stated priority. Squaring before normalising means a score of 10 carries <span className="font-mono text-gray-500">10²=100</span> times the weight of a score of 1, so your top picks truly dominate.</p>
                          <p><span className="text-gray-500 font-medium">Dimension score (0–1)</span> — each country's raw value, where 1 = most critical globally on that metric.</p>
                          <p><span className="text-gray-500 font-medium">Contribution</span> = weight × dimension score. The final MCDA score is the sum of all contributions.</p>
                        </div>
                      </div>
                    )
                  })()}

                  {/* ── Dimension glossary ─────────────────────────────── */}
                  <div className="mb-3">
                    <button onClick={() => setShowGlossary(v => !v)}
                      className="text-xs text-gray-600 hover:text-gray-400 flex items-center gap-1 transition-colors">
                      <span>{showGlossary ? '▾' : '▸'}</span>
                      What do these 12 metrics measure?
                    </button>
                    {showGlossary && (
                      <div className="mt-2 rounded-xl overflow-hidden border border-gray-800">
                        <table className="w-full text-xs">
                          <thead className="bg-gray-800/60">
                            <tr>
                              <th className="px-3 py-2 text-left text-gray-500 font-medium w-8"></th>
                              <th className="px-3 py-2 text-left text-gray-500 font-medium">Metric</th>
                              <th className="px-3 py-2 text-left text-gray-500 font-medium">What it measures</th>
                              <th className="px-3 py-2 text-left text-gray-500 font-medium hidden sm:table-cell">Source</th>
                            </tr>
                          </thead>
                          <tbody className="divide-y divide-gray-800/60">
                            {Object.entries(DIM_META).map(([k, m]) => (
                              <tr key={k} className="hover:bg-gray-800/30">
                                <td className="px-3 py-2">
                                  <span className="w-2.5 h-2.5 rounded-sm inline-block"
                                    style={{ background: DIM_COLORS[k] ?? '#6b7280' }} />
                                </td>
                                <td className="px-3 py-2">
                                  <div className="font-medium text-gray-300">{m.label}</div>
                                  <div className="text-gray-600">{m.group}</div>
                                </td>
                                <td className="px-3 py-2 text-gray-500 leading-relaxed max-w-xs">{m.desc}</td>
                                <td className="px-3 py-2 text-gray-700 hidden sm:table-cell">{m.source}</td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    )}
                  </div>

                  <form onSubmit={handleRefine} className="flex gap-2">
                    <input
                      className="flex-1 bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-sm text-gray-100 placeholder-gray-600 focus:outline-none focus:border-blue-500"
                      placeholder="Not right? Refine: e.g. make water more important, I don't care about governance"
                      value={aRefine}
                      onChange={e => setARefine(e.target.value)}
                    />
                    <button type="submit" disabled={aLoading || !aRefine.trim()}
                      className="bg-gray-700 hover:bg-gray-600 disabled:opacity-40 text-gray-200 px-4 py-2 rounded-lg text-sm font-medium whitespace-nowrap">
                      {aLoading ? '…' : 'Refine ↩'}
                    </button>
                  </form>
                </div>

                {/* ── Inverse query + compare controls ─────────────────── */}
                <div className="flex gap-3 items-center">
                  <input
                    className="flex-1 bg-gray-900 border border-gray-800 rounded-lg px-3 py-2 text-sm text-gray-100 placeholder-gray-600 focus:outline-none focus:border-blue-600"
                    placeholder="Find a country — type name or ISO3 to see where it ranks…"
                    value={aSearch}
                    onChange={e => setASearch(e.target.value)}
                  />
                  {aCompare.size > 0 && (
                    <button onClick={() => setACompare(new Set())}
                      className="text-xs text-gray-500 hover:text-gray-300 border border-gray-700 rounded-lg px-3 py-2 whitespace-nowrap">
                      Clear compare ({aCompare.size})
                    </button>
                  )}
                </div>

                {/* Inverse-query answer: country not in top-20 */}
                {(() => {
                  const q = aSearch.trim().toUpperCase()
                  if (!q || q.length < 2) return null
                  const inTop = aResult.ranked.some(r =>
                    r.country_iso3.toUpperCase().includes(q) ||
                    r.country_name.toUpperCase().includes(q))
                  if (inTop) return null
                  const match = Object.entries(aResult.full_rank_map).find(
                    ([iso]) => iso.toUpperCase().includes(q)
                  )
                  if (!match) return (
                    <div className="text-xs text-gray-600 px-1">No country found matching "{aSearch}".</div>
                  )
                  const [iso, rank] = match
                  const name = aResult.ranked.find(r => r.country_iso3 === iso)?.country_name ?? iso
                  return (
                    <div className="bg-gray-900 border border-gray-700 rounded-xl px-4 py-3 text-sm flex items-center justify-between">
                      <span className="text-gray-300">
                        <span className="font-semibold text-white">{name}</span> ranks
                        <span className="font-mono text-amber-400 mx-1">#{rank}</span>
                        overall — outside the top 20 for your current priorities.
                      </span>
                      <button onClick={() => handleFocusCountry(iso)}
                        className="text-xs text-blue-400 hover:text-blue-300 ml-4 whitespace-nowrap">
                        Analyze anyway →
                      </button>
                    </div>
                  )
                })()}

                {/* ── Country not in dataset notice ─────────────────────── */}
                {aFocusRaw.trim() && !aLoading && (() => {
                  // Backend returned an explicit error for this country
                  const proConError = aResult.pro_con && 'error' in aResult.pro_con
                    ? (aResult.pro_con as { error: string }).error : null
                  // User typed something we couldn't resolve to an ISO3 in the dataset
                  const notResolved = !aCountry && aFocusRaw.trim()
                  if (!proConError && !notResolved) return null
                  const label = aFocusRaw.trim()
                  return (
                    <div className="bg-gray-900 border border-amber-900/60 rounded-2xl px-5 py-4 flex items-start gap-3">
                      <span className="text-amber-500 text-lg shrink-0">⚠</span>
                      <div>
                        <p className="text-sm font-semibold text-amber-300 mb-1">
                          "{label}" is not in the crisis dataset
                        </p>
                        <p className="text-xs text-gray-400 leading-relaxed">
                          {proConError
                            ? proConError
                            : `Could not match "${label}" to a country with active OCHA/FTS data.`}
                          {' '}This tool covers countries with active HNO or HRP plans tracked by the Financial Tracking Service.
                          Countries without an active humanitarian response plan (e.g. middle-income countries, non-crisis contexts)
                          are not included.
                        </p>
                        <p className="text-xs text-gray-600 mt-1.5">
                          Try searching the ranked list or use the find-a-country bar above to see which countries are covered.
                        </p>
                      </div>
                    </div>
                  )
                })()}

                {/* ── Standalone Country Analysis card ──────────────────── */}
                {pc && (() => {
                  const dims = Object.entries(aResult.weights)
                    .map(([k, w]) => {
                      const val = pc.dim_scores?.[k] ?? null
                      const importance = aResult.importance_scores[k] ?? 1
                      return { k, w, val, importance, contrib: val != null ? w * val : 0 }
                    })
                    .filter(d => d.w > 0.005)
                    .sort((a, b) => b.contrib - a.contrib)
                  const maxC = Math.max(...dims.map(d => d.contrib), 0.001)

                  // Verdict: based on rank percentile among all scored countries
                  const rankPct = pc.mcda_rank / pc.n_countries
                  const verdict = rankPct <= 0.15
                    ? { label: 'Strong case', color: 'bg-green-900/60 border-green-700 text-green-300', dot: 'bg-green-400' }
                    : rankPct <= 0.40
                    ? { label: 'Mixed case', color: 'bg-amber-900/50 border-amber-700 text-amber-300', dot: 'bg-amber-400' }
                    : { label: 'Weak case', color: 'bg-red-900/50 border-red-800 text-red-300', dot: 'bg-red-500' }

                  // Better alternatives: ranked countries above this one (already in top-N list)
                  const betterAlts = aResult.ranked
                    .filter(r => r.mcda_rank < pc.mcda_rank && r.country_iso3 !== pc.country_iso3)
                    .sort((a, b) => a.mcda_rank - b.mcda_rank)
                    .slice(0, 3)

                  return (
                    <div className="bg-gray-900 border border-blue-800 rounded-2xl overflow-hidden">
                      {/* Header */}
                      <div className="px-5 py-3 border-b border-blue-900/50 flex items-center justify-between bg-blue-950/20">
                        <div className="flex items-center gap-3 flex-wrap">
                          <span className="text-sm font-semibold text-white">{pc.country_name}</span>
                          <span className="text-xs text-gray-400">{pc.continent} · {pc.region_name}</span>
                          <span className="font-mono text-xs bg-gray-800 px-2 py-0.5 rounded text-gray-300">
                            #{pc.mcda_rank} of {pc.n_countries}
                          </span>
                          <span className="font-mono text-xs font-bold" style={{ color: scoreToColor(pc.mcda_score) }}>
                            score {pc.mcda_score.toFixed(3)}
                          </span>
                          {/* Verdict badge */}
                          <span className={`inline-flex items-center gap-1.5 text-xs font-semibold px-2.5 py-0.5 rounded-full border ${verdict.color}`}>
                            <span className={`w-1.5 h-1.5 rounded-full ${verdict.dot}`} />
                            {verdict.label}
                          </span>
                        </div>
                        <button onClick={() => { setACountry(''); setAFocusRaw('') }}
                          className="text-xs text-gray-600 hover:text-gray-400 shrink-0">✕ clear</button>
                      </div>

                      <div className="p-5 flex flex-col gap-5">

                        {/* Key metrics row */}
                        <div className="grid grid-cols-4 gap-2">
                          {[
                            { l: 'People in need', v: millions(pc.pin), c: 'text-orange-300' },
                            { l: 'Funded', v: pct(pc.coverage_ratio), c: 'text-blue-300' },
                            { l: 'Requirements', v: `$${millions(pc.requirements_usd)}`, c: 'text-gray-300' },
                            { l: 'Yrs underfunded', v: pc.years_underfunded != null ? String(pc.years_underfunded) : '—', c: 'text-purple-300' },
                          ].map(m => (
                            <div key={m.l} className="bg-gray-800 rounded-lg p-2.5 text-center">
                              <div className={`text-sm font-bold ${m.c}`}>{m.v}</div>
                              <div className="text-xs text-gray-600 mt-0.5">{m.l}</div>
                            </div>
                          ))}
                        </div>

                        {/* Why fund / Why not — split LLM narrative */}
                        <div className="grid grid-cols-2 gap-3">
                          <div className="bg-green-950/20 border border-green-900/40 rounded-xl p-4">
                            <p className="text-xs font-bold text-green-400 mb-2 flex items-center gap-1.5">
                              <span>✓</span> Why you should fund {pc.country_name}
                            </p>
                            <p className="text-xs text-gray-300 leading-relaxed">{pc.why_fund}</p>
                          </div>
                          <div className="bg-red-950/20 border border-red-900/40 rounded-xl p-4">
                            <p className="text-xs font-bold text-red-400 mb-2 flex items-center gap-1.5">
                              <span>✗</span> Why you might not
                            </p>
                            <p className="text-xs text-gray-300 leading-relaxed">{pc.why_not}</p>
                          </div>
                        </div>

                        {/* Better alternatives (when weak/mixed case and we have ranked alternatives) */}
                        {betterAlts.length > 0 && rankPct > 0.15 && (
                          <div className="bg-gray-800/40 rounded-xl px-4 py-3 flex items-center gap-3 flex-wrap">
                            <span className="text-xs text-gray-500 shrink-0">
                              Stronger alternatives for your mandate:
                            </span>
                            {betterAlts.map(r => (
                              <button key={r.country_iso3}
                                onClick={() => handleFocusCountry(r.country_iso3)}
                                className="text-xs bg-gray-700 hover:bg-gray-600 border border-gray-600 rounded-lg px-3 py-1.5 text-gray-200 transition-colors flex items-center gap-1.5">
                                <span className="font-mono text-gray-500">#{r.mcda_rank}</span>
                                {r.country_name}
                              </button>
                            ))}
                          </div>
                        )}

                        {/* Dimension-level detail (collapsible feel via border-top section) */}
                        <div className="pt-3 border-t border-gray-800/60">
                          <p className="text-xs text-gray-500 font-medium mb-3">Dimension detail</p>
                          <div className="grid grid-cols-2 gap-3 mb-4">
                            <div>
                              <p className="text-xs text-green-500 mb-1.5">Strengths on your criteria</p>
                              {pc.pros.length === 0
                                ? <p className="text-xs text-gray-600 italic">No strong signals.</p>
                                : pc.pros.slice(0, 4).map(c => (
                                  <div key={c.dimension} className="mb-2">
                                    <div className="flex items-center gap-1.5">
                                      <span className="text-green-600 shrink-0 text-xs">▲</span>
                                      <span className="text-xs font-medium text-green-400">{c.label}</span>
                                      <span className="text-xs text-gray-700 ml-auto">top {Math.round((1 - c.percentile) * 100)}%</span>
                                    </div>
                                    <p className="text-xs text-gray-500 leading-relaxed pl-3 mt-0.5">{c.narrative}</p>
                                  </div>
                                ))}
                            </div>
                            <div>
                              <p className="text-xs text-amber-600 mb-1.5">Weaknesses on your criteria</p>
                              {pc.cons.length === 0
                                ? <p className="text-xs text-gray-600 italic">No significant signals.</p>
                                : pc.cons.slice(0, 4).map(c => (
                                  <div key={c.dimension} className="mb-2">
                                    <div className="flex items-center gap-1.5">
                                      <span className="text-amber-700 shrink-0 text-xs">▼</span>
                                      <span className="text-xs font-medium text-amber-500">{c.label}</span>
                                      <span className="text-xs text-gray-700 ml-auto">btm {Math.round(c.percentile * 100)}%</span>
                                    </div>
                                    <p className="text-xs text-gray-500 leading-relaxed pl-3 mt-0.5">{c.narrative}</p>
                                    {c.better_country && (
                                      <p className="text-xs text-gray-700 pl-3 mt-0.5">
                                        {c.better_country.country_name} outperforms
                                        {c.better_country.value != null && c.value != null
                                          ? ` (${(c.better_country.value * 10).toFixed(1)} vs ${(c.value * 10).toFixed(1)})`
                                          : ''}.
                                      </p>
                                    )}
                                  </div>
                                ))}
                            </div>
                          </div>

                          {/* Score decomposition */}
                          <p className="text-xs text-gray-600 font-medium mb-1.5">
                            Score breakdown — {pc.country_name} · <span className="font-mono text-white">{pc.mcda_score.toFixed(3)}</span>
                          </p>
                          <div className="flex items-center gap-2 mb-1 px-0.5">
                            <span className="text-xs text-gray-700 w-24 shrink-0">Dimension</span>
                            <span className="text-xs text-gray-700 w-8 shrink-0 text-center">Imp</span>
                            <span className="text-xs text-gray-700 flex-1 text-center">Contribution</span>
                            <span className="text-xs text-gray-700 w-8 shrink-0 text-right">Wt%</span>
                            <span className="text-xs text-gray-700 w-8 shrink-0 text-right">Dim</span>
                            <span className="text-xs text-gray-700 w-8 shrink-0 text-right">Ctb</span>
                          </div>
                          <div className="flex flex-col gap-1.5">
                            {dims.map(d => {
                              const impColor = d.importance >= 8 ? 'text-red-400 bg-red-950/60 border-red-900'
                                : d.importance >= 6 ? 'text-orange-400 bg-orange-950/60 border-orange-900'
                                : d.importance >= 4 ? 'text-blue-400 bg-blue-950/60 border-blue-900'
                                : 'text-gray-600 bg-gray-800 border-gray-700'
                              return (
                                <div key={d.k} className="flex items-center gap-2">
                                  <span className="text-xs text-gray-500 w-24 shrink-0 truncate" title={dimLabel(d.k)}>
                                    {dimLabel(d.k)}
                                  </span>
                                  <span className={`text-xs font-mono font-bold w-8 shrink-0 text-center rounded border px-1 ${impColor}`}>
                                    {d.importance}
                                  </span>
                                  <div className="flex-1 h-1.5 bg-gray-800 rounded-full overflow-hidden">
                                    <div className="h-full rounded-full"
                                      style={{ width: `${(d.contrib / maxC) * 100}%`, background: DIM_COLORS[d.k] ?? '#6b7280' }} />
                                  </div>
                                  <span className="text-xs font-mono text-gray-600 w-8 shrink-0 text-right">{(d.w * 100).toFixed(0)}%</span>
                                  <span className="text-xs font-mono text-gray-600 w-8 shrink-0 text-right">{d.val != null ? (d.val * 10).toFixed(1) : '—'}</span>
                                  <span className="text-xs font-mono text-gray-400 w-8 shrink-0 text-right font-semibold">{d.contrib > 0 ? (d.contrib * 100).toFixed(1) : '—'}</span>
                                </div>
                              )
                            })}
                          </div>
                          <p className="text-xs text-gray-700 mt-1.5">Imp = importance (1–10) · Wt% = derived weight · Dim = raw score ×10 · Ctb = contribution ×100</p>
                        </div>
                      </div>
                    </div>
                  )
                })()}

                {/* ── Ranked table ──────────────────────────────────────── */}
                <div className="bg-gray-900 border border-gray-800 rounded-2xl overflow-hidden">
                  <div className="px-5 py-3 border-b border-gray-800 flex items-center justify-between">
                    <div>
                      <p className="text-sm font-semibold text-white">
                        Rankings — {aResult.year}
                        {aResult.sector !== 'ALL' && (
                          <span className="ml-2 text-xs font-normal text-blue-400 bg-blue-950 px-2 py-0.5 rounded-full">
                            {SECTOR_OPTIONS.find(s => s.value === aResult.sector)?.label ?? aResult.sector}
                          </span>
                        )}
                      </p>
                      <p className="text-xs text-gray-600 mt-0.5">Sort by header · click row to analyze · checkbox to compare</p>
                    </div>
                    {aCompare.size >= 2 && (
                      <span className="text-xs text-blue-400 border border-blue-800 bg-blue-950 px-2 py-1 rounded-lg">
                        {aCompare.size} selected for comparison ↓
                      </span>
                    )}
                  </div>
                  <div className="overflow-x-auto">
                    <table className="w-full text-sm">
                      <thead className="text-gray-500 text-xs border-b border-gray-800">
                        <tr>
                          <th className="px-3 py-2.5 w-8">
                            <input type="checkbox" className="accent-blue-500" title="Select all"
                              checked={aCompare.size === sortedRanked.length}
                              onChange={e => setACompare(e.target.checked
                                ? new Set(sortedRanked.map(r => r.country_iso3)) : new Set())} />
                          </th>
                          <th className="px-3 py-2.5 text-left w-8 cursor-pointer hover:text-gray-300"
                            onClick={() => { setASortBy('mcda_rank'); setASortDir(1) }}>#</th>
                          <th className="px-3 py-2.5 text-left">Country</th>
                          {thSort('mcda_score', 'Score')}
                          {thSort('rank_delta', 'Δ vs prev yr')}
                          {topDims.map(k => thSort(k, dimLabel(k)))}
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-gray-800/40">
                        {sortedRanked
                          .filter(r => {
                            const q = aSearch.trim().toUpperCase()
                            if (!q) return true
                            return r.country_iso3.toUpperCase().includes(q) ||
                                   r.country_name.toUpperCase().includes(q)
                          })
                          .map((r, i) => {
                            const selected = pc?.country_iso3 === r.country_iso3
                            const inCompare = aCompare.has(r.country_iso3)
                            const delta = r.rank_delta as number | null
                            return (
                              <>
                                <tr key={r.country_iso3}
                                  className={`transition-colors ${selected ? 'bg-blue-950/40' : inCompare ? 'bg-purple-950/20' : 'hover:bg-gray-800/40'}`}>
                                  <td className="px-3 py-3">
                                    <input type="checkbox" className="accent-blue-500"
                                      checked={inCompare}
                                      onChange={e => {
                                        const s = new Set(aCompare)
                                        e.target.checked ? s.add(r.country_iso3) : s.delete(r.country_iso3)
                                        setACompare(s)
                                      }} />
                                  </td>
                                  <td className="px-3 py-3 text-xs text-gray-500 font-mono cursor-pointer"
                                    onClick={() => selected ? (setACountry(''), setAFocusRaw('')) : handleFocusCountry(r.country_iso3)}>
                                    {i < 3 && aSortBy === 'mcda_rank' ? ['🥇','🥈','🥉'][i] : r.mcda_rank}
                                  </td>
                                  <td className="px-3 py-3 cursor-pointer"
                                    onClick={() => selected ? (setACountry(''), setAFocusRaw('')) : handleFocusCountry(r.country_iso3)}>
                                    <div className="flex items-center gap-1.5">
                                      <div>
                                        <div className="text-xs font-medium text-gray-100">{r.country_name}</div>
                                        <div className="text-xs text-gray-600">{r.continent}</div>
                                      </div>
                                      {!r.has_vulnerability && (
                                        <span title="Vulnerability scores estimated (no source data)" className="text-gray-700 text-xs border border-gray-800 rounded px-1">~vuln</span>
                                      )}
                                    </div>
                                  </td>
                                  <td className="px-3 py-3 text-right cursor-pointer"
                                    onClick={() => selected ? (setACountry(''), setAFocusRaw('')) : handleFocusCountry(r.country_iso3)}>
                                    <div className="flex items-center justify-end gap-1.5">
                                      <div className="w-14 h-1.5 bg-gray-800 rounded-full overflow-hidden">
                                        <div className="h-full rounded-full" style={{ width:`${r.mcda_score*100}%`, background: scoreToColor(r.mcda_score) }} />
                                      </div>
                                      <span className="text-xs font-mono font-semibold" style={{ color: scoreToColor(r.mcda_score) }}>
                                        {r.mcda_score.toFixed(2)}
                                      </span>
                                    </div>
                                  </td>
                                  <td className="px-3 py-3 text-right text-xs font-mono">
                                    {delta == null ? <span className="text-gray-700">—</span>
                                      : delta > 0 ? <span className="text-green-400">↑{Math.abs(Math.round(delta))}</span>
                                      : delta < 0 ? <span className="text-red-400">↓{Math.abs(Math.round(delta))}</span>
                                      : <span className="text-gray-600">→</span>}
                                  </td>
                                  {topDims.map(k => {
                                    const v = r[k] as number | null
                                    return (
                                      <td key={k} className="px-3 py-3 text-right">
                                        {v != null
                                          ? <span className={`text-xs font-mono ${v >= 0.7 ? 'text-red-400' : v >= 0.4 ? 'text-orange-400' : 'text-gray-500'}`}>{(v*10).toFixed(1)}</span>
                                          : <span className="text-gray-700 text-xs">—</span>}
                                      </td>
                                    )
                                  })}
                                </tr>
                                {selected && (
                                  <tr key={`${r.country_iso3}-loading`}>
                                    <td colSpan={5 + topDims.length} className="px-5 py-2 bg-blue-950/10 border-b border-blue-900/20">
                                      <span className="text-xs text-blue-400">
                                        {aLoading ? 'Fetching analysis…' : 'Analysis shown above ↑'}
                                      </span>
                                    </td>
                                  </tr>
                                )}
                              </>
                            )
                          })}
                      </tbody>
                    </table>
                  </div>
                </div>

                {/* ── Country comparison panel ───────────────────────────── */}
                {aCompare.size >= 2 && (() => {
                  const compared = sortedRanked.filter(r => aCompare.has(r.country_iso3))
                  const allDims = Object.keys(aResult.importance_scores)
                    .sort((a, b) => aResult.importance_scores[b] - aResult.importance_scores[a])
                  return (
                    <div className="bg-gray-900 border border-gray-700 rounded-2xl overflow-hidden">
                      <div className="px-5 py-3 border-b border-gray-800 flex items-center justify-between">
                        <p className="text-sm font-semibold text-white">Side-by-side comparison</p>
                        <button onClick={() => setACompare(new Set())} className="text-xs text-gray-600 hover:text-gray-400">Clear</button>
                      </div>
                      <div className="overflow-x-auto">
                        <table className="w-full text-xs">
                          <thead className="border-b border-gray-800">
                            <tr>
                              <th className="px-4 py-2.5 text-left text-gray-600 font-normal w-40">Dimension</th>
                              {compared.map(r => (
                                <th key={r.country_iso3} className="px-4 py-2.5 text-center font-semibold text-gray-200">
                                  {r.country_name}
                                  <div className="text-gray-600 font-normal">#{r.mcda_rank}</div>
                                </th>
                              ))}
                            </tr>
                          </thead>
                          <tbody className="divide-y divide-gray-800/40">
                            <tr className="bg-gray-800/30">
                              <td className="px-4 py-2 text-gray-500 font-medium">Overall Score</td>
                              {compared.map(r => (
                                <td key={r.country_iso3} className="px-4 py-2 text-center">
                                  <span className="font-mono font-bold text-sm" style={{ color: scoreToColor(r.mcda_score) }}>
                                    {r.mcda_score.toFixed(3)}
                                  </span>
                                </td>
                              ))}
                            </tr>
                            {allDims.map(dim => {
                              const vals = compared.map(r => (r[dim] as number | null) ?? null)
                              const maxVal = Math.max(...vals.filter((v): v is number => v != null))
                              const importance = aResult.importance_scores[dim]
                              return (
                                <tr key={dim} className={importance >= 7 ? 'bg-blue-950/10' : ''}>
                                  <td className="px-4 py-2 text-gray-400">
                                    {dimLabel(dim)}
                                    {importance >= 7 && <span className="ml-1 text-blue-600 text-xs">★</span>}
                                  </td>
                                  {compared.map((r, ci) => {
                                    const v = vals[ci]
                                    const isMax = v != null && v === maxVal
                                    return (
                                      <td key={r.country_iso3} className="px-4 py-2 text-center">
                                        {v != null ? (
                                          <span className={`font-mono ${isMax ? 'text-white font-bold' : 'text-gray-500'}`}>
                                            {(v * 10).toFixed(1)}
                                          </span>
                                        ) : <span className="text-gray-700">—</span>}
                                      </td>
                                    )
                                  })}
                                </tr>
                              )
                            })}
                          </tbody>
                        </table>
                      </div>
                    </div>
                  )
                })()}

                {/* ── Metric leaders (no country pinned) ────────────────── */}
                {!pc && aResult.dimension_leaders.length > 0 && (
                  <div className="bg-gray-900 border border-gray-800 rounded-2xl overflow-hidden">
                    <div className="px-5 py-3 border-b border-gray-800">
                      <p className="text-sm font-semibold text-white">Who leads each metric</p>
                      <p className="text-xs text-gray-600 mt-0.5">Click to expand pro/con for that country</p>
                    </div>
                    <table className="w-full text-xs">
                      <thead className="text-gray-600 border-b border-gray-800">
                        <tr>
                          <th className="px-4 py-2 text-left font-normal">Metric</th>
                          <th className="px-4 py-2 text-left font-normal">Top country</th>
                          <th className="px-4 py-2 text-right font-normal">Score</th>
                          <th className="px-4 py-2 text-right font-normal">Overall rank</th>
                          <th className="px-4 py-2 text-right font-normal">Weight</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-gray-800/40">
                        {[...aResult.dimension_leaders]
                          .sort((a, b) => b.weight - a.weight)
                          .map(d => (
                            <tr key={d.dimension}
                              className="hover:bg-gray-800/40 cursor-pointer transition-colors"
                              onClick={() => setACountry(d.country_iso3)}>
                              <td className="px-4 py-2.5">
                                <span className="text-gray-300 font-medium">{d.label}</span>
                                <span className="text-gray-600 ml-2">{d.group}</span>
                              </td>
                              <td className="px-4 py-2.5 text-gray-200">{d.country_name}</td>
                              <td className="px-4 py-2.5 text-right font-mono text-orange-300">
                                {d.value != null ? (d.value*10).toFixed(1) : '—'}
                              </td>
                              <td className="px-4 py-2.5 text-right text-gray-500 font-mono">#{d.mcda_rank}</td>
                              <td className="px-4 py-2.5 text-right">
                                <span className={`font-mono ${d.weight >= 0.1 ? 'text-blue-400' : 'text-gray-600'}`}>
                                  {(d.weight*100).toFixed(0)}%
                                </span>
                              </td>
                            </tr>
                          ))}
                      </tbody>
                    </table>
                  </div>
                )}
              </>
            )}

            {!aResult && !aLoading && !aError && (
              <div className="mt-16 text-center">
                <p className="text-3xl mb-3">🎯</p>
                <p className="text-sm text-gray-500">Describe your mandate above to rank which crises match — and why.</p>
              </div>
            )}
          </div>
          )
        })()}

        {/* ═══ QUERY TAB ═══════════════════════════════════════════════════ */}
        {tab === 'query' && (
          <div className="flex flex-col gap-6">
            <form onSubmit={handleQuery} className="flex flex-col gap-3">
              <textarea
                className="w-full bg-gray-900 border border-gray-700 rounded-xl px-4 py-3 text-gray-100 placeholder-gray-500 focus:outline-none focus:border-blue-500 resize-none text-sm"
                rows={2}
                placeholder="e.g. Which African countries have the most people in need but under 30% funded in 2024?"
                value={query}
                onChange={e => setQuery(e.target.value)}
              />
              <div className="flex items-center gap-4 flex-wrap">
                {['from','to'].map((label, idx) => (
                  <label key={label} className="text-sm text-gray-400 flex items-center gap-2">
                    Year {label}
                    <select className="bg-gray-900 border border-gray-700 rounded px-2 py-1 text-gray-100 text-sm"
                      value={idx === 0 ? yearFrom : yearTo}
                      onChange={e => idx === 0 ? setYearFrom(Number(e.target.value)) : setYearTo(Number(e.target.value))}>
                      {YEARS.map(y => <option key={y}>{y}</option>)}
                    </select>
                  </label>
                ))}
                <label className="text-sm text-gray-400 flex items-center gap-2 ml-auto">
                  <input type="checkbox" checked={showSql} onChange={e => setShowSql(e.target.checked)} className="accent-blue-500" />
                  Show SQL
                </label>
                <button type="submit" disabled={qLoading || !query.trim()}
                  className="bg-blue-600 hover:bg-blue-500 disabled:opacity-40 text-white px-6 py-2 rounded-lg text-sm font-medium transition-colors">
                  {qLoading ? 'Analysing…' : 'Analyse'}
                </button>
              </div>
            </form>

            {qError && <div className="bg-red-950 border border-red-800 text-red-300 rounded-xl px-4 py-3 text-sm">{qError}</div>}

            {qResult && (
              <div className="flex flex-col gap-6">
                {showSql && (
                  <div className="bg-gray-900 border border-gray-800 rounded-xl p-4">
                    <p className="text-xs text-gray-500 mb-2 uppercase tracking-wide font-medium">Generated SQL</p>
                    <pre className="text-xs text-green-300 overflow-x-auto whitespace-pre-wrap">{qResult.sql}</pre>
                  </div>
                )}
                <p className="text-sm text-gray-500">{qResult.row_count} rows returned</p>

                {/* World map for query results */}
                {Object.keys(qIsoMap).length > 0 && (
                  <div className="bg-gray-900 border border-gray-800 rounded-2xl p-6">
                    <p className="text-xs font-semibold text-gray-400 mb-3 uppercase tracking-wide">Result Countries</p>
                    <ComposableMap projectionConfig={{ scale: 153 }}>
                      <Geographies geography={GEO_URL}>
                        {({ geographies }: { geographies: { rsmKey: string; properties: Record<string,unknown> }[] }) =>
                          geographies.map(geo => {
                            const iso = geo.properties.ISO_A3 as string
                            const score = qIsoMap[iso]
                            return (
                              <Geography key={geo.rsmKey} geography={geo}
                                fill={score != null ? scoreToColor(score) : '#1f2937'}
                                stroke="#111827" strokeWidth={0.4}
                                style={{ default:{outline:'none'}, hover:{fill:'#60a5fa',outline:'none'}, pressed:{outline:'none'} }}
                              />
                            )
                          })
                        }
                      </Geographies>
                    </ComposableMap>
                  </div>
                )}

                {/* Results table */}
                <div className="bg-gray-900 border border-gray-800 rounded-2xl overflow-hidden">
                  <div className="overflow-x-auto">
                    <table className="w-full text-sm">
                      <thead className="text-gray-500 text-xs uppercase border-b border-gray-800">
                        <tr>{qResult.columns.map(col => (
                          <th key={col} className="px-4 py-3 text-left font-medium whitespace-nowrap">{col}</th>
                        ))}</tr>
                      </thead>
                      <tbody className="divide-y divide-gray-800/60">
                        {qResult.rows.map((row, i) => (
                          <tr key={i} className="hover:bg-gray-800/50 transition-colors">
                            {qResult.columns.map(col => (
                              <td key={col} className="px-4 py-2.5 text-gray-300 whitespace-nowrap font-mono text-xs">
                                {row[col] == null
                                  ? <span className="text-gray-700">—</span>
                                  : typeof row[col] === 'number'
                                    ? (row[col] as number) > 1000
                                      ? (row[col] as number).toLocaleString()
                                      : (row[col] as number).toFixed(3)
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

            {!qResult && !qLoading && !qError && (
              <div className="mt-20 text-center text-gray-700">
                <p className="text-4xl mb-4">🔍</p>
                <p className="text-sm text-gray-500">Try: <em className="text-gray-400">Which countries have more than 5M people in need but under 30% funded?</em></p>
              </div>
            )}
          </div>
        )}
      </main>
    </div>
  )
}
