import { useState, useEffect, useRef, useCallback, type MouseEvent } from 'react'
import axios from 'axios'
import {
  ScatterChart, Scatter, XAxis, YAxis, ZAxis, CartesianGrid,
  Tooltip as ReTooltip, ResponsiveContainer, ReferenceLine,
  Label, Cell,
} from 'recharts'
import { ComposableMap, Geographies, Geography } from 'react-simple-maps'

const GEO_URL = 'https://cdn.jsdelivr.net/npm/world-atlas@2/countries-110m.json'
const YEARS = [2019, 2020, 2021, 2022, 2023, 2024, 2025]

// world-atlas@2 topojson has only `id` (ISO-3166-1 numeric) — no alpha-3 codes.
// Map numeric → alpha-3 so the choropleth can join against country_iso3 from the backend.
const NUM_TO_ISO3: Record<string, string> = {
  '004':'AFG','008':'ALB','012':'DZA','020':'AND','024':'AGO','028':'ATG','031':'AZE','032':'ARG','036':'AUS','040':'AUT',
  '044':'BHS','048':'BHR','050':'BGD','051':'ARM','052':'BRB','056':'BEL','064':'BTN','068':'BOL','070':'BIH','072':'BWA',
  '076':'BRA','084':'BLZ','090':'SLB','096':'BRN','100':'BGR','104':'MMR','108':'BDI','112':'BLR','116':'KHM','120':'CMR',
  '124':'CAN','132':'CPV','140':'CAF','144':'LKA','148':'TCD','152':'CHL','156':'CHN','158':'TWN','170':'COL','174':'COM',
  '178':'COG','180':'COD','188':'CRI','191':'HRV','192':'CUB','196':'CYP','203':'CZE','204':'BEN','208':'DNK','212':'DMA',
  '214':'DOM','218':'ECU','222':'SLV','226':'GNQ','231':'ETH','232':'ERI','233':'EST','242':'FJI','246':'FIN','250':'FRA',
  '260':'ATF','262':'DJI','266':'GAB','268':'GEO','270':'GMB','275':'PSE','276':'DEU','288':'GHA','296':'KIR','300':'GRC',
  '304':'GRL','308':'GRD','320':'GTM','324':'GIN','328':'GUY','332':'HTI','340':'HND','344':'HKG','348':'HUN','352':'ISL',
  '356':'IND','360':'IDN','364':'IRN','368':'IRQ','372':'IRL','376':'ISR','380':'ITA','384':'CIV','388':'JAM','392':'JPN',
  '398':'KAZ','400':'JOR','404':'KEN','408':'PRK','410':'KOR','414':'KWT','417':'KGZ','418':'LAO','422':'LBN','426':'LSO',
  '428':'LVA','430':'LBR','434':'LBY','438':'LIE','440':'LTU','442':'LUX','450':'MDG','454':'MWI','458':'MYS','462':'MDV',
  '466':'MLI','470':'MLT','478':'MRT','480':'MUS','484':'MEX','496':'MNG','498':'MDA','499':'MNE','504':'MAR','508':'MOZ',
  '512':'OMN','516':'NAM','520':'NRU','524':'NPL','528':'NLD','540':'NCL','548':'VUT','554':'NZL','558':'NIC','562':'NER',
  '566':'NGA','578':'NOR','586':'PAK','591':'PAN','598':'PNG','600':'PRY','604':'PER','608':'PHL','616':'POL','620':'PRT',
  '624':'GNB','626':'TLS','630':'PRI','634':'QAT','642':'ROU','643':'RUS','646':'RWA','682':'SAU','686':'SEN','688':'SRB',
  '690':'SYC','694':'SLE','702':'SGP','703':'SVK','704':'VNM','705':'SVN','706':'SOM','710':'ZAF','716':'ZWE','724':'ESP',
  '728':'SSD','729':'SDN','732':'ESH','740':'SUR','748':'SWZ','752':'SWE','756':'CHE','760':'SYR','762':'TJK','764':'THA',
  '768':'TGO','780':'TTO','784':'ARE','788':'TUN','792':'TUR','795':'TKM','798':'TUV','800':'UGA','804':'UKR','807':'MKD',
  '818':'EGY','826':'GBR','834':'TZA','840':'USA','854':'BFA','858':'URY','860':'UZB','862':'VEN','882':'WSM','887':'YEM',
  '894':'ZMB','-99':'XKX',
}
const geoIso3 = (geo: { id?: string | number; properties?: Record<string, unknown> }) =>
  NUM_TO_ISO3[String(geo.id ?? '').padStart(3, '0')] ?? null

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
  inform_severity:     { label: 'Crisis Severity (INFORM)', group: 'Severity',
    desc: 'INFORM Global Crisis Severity Index: composite of impact (20%), conditions of affected people (50%), and crisis complexity (30%). Normalised 0–1 from 1–10 scale.',
    source: 'INFORM / ACAPS Global Crisis Severity Index' },
  mismatch_score:      { label: 'Severity-Funding Mismatch', group: 'Severity',
    desc: 'severity × (1 − coverage ratio): peaks where a very severe crisis receives the least funding. The top-left-quadrant "most overlooked" metric.',
    source: 'INFORM Severity × FTS coverage ratio' },
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
      <div className="flex h-2 w-28 rounded overflow-hidden bg-[color:var(--color-surface-muted)]">
        <div style={{ width: `${gapW}%`, background: scoreToColor(row.overlooked_score) }} />
        <div style={{ width: `${Math.max(0, bonusW)}%` }} className="bg-[color:var(--color-fg-muted)] opacity-40" />
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
    <div className="bg-white rounded-[10px] p-3 text-xs shadow-[0_4px_12px_rgba(24,24,27,0.06),_0_1px_2px_rgba(24,24,27,0.04)] max-w-xs" style={{ boxShadow: '0 2px 8px rgba(24, 24, 27, 0.08)' }}>
      <div className="font-semibold text-[color:var(--color-fg)] flex items-center gap-2 mb-1.5">
        {r.country_name}
        {r.robust && <span className="bg-[color:var(--color-accent-bg)] text-[color:var(--color-accent-hover)] px-1.5 py-0.5 rounded-[6px] text-[10px] font-medium uppercase tracking-wider">Robust</span>}
      </div>
      <div className="text-[color:var(--color-fg-muted)] space-y-0.5 font-mono tabular-nums">
        <div>Score: <span className="text-[color:var(--color-fg)]">{r.overlooked_score.toFixed(3)}</span>
          <span className="text-[color:var(--color-fg-subtle)] ml-2">Borda #{r.borda_rank}</span></div>
        <div>People in need: <span className="text-[color:var(--color-fg)]">{millions(r.pin)}</span></div>
        <div>Funded: <span className="text-[color:var(--color-fg)]">{pct(r.coverage_ratio)}</span></div>
        <div>Requirements: <span className="text-[color:var(--color-fg)]">${millions(r.requirements_usd)}</span></div>
        {(r.years_underfunded ?? 0) > 0 &&
          <div>Underfunded {r.years_underfunded}/{r.n_coverage_years} years</div>}
      </div>
      <div className="mt-2 text-[color:var(--color-fg-muted)] leading-relaxed">{r.explanation}</div>
    </div>
  )
}

// ── PCA tooltip ───────────────────────────────────────────────────────────────

function PcaTooltip({ active, payload }: { active?: boolean; payload?: { payload: ScoreRow }[] }) {
  if (!active || !payload?.length) return null
  const r = payload[0].payload
  return (
    <div className="bg-white rounded-[10px] p-3 text-xs shadow-[0_4px_12px_rgba(24,24,27,0.06),_0_1px_2px_rgba(24,24,27,0.04)] max-w-xs" style={{ boxShadow: '0 2px 8px rgba(24, 24, 27, 0.08)' }}>
      <div className="font-semibold text-[color:var(--color-fg)] mb-1.5">
        {r.country_name} {r.robust ? '★' : ''}
      </div>
      <div className="text-[color:var(--color-fg-muted)] space-y-0.5 font-mono tabular-nums">
        <div>Score: <span className="text-[color:var(--color-fg)]">{r.overlooked_score.toFixed(3)}</span></div>
        <div>Borda rank: <span className="text-[color:var(--color-fg)]">#{r.borda_rank}</span></div>
        <div className="text-[color:var(--color-fg-subtle)] mt-1 font-sans">{r.explanation.split(';')[0]}</div>
      </div>
    </div>
  )
}

// ── Column label mapping for query results ────────────────────────────────────
const COLUMN_LABELS: Record<string, string> = {
  country_iso3:        'Country Code',
  country_name:        'Country',
  continent:           'Continent',
  region_name:         'Region',
  year:                'Year',
  sector:              'Sector',
  rank:                'Rank',
  borda_rank:          'Borda Rank',
  mcda_rank:           'Priority Rank',
  mcda_score:          'MCDA Score',
  overlooked_score:    'Overlooked Score',
  gap_component:       'Gap Component',
  confidence_weight:   'Confidence',
  need_scale:          'Need Scale',
  funding_gap:         'Funding Gap',
  structural_score:    'Structural Score',
  structural_neglect:  'Structural Neglect',
  trend_score:         'Trend Score',
  trend_worsening:     'Trend (Worsening)',
  rank_delta:          'Rank Change',
  has_vulnerability:   'Has Vulnerability',
  pin:                 'People in Need',
  targeted:            'Targeted Population',
  requirements_usd:    'Requirements (USD)',
  fts_funding_usd:     'FTS Funding (USD)',
  coverage_ratio:      'Coverage Ratio',
  coverage_slope:      'Coverage Trend',
  years_underfunded:   'Years Underfunded',
  n_coverage_years:    '# Coverage Years',
  robust:              'Robust',
  data_complete:       'Data Complete',
  explanation:         'Explanation',
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
  const [aYear]                        = useState(2025)
  const [aLoading, setALoading]       = useState(false)
  const [aError, setAError]           = useState<string | null>(null)
  const [aResult, setAResult]         = useState<AnalyzeResponse | null>(null)
  const [aSector, setASector]         = useState('ALL')
  const [aSortBy, setASortBy]         = useState<string>('mcda_rank')
  const [aSortDir, setASortDir]       = useState<1|-1>(1)
  const [aCompare, setACompare]       = useState<Set<string>>(new Set())
  const [aSearch, setASearch]         = useState('')
  const [showGlossary, setShowGlossary] = useState(false)
  const [pendingScores, setPendingScores] = useState<Record<string, number>>({})
  const [dimOrder, setDimOrder] = useState<string[]>([])
  const [typewriter, setTypewriter] = useState('')
  const [sqlTypewriter, setSqlTypewriter] = useState('')

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
    const updated = { ...aResult.importance_scores, [dim]: Math.max(0, Math.min(10, aResult.importance_scores[dim] + delta)) }
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
  const [showSql, setShowSql]   = useState(true)
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
  }, [runScore])

  // Mirror backend importance scores into local slider state on every analyze response.
  // Also capture the dimension display order on the FIRST response so sliders
  // don't reshuffle as scores change.
  useEffect(() => {
    if (aResult) {
      setPendingScores(aResult.importance_scores)
      setDimOrder(prev => prev.length > 0 ? prev : Object.keys(aResult.importance_scores))
    }
  }, [aResult])

  // Typewriter placeholder for the Prioritize prompt — cycles example mandates
  // when the textarea is empty. Pauses when the user starts typing.
  useEffect(() => {
    if (aPrompt.trim().length > 0) return
    const examples = [
      'Water scarcity in chronically neglected conflict zones',
      'Food insecurity at massive scale regardless of funding trend',
      'Structural neglect and governance fragility',
      'Climate-vulnerable displacement crises',
    ]
    let i = 0, j = 0, dir: 'type' | 'hold' | 'erase' = 'type'
    let cancelled = false
    const tick = () => {
      if (cancelled) return
      const full = examples[i]
      if (dir === 'type') {
        j += 1
        setTypewriter(full.slice(0, j))
        if (j >= full.length) { dir = 'hold'; setTimeout(tick, 1600); return }
        setTimeout(tick, 45)
      } else if (dir === 'hold') {
        dir = 'erase'
        setTimeout(tick, 0)
      } else {
        j -= 1
        setTypewriter(full.slice(0, j))
        if (j <= 0) { dir = 'type'; i = (i + 1) % examples.length; setTimeout(tick, 120); return }
        setTimeout(tick, 22)
      }
    }
    tick()
    return () => { cancelled = true }
  }, [aPrompt])

  // Typewriter placeholder for the SQL Query prompt — cycles example queries.
  useEffect(() => {
    if (query.trim().length > 0) return
    const examples = [
      'Which African countries have the most people in need but under 30% funded in 2024?',
      'Top 10 countries with the largest funding shortfall in 2025',
      'WASH-sector crises with coverage below 20%',
      'Countries where coverage has worsened every year since 2020',
    ]
    let i = 0, j = 0, dir: 'type' | 'hold' | 'erase' = 'type'
    let cancelled = false
    const tick = () => {
      if (cancelled) return
      const full = examples[i]
      if (dir === 'type') {
        j += 1
        setSqlTypewriter(full.slice(0, j))
        if (j >= full.length) { dir = 'hold'; setTimeout(tick, 1600); return }
        setTimeout(tick, 45)
      } else if (dir === 'hold') {
        dir = 'erase'
        setTimeout(tick, 0)
      } else {
        j -= 1
        setSqlTypewriter(full.slice(0, j))
        if (j <= 0) { dir = 'type'; i = (i + 1) % examples.length; setTimeout(tick, 120); return }
        setTimeout(tick, 22)
      }
    }
    tick()
    return () => { cancelled = true }
  }, [query])

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
    <div className="min-h-screen bg-[color:var(--color-surface)] text-[color:var(--color-fg)] flex flex-col">

      {/* Tabs — sticky glass nav with sliding underline */}
      <div className="glass-nav sticky top-0 z-30 px-12">
        <div className="relative flex gap-10">
          {([
            ['score',   'Gap Scoring'],
            ['analyze', 'Prioritize'],
            ['query',   'SQL Query'],
          ] as const).map(([t, label], idx) => {
            const active = tab === t
            return (
              <button key={t} onClick={() => setTab(t)}
                data-tab-index={idx}
                className={`relative px-0 py-4 text-sm font-medium transition-colors duration-300 ${
                  active
                    ? 'text-[color:var(--color-fg)]'
                    : 'text-[color:var(--color-fg-muted)] hover:text-[color:var(--color-fg)]'}`}>
                {label}
                {active && (
                  <span
                    aria-hidden
                    className="absolute left-0 right-0 bottom-0 h-[2px] rounded-full"
                    style={{
                      background: 'linear-gradient(90deg, transparent 0%, var(--color-accent) 15%, var(--color-accent) 85%, transparent 100%)',
                    }}
                  />
                )}
              </button>
            )
          })}
        </div>
      </div>

      <main className="flex-1 px-12 py-12 max-w-7xl mx-auto w-full">

        {/* ═══ SCORE TAB ═══════════════════════════════════════════════════ */}
        {tab === 'score' && (
          <div className="flex flex-col gap-12">

            {/* Controls */}
            <div className="flex flex-col gap-6">

              <div className="flex flex-wrap items-end gap-8">

                {/* Year selector + play */}
                <div className="flex flex-col gap-2">
                  <span className="text-[11px] font-medium uppercase tracking-[0.06em] text-[color:var(--color-fg-muted)]">Year</span>
                  <div className="flex items-center gap-3">
                    <div className="inline-flex rounded-[10px] border border-[color:var(--color-border)] bg-[color:var(--color-surface-muted)] p-0.5">
                      {YEARS.map(y => (
                        <button key={y} onClick={() => { setPlaying(false); setScoreYear(y); runScore(y, weights) }}
                          className={`px-3 py-1 text-[13px] font-mono tabular-nums rounded transition-colors ${
 displayYear === y
                              ? 'bg-[color:var(--color-surface)] text-[color:var(--color-fg)] shadow-[0_1px_2px_rgba(24,24,27,0.04),_0_1px_1px_rgba(24,24,27,0.02)]'
                              : 'text-[color:var(--color-fg-muted)] hover:text-[color:var(--color-fg)]'}`}>
                          {y}
                        </button>
                      ))}
                    </div>
                    <button
                      onClick={() => { setAnimYear(scoreYear); setPlaying(p => !p) }}
                      className={`flex items-center gap-1.5 px-3 py-1.5 rounded-[10px] text-xs font-medium border transition-colors ${
 playing
                          ? 'border-[color:var(--color-accent)] text-[color:var(--color-accent)] bg-[color:var(--color-accent-bg)]'
                          : 'border-[color:var(--color-border)] text-[color:var(--color-fg-muted)] hover:text-[color:var(--color-fg)] hover:border-[color:var(--color-border-strong)]'}`}>
                      {playing ? (
                        <><svg className="w-3 h-3" viewBox="0 0 10 10" fill="currentColor"><rect x="2" y="2" width="6" height="6" /></svg> Stop</>
                      ) : (
                        <><svg className="w-3 h-3" viewBox="0 0 10 10" fill="currentColor"><polygon points="2,1 9,5 2,9" /></svg> Animate</>
                      )}
                    </button>
                  </div>
                </div>

                {/* Weight summary — stacked chips with fill bars */}
                <div className="flex flex-col gap-2 flex-1 min-w-[280px]">
                  <span className="text-[11px] font-medium uppercase tracking-[0.06em] text-[color:var(--color-fg-muted)]">Active weights</span>
                  <div className="grid grid-cols-4 gap-4">
                    {(['scale','gap','structural','trend'] as (keyof Weights)[]).map(k => (
                      <div key={k} className="flex flex-col gap-1">
                        <span className="text-[11px] font-medium uppercase tracking-[0.06em] text-[color:var(--color-fg-muted)]">{k}</span>
                        <span className="text-sm font-mono font-medium tabular-nums text-[color:var(--color-fg)]">{weights[k].toFixed(2)}</span>
                        <div className="h-1 rounded-full bg-[color:var(--color-border)] overflow-hidden">
                          <div className="h-full bg-[color:var(--color-accent)]" style={{ width: `${weights[k] * 100}%` }} />
                        </div>
                      </div>
                    ))}
                  </div>
                </div>

                <div className="flex gap-2 ml-auto">
                  <button onClick={() => runScore(scoreYear, weights)} disabled={sLoading}
                    className="btn-primary px-4 py-2 rounded-[12px] text-sm font-medium">
                    {sLoading ? '…' : 'Recalculate'}
                  </button>
                  {sResult && (
                    <button onClick={() => exportCsv(sResult.rows, sResult.year)}
                      className="border border-[color:var(--color-border)] hover:border-[color:var(--color-border-strong)] text-[color:var(--color-fg-muted)] hover:text-[color:var(--color-fg)] px-3 py-2 rounded-[10px] text-sm font-medium transition-colors">
                      ↓ CSV
                    </button>
                  )}
                </div>

              </div>

              {/* Formula — its own documentation panel */}
              <div className="bg-[color:var(--color-surface-muted)] border border-[color:var(--color-border)] rounded-[10px] px-4 py-3">
                <div className="text-[11px] font-medium uppercase tracking-[0.06em] text-[color:var(--color-fg-muted)] mb-1.5">Formula</div>
                <div className="text-xs font-mono tabular-nums text-[color:var(--color-fg-muted)] leading-relaxed pl-2">
                  <span className="text-[color:var(--color-fg)] font-semibold">overlooked</span> = (
                  <span className="text-[color:var(--color-fg)] font-semibold">w_scale·scale</span> +
                  <span className="text-[color:var(--color-fg)] font-semibold"> w_gap·gap</span> +
                  <span className="text-[color:var(--color-fg)] font-semibold"> w_structural·structural</span> +
                  <span className="text-[color:var(--color-fg)] font-semibold"> w_trend·trend</span>) ×
                  <span className="text-[color:var(--color-fg)] font-semibold"> confidence</span>
                  <span className="ml-3 text-[color:var(--color-fg-subtle)]">· validated by Borda ensemble of 4 rankings</span>
                </div>
              </div>
            </div>

            {sError && (
              <div className="bg-[color:var(--color-surface-muted)] border border-[color:var(--color-border)] rounded-[10px] px-4 py-3 text-sm text-[color:var(--color-fg)]">{sError}</div>
            )}

            {sLoading && !sResult && (
              <div className="flex items-center justify-center py-24 text-[color:var(--color-fg-subtle)]">
                <div className="text-sm">Loading {displayYear} data…</div>
              </div>
            )}

            {sResult && (
              <>
                {/* Stat cards — flat 4-col grid, no borders */}
                <div className="grid grid-cols-4 gap-6">
                  {[
                    { label: 'Countries assessed', value: sResult.row_count },
                    { label: 'Robustly overlooked', value: sResult.meta.robust_count,
                      sub: 'top quartile in all 4 rankings' },
                    { label: '#1 most overlooked', value: rows[0]?.country_name ?? '—', small: true },
                    { label: 'PC1+PC2 variance',
                      value: (Object.values(sResult.meta.pca_explained_var).slice(0,2).reduce((a,b)=>a+b,0)*100).toFixed(0)+'%' },
                  ].map((c, idx) => (
                    <div key={c.label} className={`flex flex-col gap-2 ${idx > 0 ? 'border-l border-[color:var(--color-border)] pl-6' : ''}`}>
                      <div className="text-[11px] font-medium uppercase tracking-[0.06em] text-[color:var(--color-fg-muted)]">{c.label}</div>
                      <div className={`${c.small ? 'font-serif text-[22px]' : 'font-serif text-[34px]'} font-normal tabular-nums text-[color:var(--color-fg)] leading-[1.1]`}>{c.value}</div>
                      {c.sub && <div className="text-xs text-[color:var(--color-fg-subtle)]">{c.sub}</div>}
                    </div>
                  ))}
                </div>

                {/* ── BUBBLE CHART ── */}
                <div className="border border-[color:var(--color-border)] rounded-[20px] p-8 bg-[color:var(--color-surface)]">
                  <div className="flex items-baseline justify-between mb-1">
                    <h2 className="font-serif text-[26px] font-normal tracking-[-0.01em] text-[color:var(--color-fg)] leading-tight">Need vs. Funding Space</h2>
                    <span className="text-[11px] font-medium uppercase tracking-[0.06em] text-[color:var(--color-fg-muted)]">bubble = requirements · top-8 labelled</span>
                  </div>
                  <p className="text-sm text-[color:var(--color-fg-muted)] mb-6">
                    Crises in the top-left corner have the largest unmet need — high people in need, low funding coverage.
                  </p>
                  <ResponsiveContainer width="100%" height={380}>
                    <ScatterChart margin={{ top: 20, right: 30, bottom: 40, left: 60 }}>
                      <CartesianGrid strokeDasharray="2 4" stroke="#e4e4e7" strokeOpacity={0.8} />
                      <XAxis type="number" dataKey="x" domain={[0, 105]}
                        tick={{ fill: '#52525b', fontSize: 11, fontFamily: 'JetBrains Mono, monospace' }}
                        stroke="#d4d4d8" tickLine={{ stroke: '#d4d4d8' }}>
                        <Label value="% Funded (coverage ratio)" position="insideBottom" offset={-25}
                          style={{ fill: '#52525b', fontSize: 12, fontWeight: 500 }} />
                      </XAxis>
                      <YAxis type="number" dataKey="y" domain={[-2, 2]}
                        tick={{ fill: '#52525b', fontSize: 11, fontFamily: 'JetBrains Mono, monospace' }}
                        stroke="#d4d4d8" tickLine={{ stroke: '#d4d4d8' }}
                        tickFormatter={v => `${Math.pow(10,v).toFixed(1)}M`}>
                        <Label value="People in Need (log scale)" angle={-90} position="left" offset={40}
                          style={{ fill: '#52525b', fontSize: 12, fontWeight: 500, textAnchor: 'middle' }} />
                      </YAxis>
                      <ZAxis type="number" dataKey="_z" range={[40, 900]} />
                      <ReferenceLine x={30} stroke="#a1a1aa" strokeDasharray="4 4"
                        label={{ value: '30% threshold', fill: '#52525b', fontSize: 11, fontFamily: 'JetBrains Mono, monospace', position: 'insideTopRight' }} />
                      <ReTooltip content={<BubbleTooltip />} cursor={{ strokeDasharray: '3 3', stroke: '#a1a1aa' }} />
                      <Scatter data={bubbleData} shape={<BubbleDot />} />
                    </ScatterChart>
                  </ResponsiveContainer>
                  {/* Legend */}
                  <div className="flex items-center gap-6 mt-4 text-xs text-[color:var(--color-fg-muted)] justify-center">
                    {[['#3884c8','low gap'],['#f97316','medium gap'],['#dc2626','high gap'],['#7c3aed','robustly overlooked']].map(([c,l])=>(
                      <span key={l} className="flex items-center gap-2">
                        <span className="w-3 h-3 rounded-[6px] inline-block" style={{background: c}} />
                        <span className="font-medium">{l}</span>
                      </span>
                    ))}
                  </div>
                </div>

                {/* ── WORLD MAP ── */}
                <div className="border border-[color:var(--color-border)] rounded-[20px] p-8 bg-[color:var(--color-surface)]">
                  <h2 className="font-serif text-[26px] font-normal tracking-[-0.01em] text-[color:var(--color-fg)] leading-tight mb-1">World Map — Overlooked Score {displayYear}</h2>
                  <p className="text-sm text-[color:var(--color-fg-muted)] mb-6">Hover over a country for details. Purple = robustly overlooked across all ranking methods.</p>

                  <div className="relative">
                    <ComposableMap projectionConfig={{ scale: 153 }} style={{ background: 'transparent' }}>
                      <Geographies geography={GEO_URL}>
                        {({ geographies }: { geographies: { rsmKey: string; properties: Record<string,unknown> }[] }) =>
                          geographies.map(geo => {
                            const iso  = geoIso3(geo) ?? ''
                            const row  = sIsoMap[iso]
                            const fill = row ? scoreToColor(row.overlooked_score, row.robust) : '#fafafa'
                            return (
                              <Geography
                                key={geo.rsmKey} geography={geo}
                                fill={fill} stroke="#e4e4e7" strokeWidth={0.5}
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
                                  hover:    { fill: row ? '#0d9488' : '#f0fdfa', outline: 'none', cursor: row ? 'pointer' : 'default' },
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
                      className="pointer-events-none fixed z-50 bg-white rounded-[10px] p-3 text-xs w-64"
                      style={{
                        left: tooltipPos.x + 14,
                        top:  tooltipPos.y - 10,
                        transform: tooltipPos.x > window.innerWidth - 280 ? 'translateX(-110%)' : undefined,
                        }}
                    >
                      <div className="flex items-center gap-2 mb-2">
                        <span className="font-semibold text-sm text-[color:var(--color-fg)]">{hoveredCountry.country_name}</span>
                        {hoveredCountry.robust && (
                          <span className="bg-[color:var(--color-accent-bg)] text-[color:var(--color-accent-hover)] px-1.5 py-0.5 rounded-[6px] text-[10px] font-medium uppercase tracking-wider">Robust</span>
                        )}
                      </div>
                      <div className="space-y-1 text-[color:var(--color-fg-muted)] font-mono tabular-nums">
                        <div className="flex justify-between">
                          <span>Overlooked score</span>
                          <span style={{ color: scoreToColor(hoveredCountry.overlooked_score, hoveredCountry.robust) }}>
                            {hoveredCountry.overlooked_score.toFixed(3)}
                          </span>
                        </div>
                        <div className="flex justify-between">
                          <span>Borda rank</span>
                          <span className="text-[color:var(--color-fg)]">#{hoveredCountry.borda_rank}</span>
                        </div>
                        <div className="flex justify-between">
                          <span>People in need</span>
                          <span className="text-[color:var(--color-fg)]">{millions(hoveredCountry.pin)}</span>
                        </div>
                        <div className="flex justify-between">
                          <span>Funded</span>
                          <span className="text-[color:var(--color-fg)]">{pct(hoveredCountry.coverage_ratio)}</span>
                        </div>
                        {(hoveredCountry.years_underfunded ?? 0) > 0 && (
                          <div className="flex justify-between">
                            <span>Yrs &lt;30%</span>
                            <span className="text-[color:var(--color-accent-hover)]">{hoveredCountry.years_underfunded}/{hoveredCountry.n_coverage_years}</span>
                          </div>
                        )}
                      </div>
                      <div className="mt-2 pt-2 border-t border-[color:var(--color-border)] text-[color:var(--color-fg-muted)] leading-relaxed">
                        {hoveredCountry.explanation}
                      </div>
                    </div>
                  )}

                  {/* Gradient legend */}
                  <div className="flex items-center gap-4 mt-5 justify-center flex-wrap text-xs text-[color:var(--color-fg-muted)]">
                    <span className="font-medium">Low gap</span>
                    <div className="h-2 w-48 rounded-full" style={{
                      background: 'linear-gradient(to right, #3884c8, #fb9216, #dc2626)'
                    }} />
                    <span className="font-medium">High gap</span>
                    <span className="ml-4 flex items-center gap-2">
                      <span className="w-3 h-3 rounded-[6px] inline-block" style={{ background: '#7c3aed' }} />
                      <span className="font-medium">Robustly overlooked</span>
                    </span>
                    <span className="flex items-center gap-2">
                      <span className="w-3 h-3 rounded-[6px] inline-block bg-[color:var(--color-surface-muted)] border border-[color:var(--color-border)]" />
                      <span className="font-medium">No crisis data</span>
                    </span>
                  </div>
                </div>

                {/* ── PCA SCATTER ── */}
                <div className="border border-[color:var(--color-border)] rounded-[20px] p-8 bg-[color:var(--color-surface)]">
                  <h2 className="font-serif text-[26px] font-normal tracking-[-0.01em] text-[color:var(--color-fg)] leading-tight mb-1">Country Similarity Map (PCA)</h2>
                  <p className="text-sm text-[color:var(--color-fg-muted)] mb-6">
                    Countries near each other share similar crisis profiles across all dimensions.
                    PC1 captures overall severity + gap · PC2 separates structural from acute neglect.
                  </p>
                  <ResponsiveContainer width="100%" height={320}>
                    <ScatterChart margin={{ top: 10, right: 20, bottom: 30, left: 50 }}>
                      <CartesianGrid strokeDasharray="2 4" stroke="#e4e4e7" strokeOpacity={0.8} />
                      <XAxis type="number" dataKey="pc1"
                        tick={{ fill: '#52525b', fontSize: 11, fontFamily: 'JetBrains Mono, monospace' }}
                        stroke="#d4d4d8" tickLine={{ stroke: '#d4d4d8' }}>
                        <Label value="PC1 — severity + funding gap" position="insideBottom" offset={-20}
                          style={{ fill: '#52525b', fontSize: 12, fontWeight: 500 }} />
                      </XAxis>
                      <YAxis type="number" dataKey="pc2"
                        tick={{ fill: '#52525b', fontSize: 11, fontFamily: 'JetBrains Mono, monospace' }}
                        stroke="#d4d4d8" tickLine={{ stroke: '#d4d4d8' }}>
                        <Label value="PC2 — structural vs acute" angle={-90} position="left" offset={30}
                          style={{ fill: '#52525b', fontSize: 12, fontWeight: 500, textAnchor: 'middle' }} />
                      </YAxis>
                      <ZAxis range={[40, 40]} />
                      <ReTooltip content={<PcaTooltip />} cursor={{ strokeDasharray: '3 3', stroke: '#a1a1aa' }} />
                      <Scatter data={rows} shape={<PcaDot />} />
                    </ScatterChart>
                  </ResponsiveContainer>
                </div>

                {/* ── RANKINGS TABLE ── */}
                <div className="border border-[color:var(--color-border)] rounded-[20px] overflow-hidden bg-[color:var(--color-surface)]">
                  <div className="px-6 py-5 border-b border-[color:var(--color-border)]">
                    <h2 className="font-serif text-[26px] font-normal tracking-[-0.01em] text-[color:var(--color-fg)] leading-tight">Full Rankings — {displayYear}</h2>
                    <p className="text-sm text-[color:var(--color-fg-muted)] mt-1">
                      Bar = gap component (colour) + structural bonus (purple). Borda validates across 4 independent methods.
                    </p>
                  </div>
                  <div className="overflow-x-auto">
                    <table className="w-full text-sm">
                      <thead className="bg-[color:var(--color-surface-muted)] text-[11px] font-medium uppercase tracking-[0.06em] text-[color:var(--color-fg-muted)] border-b border-[color:var(--color-border)]">
                        <tr>
                          <th className="px-4 py-2.5 text-left w-10">#</th>
                          <th className="px-4 py-2.5 text-left">Country</th>
                          <th className="px-4 py-2.5 text-left min-w-[160px]">Score</th>
                          <th className="px-4 py-2.5 text-right">Borda</th>
                          <th className="px-4 py-2.5 text-right">PIN</th>
                          <th className="px-4 py-2.5 text-right">Funded</th>
                          <th className="px-4 py-2.5 text-right">Yrs &lt;30%</th>
                          <th className="px-4 py-2.5 text-left min-w-[280px]">Notes</th>
                        </tr>
                      </thead>
                      <tbody>
                        {rows.map((r) => (
                          <tr key={r.country_iso3}
                            className={`transition-colors hover:bg-[color:var(--color-accent-bg)] border-b border-[color:var(--color-border)] last:border-b-0 ${
 r.robust ? 'bg-[color:var(--color-accent-bg)]' : ''}`}>
                            <td className="px-4 py-3 font-mono tabular-nums text-xs text-[color:var(--color-fg-muted)]">
                              {r.rank}
                            </td>
                            <td className="px-4 py-3">
                              <div className="flex items-center gap-2 flex-wrap">
                                <span className="text-[color:var(--color-fg)] text-sm font-medium">{r.country_name}</span>
                                {r.robust && <span className="text-[10px] font-medium uppercase tracking-wider bg-[color:var(--color-accent-bg)] text-[color:var(--color-accent-hover)] px-1.5 py-0.5 rounded-[6px]">Robust</span>}
                                {!r.data_complete && <span className="text-[10px] font-medium uppercase tracking-wider border border-[color:var(--color-border-strong)] text-[color:var(--color-fg-muted)] px-1.5 py-0.5 rounded-[6px]">Partial</span>}
                              </div>
                              <div className="text-xs text-[color:var(--color-fg-subtle)] mt-0.5">{r.region_name}</div>
                            </td>
                            <td className="px-4 py-3">
                              <ScoreBar row={r} maxScore={maxScore} />
                            </td>
                            <td className="px-4 py-3 text-right font-mono tabular-nums text-xs text-[color:var(--color-fg-muted)]">#{r.borda_rank}</td>
                            <td className="px-4 py-3 text-right font-mono tabular-nums text-xs text-[color:var(--color-fg)]">{millions(r.pin)}</td>
                            <td className="px-4 py-3 text-right font-mono tabular-nums text-xs">
                              <span style={{ color: r.coverage_ratio != null ? scoreToColor(1 - r.coverage_ratio) : '#a1a1aa' }}>
                                {pct(r.coverage_ratio)}
                              </span>
                            </td>
                            <td className="px-4 py-3 text-right font-mono tabular-nums text-xs text-[color:var(--color-fg-muted)]">
                              {r.years_underfunded != null ? `${r.years_underfunded}/${r.n_coverage_years}` : '—'}
                            </td>
                            <td className="px-4 py-3 text-xs text-[color:var(--color-fg-muted)] leading-relaxed max-w-xs">{r.explanation}</td>
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
            inform_severity: '#f43f5e', mismatch_score: '#dc2626',
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
              className="px-3 py-2.5 text-right cursor-pointer select-none hover:text-[color:var(--color-fg)] transition-colors font-normal"
              onClick={() => {
                if (aSortBy === key) setASortDir(d => (d === 1 ? -1 : 1) as 1|-1)
                else { setASortBy(key); setASortDir(-1) }
              }}>
              {label.split(' ').slice(0,2).join('\u00a0')}
              <span className="ml-0.5 text-[color:var(--color-fg-subtle)]">{aSortBy === key ? (aSortDir === -1 ? '↓' : '↑') : '↕'}</span>
            </th>
          )

          return (
          <div className="flex flex-col gap-5 max-w-5xl mx-auto w-full">

            {/* ── Prompt card: glass floating over a soft radial backdrop ─── */}
            <div className="relative rounded-[32px] p-2"
              style={{
                background: 'radial-gradient(ellipse 80% 100% at 50% 0%, rgba(13,148,136,0.10) 0%, rgba(13,148,136,0.04) 35%, transparent 75%)',
              }}>
              <div className="glass rounded-[28px] p-8 flex flex-col gap-4">
                <p className="text-[11px] font-semibold uppercase tracking-[0.08em] text-[color:var(--color-fg-muted)]">Describe your mandate</p>
                <form onSubmit={handleAnalyze} className="flex gap-3 items-start">
                  <textarea
                    className="flex-1 bg-white/60 border border-[color:var(--color-border)] rounded-[12px] px-4 py-3 text-[color:var(--color-fg)] placeholder:text-[color:var(--color-fg-subtle)] focus:outline-none focus:border-[color:var(--color-accent)] resize-none text-sm leading-relaxed"
                    rows={2}
                    placeholder={typewriter}
                    value={aPrompt}
                    onChange={e => setAPrompt(e.target.value)}
                  />
                  <div className="flex flex-col gap-2 shrink-0 min-w-[220px]">
                    <select className="bg-white/60 border border-[color:var(--color-border)] rounded-[12px] px-3 py-2 text-[color:var(--color-fg)] text-sm"
                      value={aSector} onChange={e => setASector(e.target.value)}>
                      {SECTOR_OPTIONS.map(s => <option key={s.value} value={s.value}>{s.label}</option>)}
                    </select>
                    <input
                      className="bg-white/60 border border-[color:var(--color-border)] rounded-[12px] px-3 py-2 text-[color:var(--color-fg)] text-sm placeholder:text-[color:var(--color-fg-subtle)] focus:outline-none focus:border-[color:var(--color-accent)]"
                      placeholder="Focus country, e.g. Brazil or BRA (optional)"
                      value={aFocusRaw}
                      onChange={e => setAFocusRaw(e.target.value)}
                    />
                    <button type="submit" disabled={aLoading}
                      className="btn-primary px-4 py-2 rounded-[12px] text-sm font-medium">
                      {aLoading ? '…' : 'Analyze →'}
                    </button>
                  </div>
                </form>
              </div>
            </div>

            {aError && <div className="bg-[color:var(--color-surface-muted)] border border-[color:var(--color-border)] text-[color:var(--color-fg)] rounded-[12px] px-4 py-3 text-sm">{aError}</div>}
            {aLoading && <div className="flex items-center justify-center py-16 text-[color:var(--color-fg-subtle)] text-sm">Scoring all countries against your priorities…</div>}

            {aResult && (
              <>
                {/* ── Priority tags + refine ───────────────────────────── */}
                <div className="bg-[color:var(--color-surface)] border border-[color:var(--color-border)] rounded-[20px] p-7">
                  <div className="flex items-center justify-between mb-3">
                    <p className="text-xs text-[color:var(--color-fg-muted)] uppercase tracking-wide font-medium">How I understood your priorities</p>
                    <span className="text-xs text-[color:var(--color-fg-subtle)] font-mono">score = Σ(weight × dim) &nbsp;·&nbsp; weight = imp²/Σ(imp²)</span>
                  </div>
                  <div className="grid grid-cols-2 md:grid-cols-3 gap-x-8 gap-y-4 mb-4">
                    {(dimOrder.length > 0 ? dimOrder : Object.keys(aResult.importance_scores))
                      .filter(k => k in aResult.importance_scores)
                      .map(k => {
                        const s = aResult.importance_scores[k]
                        const meta = DIM_META[k]
                        const local = pendingScores[k] ?? s
                        const color = DIM_COLORS[k] ?? '#a1a1aa'
                        return (
                          <div key={k} className="flex flex-col gap-1"
                            title={meta ? `${meta.label} — ${meta.desc}\nSource: ${meta.source}` : k}>
                            <div className="flex items-baseline justify-between">
                              <span className="text-xs font-medium text-[color:var(--color-fg)] flex items-center gap-1.5">
                                <span className="w-2 h-2 rounded-full inline-block" style={{ background: color }} />
                                {dimLabel(k)}
                              </span>
                              <span className="text-xs font-mono tabular-nums text-[color:var(--color-fg-muted)]">{local}/10</span>
                            </div>
                            <input
                              type="range"
                              min={0}
                              max={10}
                              step={1}
                              value={local}
                              disabled={aLoading}
                              onChange={e => setPendingScores(p => ({ ...p, [k]: Number(e.target.value) }))}
                              onPointerUp={() => { if (local !== s) handleAdjustScore(k, local - s) }}
                              onKeyUp={e => { if (e.key === 'Enter' || e.key === 'ArrowLeft' || e.key === 'ArrowRight') { if (local !== s) handleAdjustScore(k, local - s) } }}
                              className="dim-slider"
                              style={{
                                ['--slider-color' as string]: color,
                                background: `linear-gradient(to right, ${color} 0%, ${color} ${local * 10}%, var(--color-border) ${local * 10}%, var(--color-border) 100%)`,
                              }}
                            />
                          </div>
                        )
                      })}
                  </div>

                  {/* ── Weight distribution bar ──────────────────────────── */}
                  {(() => {
                    const sorted = Object.entries(aResult.weights).sort((a, b) => b[1] - a[1])
                    const total = sorted.reduce((s, [, w]) => s + w, 0) || 1
                    return (
                      <div className="mb-4 p-3 bg-[color:var(--color-surface-muted)] rounded-[12px]">
                        <div className="flex items-center justify-between mb-2">
                          <p className="text-xs text-[color:var(--color-fg-subtle)]">How weight distributes across your 12 dimensions</p>
                          <p className="text-xs text-[color:var(--color-fg-subtle)] font-mono">weight = imp² / Σ(imp²)</p>
                        </div>
                        <div className="flex h-3 rounded-full overflow-hidden gap-px mb-2">
                          {sorted.map(([k, w]) => (
                            <div key={k}
                              title={`${DIM_META[k]?.label ?? k}: ${(w / total * 100).toFixed(1)}% weight\n${DIM_META[k]?.desc ?? ''}`}
                              style={{ width: `${w / total * 100}%`, background: DIM_COLORS[k] ?? '#a1a1aa', minWidth: 1 }} />
                          ))}
                        </div>
                        <div className="flex flex-wrap gap-x-3 gap-y-0.5 mb-2">
                          {sorted.filter(([, w]) => w / total >= 0.04).map(([k, w]) => (
                            <span key={k} className="text-xs text-[color:var(--color-fg-muted)] flex items-center gap-1"
                              title={DIM_META[k]?.desc}>
                              <span className="w-2 h-2 rounded-[6px] shrink-0 inline-block"
                                style={{ background: DIM_COLORS[k] ?? '#a1a1aa' }} />
                              {dimLabel(k)} <span className="font-mono text-[color:var(--color-fg-subtle)]">{(w / total * 100).toFixed(0)}%</span>
                            </span>
                          ))}
                        </div>
                        {/* Math explainer */}
                        <div className="border-t border-[color:var(--color-border)]/50 pt-2 mt-1 text-xs text-[color:var(--color-fg-subtle)] leading-relaxed space-y-0.5">
                          <p><span className="text-[color:var(--color-fg-muted)] font-medium">Importance (1–10)</span> — your stated priority. Squaring before normalising means a score of 10 carries <span className="font-mono text-[color:var(--color-fg-muted)]">10²=100</span> times the weight of a score of 1, so your top picks truly dominate.</p>
                          <p><span className="text-[color:var(--color-fg-muted)] font-medium">Dimension score (0–1)</span> — each country's raw value, where 1 = most critical globally on that metric.</p>
                          <p><span className="text-[color:var(--color-fg-muted)] font-medium">Contribution</span> = weight × dimension score. The final MCDA score is the sum of all contributions.</p>
                        </div>
                      </div>
                    )
                  })()}

                  {/* ── Dimension glossary ─────────────────────────────── */}
                  <div className="mb-3">
                    <button onClick={() => setShowGlossary(v => !v)}
                      className="text-xs text-[color:var(--color-fg-subtle)] hover:text-[color:var(--color-fg-muted)] flex items-center gap-1 transition-colors">
                      <span>{showGlossary ? '▾' : '▸'}</span>
                      What do these 12 metrics measure?
                    </button>
                    {showGlossary && (
                      <div className="mt-2 rounded-[12px] overflow-hidden border border-[color:var(--color-border)]">
                        <table className="w-full text-xs">
                          <thead className="bg-[color:var(--color-surface-muted)]">
                            <tr>
                              <th className="px-3 py-2 text-left text-[color:var(--color-fg-muted)] font-medium w-8"></th>
                              <th className="px-3 py-2 text-left text-[color:var(--color-fg-muted)] font-medium">Metric</th>
                              <th className="px-3 py-2 text-left text-[color:var(--color-fg-muted)] font-medium">What it measures</th>
                              <th className="px-3 py-2 text-left text-[color:var(--color-fg-muted)] font-medium hidden sm:table-cell">Source</th>
                            </tr>
                          </thead>
                          <tbody className="divide-y divide-[color:var(--color-border)]">
                            {Object.entries(DIM_META).map(([k, m]) => (
                              <tr key={k} className="hover:bg-[color:var(--color-surface-muted)]/30">
                                <td className="px-3 py-2">
                                  <span className="w-2.5 h-2.5 rounded-[6px] inline-block"
                                    style={{ background: DIM_COLORS[k] ?? '#a1a1aa' }} />
                                </td>
                                <td className="px-3 py-2">
                                  <div className="font-medium text-[color:var(--color-fg)]">{m.label}</div>
                                  <div className="text-[color:var(--color-fg-subtle)]">{m.group}</div>
                                </td>
                                <td className="px-3 py-2 text-[color:var(--color-fg-muted)] leading-relaxed max-w-xs">{m.desc}</td>
                                <td className="px-3 py-2 text-[color:var(--color-fg-subtle)] hidden sm:table-cell">{m.source}</td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    )}
                  </div>

                  <form onSubmit={handleRefine} className="flex gap-2">
                    <input
                      className="flex-1 bg-[color:var(--color-surface-muted)] border border-[color:var(--color-border)] rounded-[12px] px-3 py-2 text-sm text-[color:var(--color-fg)] placeholder:text-[color:var(--color-fg-subtle)] focus:outline-none focus:border-[color:var(--color-accent)]"
                      placeholder="Not right? Refine: e.g. make water more important, I don't care about governance"
                      value={aRefine}
                      onChange={e => setARefine(e.target.value)}
                    />
                    <button type="submit" disabled={aLoading || !aRefine.trim()}
                      className="bg-gray-700 hover:bg-gray-600 disabled:opacity-40 text-[color:var(--color-fg)] px-4 py-2 rounded-[12px] text-sm font-medium whitespace-nowrap">
                      {aLoading ? '…' : 'Refine ↩'}
                    </button>
                  </form>
                </div>

                {/* ── Inverse query + compare controls ─────────────────── */}
                <div className="flex gap-3 items-center">
                  <input
                    className="flex-1 bg-[color:var(--color-surface)] border border-[color:var(--color-border)] rounded-[12px] px-3 py-2 text-sm text-[color:var(--color-fg)] placeholder:text-[color:var(--color-fg-subtle)] focus:outline-none focus:border-[color:var(--color-accent)]"
                    placeholder="Find a country — type name or ISO3 to see where it ranks…"
                    value={aSearch}
                    onChange={e => setASearch(e.target.value)}
                  />
                  {aCompare.size > 0 && (
                    <button onClick={() => setACompare(new Set())}
                      className="text-xs text-[color:var(--color-fg-muted)] hover:text-[color:var(--color-fg)] border border-[color:var(--color-border)] rounded-[20px] px-3 py-2 whitespace-nowrap">
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
                    <div className="text-xs text-[color:var(--color-fg-subtle)] px-1">No country found matching "{aSearch}".</div>
                  )
                  const [iso, rank] = match
                  const name = aResult.ranked.find(r => r.country_iso3 === iso)?.country_name ?? iso
                  return (
                    <div className="bg-[color:var(--color-surface)] border border-[color:var(--color-border)] rounded-[20px] px-4 py-3 text-sm flex items-center justify-between">
                      <span className="text-[color:var(--color-fg)]">
                        <span className="font-semibold text-[color:var(--color-fg)]">{name}</span> ranks
                        <span className="font-mono text-[color:var(--color-fg-muted)] mx-1">#{rank}</span>
                        overall — outside the top 20 for your current priorities.
                      </span>
                      <button onClick={() => handleFocusCountry(iso)}
                        className="text-xs text-[color:var(--color-accent)] hover:text-[color:var(--color-accent)] ml-4 whitespace-nowrap">
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
                    <div className="bg-[color:var(--color-surface)] border border-[color:var(--color-border)] rounded-[20px] px-5 py-4 flex items-start gap-3">
                      <span className="text-[color:var(--color-fg-muted)] text-lg shrink-0">⚠</span>
                      <div>
                        <p className="text-sm font-semibold text-[color:var(--color-fg)] mb-1">
                          "{label}" is not in the crisis dataset
                        </p>
                        <p className="text-xs text-[color:var(--color-fg-muted)] leading-relaxed">
                          {proConError
                            ? proConError
                            : `Could not match "${label}" to a country with active OCHA/FTS data.`}
                          {' '}This tool covers countries with active HNO or HRP plans tracked by the Financial Tracking Service.
                          Countries without an active humanitarian response plan (e.g. middle-income countries, non-crisis contexts)
                          are not included.
                        </p>
                        <p className="text-xs text-[color:var(--color-fg-subtle)] mt-1.5">
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
                    ? { label: 'Strong case', color: 'bg-[color:var(--color-accent-bg)] border-[color:var(--color-accent)] text-[color:var(--color-accent-hover)]', dot: 'bg-[color:var(--color-accent)]' }
                    : rankPct <= 0.40
                    ? { label: 'Mixed case', color: 'bg-amber-900/50 border-amber-700 text-[color:var(--color-fg)]', dot: 'bg-[color:var(--color-fg-muted)]' }
                    : { label: 'Weak case', color: 'bg-[color:var(--color-surface-muted)] border-[color:var(--color-border-strong)] text-[color:var(--color-fg)]', dot: 'bg-[color:var(--color-fg-muted)]' }

                  // Better alternatives: ranked countries above this one (already in top-N list)
                  const betterAlts = aResult.ranked
                    .filter(r => r.mcda_rank < pc.mcda_rank && r.country_iso3 !== pc.country_iso3)
                    .sort((a, b) => a.mcda_rank - b.mcda_rank)
                    .slice(0, 3)

                  return (
                    <div className="bg-[color:var(--color-surface)] border border-[color:var(--color-border)] rounded-[20px] overflow-hidden">
                      {/* Header */}
                      <div className="px-5 py-3 border-b border-[color:var(--color-border)] flex items-center justify-between bg-[color:var(--color-surface-muted)]">
                        <div className="flex items-center gap-3 flex-wrap">
                          <span className="text-sm font-semibold text-[color:var(--color-fg)]">{pc.country_name}</span>
                          <span className="text-xs text-[color:var(--color-fg-muted)]">{pc.continent} · {pc.region_name}</span>
                          <span className="font-mono text-xs bg-[color:var(--color-surface-muted)] px-2 py-0.5 rounded text-[color:var(--color-fg)]">
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
                          className="text-xs text-[color:var(--color-fg-subtle)] hover:text-[color:var(--color-fg-muted)] shrink-0">✕ clear</button>
                      </div>

                      <div className="p-5 flex flex-col gap-5">

                        {/* Key metrics row */}
                        <div className="grid grid-cols-4 gap-2">
                          {[
                            { l: 'People in need', v: millions(pc.pin), c: 'text-[color:var(--color-fg)]' },
                            { l: 'Funded', v: pct(pc.coverage_ratio), c: 'text-[color:var(--color-accent)]' },
                            { l: 'Requirements', v: `$${millions(pc.requirements_usd)}`, c: 'text-[color:var(--color-fg)]' },
                            { l: 'Yrs underfunded', v: pc.years_underfunded != null ? String(pc.years_underfunded) : '—', c: 'text-[color:var(--color-accent-hover)]' },
                          ].map(m => (
                            <div key={m.l} className="bg-[color:var(--color-surface-muted)] rounded-[12px] p-2.5 text-center">
                              <div className={`text-sm font-bold ${m.c}`}>{m.v}</div>
                              <div className="text-xs text-[color:var(--color-fg-subtle)] mt-0.5">{m.l}</div>
                            </div>
                          ))}
                        </div>

                        {/* Why fund / Why not — split LLM narrative */}
                        <div className="grid grid-cols-2 gap-3">
                          <div className="bg-[color:var(--color-accent-bg)]/20 border border-[color:var(--color-accent)]/40 rounded-[12px] p-4">
                            <p className="text-xs font-bold text-[color:var(--color-accent)] mb-2 flex items-center gap-1.5">
                              <span>✓</span> Why you should fund {pc.country_name}
                            </p>
                            <p className="text-xs text-[color:var(--color-fg)] leading-relaxed">{pc.why_fund}</p>
                          </div>
                          <div className="bg-[color:var(--color-surface-muted)]/20 border border-red-900/40 rounded-[12px] p-4">
                            <p className="text-xs font-bold text-[color:var(--color-fg)] mb-2 flex items-center gap-1.5">
                              <span>✗</span> Why you might not
                            </p>
                            <p className="text-xs text-[color:var(--color-fg)] leading-relaxed">{pc.why_not}</p>
                          </div>
                        </div>

                        {/* Better alternatives (when weak/mixed case and we have ranked alternatives) */}
                        {betterAlts.length > 0 && rankPct > 0.15 && (
                          <div className="bg-[color:var(--color-surface-muted)] rounded-[12px] px-4 py-3 flex items-center gap-3 flex-wrap">
                            <span className="text-xs text-[color:var(--color-fg-muted)] shrink-0">
                              Stronger alternatives for your mandate:
                            </span>
                            {betterAlts.map(r => (
                              <button key={r.country_iso3}
                                onClick={() => handleFocusCountry(r.country_iso3)}
                                className="text-xs bg-gray-700 hover:bg-gray-600 border border-gray-600 rounded-[12px] px-3 py-1.5 text-[color:var(--color-fg)] transition-colors flex items-center gap-1.5">
                                <span className="font-mono text-[color:var(--color-fg-muted)]">#{r.mcda_rank}</span>
                                {r.country_name}
                              </button>
                            ))}
                          </div>
                        )}

                        {/* ═══ DIMENSION DETAIL — strengths / weaknesses (flat) ═══ */}
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 items-stretch">

                          {/* ── STRENGTHS ───────────────────────────────────── */}
                          <div className="bg-[color:var(--color-surface)] border border-[color:var(--color-border)] rounded-[20px] overflow-hidden flex flex-col"
                            style={{ borderLeft: '3px solid var(--color-accent)' }}>
                            <div className="px-4 py-2.5 flex items-baseline justify-between border-b border-[color:var(--color-border)] bg-[color:var(--color-accent-bg)]">
                              <p className="text-[11px] font-semibold uppercase tracking-[0.08em] text-[color:var(--color-accent-hover)]">Strengths on your criteria</p>
                              <p className="text-[11px] text-[color:var(--color-fg-subtle)] font-mono">{pc.pros.length} dims</p>
                            </div>
                            {pc.pros.length === 0 ? (
                              <p className="px-4 py-4 text-xs text-[color:var(--color-fg-subtle)] italic">No strong signals on your weighted dimensions.</p>
                            ) : (
                              <ul className="divide-y divide-[color:var(--color-border)]">
                                {[...pc.pros]
                                  .sort((a, b) => (b.weight * (b.value ?? 0)) - (a.weight * (a.value ?? 0)))
                                  .slice(0, 4)
                                  .map(c => (
                                    <li key={c.dimension} className="px-4 py-3">
                                      <div className="flex items-center gap-2">
                                        <span className="shrink-0 text-sm leading-none text-[color:var(--color-accent)]">▲</span>
                                        <span className="text-sm font-semibold text-[color:var(--color-fg)]">{c.label}</span>
                                        <span className="ml-auto shrink-0 text-[11px] font-semibold uppercase tracking-[0.08em] font-mono text-[color:var(--color-accent-hover)]">
                                          top {Math.round((1 - c.percentile) * 100)}%
                                        </span>
                                      </div>
                                      <p className="text-xs text-[color:var(--color-fg-muted)] leading-relaxed mt-1 pl-5">{c.narrative}</p>
                                    </li>
                                  ))}
                              </ul>
                            )}
                          </div>

                          {/* ── WEAKNESSES ──────────────────────────────────── */}
                          <div className="bg-[color:var(--color-surface)] border border-[color:var(--color-border)] rounded-[20px] overflow-hidden flex flex-col"
                            style={{ borderRight: '3px solid #dc2626' }}>
                            <div className="px-4 py-2.5 flex items-baseline justify-between border-b border-[color:var(--color-border)] bg-[#fef2f2]">
                              <p className="text-[11px] font-semibold uppercase tracking-[0.08em]" style={{ color: '#b91c1c' }}>Weaknesses on your criteria</p>
                              <p className="text-[11px] text-[color:var(--color-fg-subtle)] font-mono">{pc.cons.length} dims</p>
                            </div>
                            {pc.cons.length === 0 ? (
                              <p className="px-4 py-4 text-xs text-[color:var(--color-fg-subtle)] italic">No significant weaknesses on your weighted dimensions.</p>
                            ) : (
                              <ul className="divide-y divide-[color:var(--color-border)]">
                                {[...pc.cons]
                                  .sort((a, b) => (b.weight * (1 - (b.value ?? 1))) - (a.weight * (1 - (a.value ?? 1))))
                                  .slice(0, 4)
                                  .map(c => (
                                    <li key={c.dimension} className="px-4 py-3">
                                      <div className="flex items-center gap-2">
                                        <span className="shrink-0 text-sm leading-none" style={{ color: '#dc2626' }}>▼</span>
                                        <span className="text-sm font-semibold text-[color:var(--color-fg)]">{c.label}</span>
                                        <span className="ml-auto shrink-0 text-[11px] font-semibold uppercase tracking-[0.08em] font-mono" style={{ color: '#b91c1c' }}>
                                          btm {Math.round(c.percentile * 100)}%
                                        </span>
                                      </div>
                                      <p className="text-xs text-[color:var(--color-fg-muted)] leading-relaxed mt-1 pl-5">{c.narrative}</p>
                                      {c.better_country && (
                                        <p className="text-xs text-[color:var(--color-fg-subtle)] mt-1 pl-5">
                                          <span className="font-medium text-[color:var(--color-fg-muted)]">{c.better_country.country_name}</span> leads here
                                          {c.better_country.value != null && c.value != null
                                            ? <> (<span className="font-mono tabular-nums">{(c.better_country.value * 10).toFixed(1)}</span> vs <span className="font-mono tabular-nums">{(c.value * 10).toFixed(1)}</span>)</>
                                            : null}.
                                        </p>
                                      )}
                                    </li>
                                  ))}
                              </ul>
                            )}
                          </div>
                        </div>

                        {/* ── Score decomposition (technical breakdown) ─── */}
                        <div className="pt-3">
                          <p className="text-[11px] font-medium uppercase tracking-[0.06em] text-[color:var(--color-fg-muted)] mb-3">
                            Score composition · all 12 dimensions
                          </p>

                          {/* Score decomposition */}
                          <p className="text-xs text-[color:var(--color-fg-subtle)] font-medium mb-1.5">
                            Score breakdown — {pc.country_name} · <span className="font-mono text-[color:var(--color-fg)]">{pc.mcda_score.toFixed(3)}</span>
                          </p>
                          <div className="flex items-center gap-2 mb-1 px-0.5">
                            <span className="text-xs text-[color:var(--color-fg-subtle)] w-24 shrink-0">Dimension</span>
                            <span className="text-xs text-[color:var(--color-fg-subtle)] w-8 shrink-0 text-center">Imp</span>
                            <span className="text-xs text-[color:var(--color-fg-subtle)] flex-1 text-center">Contribution</span>
                            <span className="text-xs text-[color:var(--color-fg-subtle)] w-8 shrink-0 text-right">Wt%</span>
                            <span className="text-xs text-[color:var(--color-fg-subtle)] w-8 shrink-0 text-right">Dim</span>
                            <span className="text-xs text-[color:var(--color-fg-subtle)] w-8 shrink-0 text-right">Ctb</span>
                          </div>
                          <div className="flex flex-col gap-1.5">
                            {dims.map(d => {
                              const impColor = d.importance >= 8 ? 'text-[color:var(--color-fg)] bg-[color:var(--color-surface-muted)] border-red-900'
                                : d.importance >= 6 ? 'text-[color:var(--color-fg)] bg-[color:var(--color-surface-muted)] border-[color:var(--color-border-strong)]'
                                : d.importance >= 4 ? 'text-[color:var(--color-accent)] bg-[color:var(--color-accent-bg)] border-[color:var(--color-accent)]'
                                : 'text-[color:var(--color-fg-subtle)] bg-[color:var(--color-surface-muted)] border-[color:var(--color-border)]'
                              return (
                                <div key={d.k} className="flex items-center gap-2">
                                  <span className="text-xs text-[color:var(--color-fg-muted)] w-24 shrink-0 truncate" title={dimLabel(d.k)}>
                                    {dimLabel(d.k)}
                                  </span>
                                  <span className={`text-xs font-mono font-bold w-8 shrink-0 text-center rounded border px-1 ${impColor}`}>
                                    {d.importance}
                                  </span>
                                  <div className="flex-1 h-1.5 bg-[color:var(--color-surface-muted)] rounded-full overflow-hidden">
                                    <div className="h-full rounded-full"
                                      style={{ width: `${(d.contrib / maxC) * 100}%`, background: DIM_COLORS[d.k] ?? '#a1a1aa' }} />
                                  </div>
                                  <span className="text-xs font-mono text-[color:var(--color-fg-subtle)] w-8 shrink-0 text-right">{(d.w * 100).toFixed(0)}%</span>
                                  <span className="text-xs font-mono text-[color:var(--color-fg-subtle)] w-8 shrink-0 text-right">{d.val != null ? (d.val * 10).toFixed(1) : '—'}</span>
                                  <span className="text-xs font-mono text-[color:var(--color-fg-muted)] w-8 shrink-0 text-right font-semibold">{d.contrib > 0 ? (d.contrib * 100).toFixed(1) : '—'}</span>
                                </div>
                              )
                            })}
                          </div>
                          <p className="text-xs text-[color:var(--color-fg-subtle)] mt-1.5">Imp = importance (1–10) · Wt% = derived weight · Dim = raw score ×10 · Ctb = contribution ×100</p>
                        </div>
                      </div>
                    </div>
                  )
                })()}

                {/* ── Ranked table ──────────────────────────────────────── */}
                <div className="bg-[color:var(--color-surface)] border border-[color:var(--color-border)] rounded-[20px] overflow-hidden">
                  <div className="px-5 py-3 border-b border-[color:var(--color-border)] flex items-center justify-between">
                    <div>
                      <p className="text-sm font-semibold text-[color:var(--color-fg)]">
                        Rankings — {aResult.year}
                        {aResult.sector !== 'ALL' && (
                          <span className="ml-2 text-xs font-normal text-[color:var(--color-accent)] bg-[color:var(--color-accent-bg)] px-2 py-0.5 rounded-full">
                            {SECTOR_OPTIONS.find(s => s.value === aResult.sector)?.label ?? aResult.sector}
                          </span>
                        )}
                      </p>
                      <p className="text-xs text-[color:var(--color-fg-subtle)] mt-0.5">Sort by header · click row to analyze · checkbox to compare</p>
                    </div>
                    {aCompare.size >= 2 && (
                      <span className="text-xs text-[color:var(--color-accent)] border border-[color:var(--color-border)] bg-[color:var(--color-accent-bg)] px-2 py-1 rounded-[12px]">
                        {aCompare.size} selected for comparison ↓
                      </span>
                    )}
                  </div>
                  <div className="overflow-x-auto">
                    <table className="w-full text-sm">
                      <thead className="text-[color:var(--color-fg-muted)] text-xs border-b border-[color:var(--color-border)]">
                        <tr>
                          <th className="px-3 py-2.5 w-8">
                            <input type="checkbox" className="accent-[color:var(--color-accent)]" title="Select all"
                              checked={aCompare.size === sortedRanked.length}
                              onChange={e => setACompare(e.target.checked
                                ? new Set(sortedRanked.map(r => r.country_iso3)) : new Set())} />
                          </th>
                          <th className="px-3 py-2.5 text-left w-8 cursor-pointer hover:text-[color:var(--color-fg)]"
                            onClick={() => { setASortBy('mcda_rank'); setASortDir(1) }}>#</th>
                          <th className="px-3 py-2.5 text-left">Country</th>
                          {thSort('mcda_score', 'Score')}
                          {thSort('rank_delta', 'Δ vs prev yr')}
                          {topDims.map(k => thSort(k, dimLabel(k)))}
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-[color:var(--color-border)]/40">
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
                                  className={`transition-colors ${selected ? 'bg-[color:var(--color-accent-bg)]' : inCompare ? 'bg-[color:var(--color-accent-bg)]' : 'hover:bg-[color:var(--color-surface-muted)]'}`}>
                                  <td className="px-3 py-3">
                                    <input type="checkbox" className="accent-[color:var(--color-accent)]"
                                      checked={inCompare}
                                      onChange={e => {
                                        const s = new Set(aCompare)
                                        e.target.checked ? s.add(r.country_iso3) : s.delete(r.country_iso3)
                                        setACompare(s)
                                      }} />
                                  </td>
                                  <td className="px-3 py-3 text-xs text-[color:var(--color-fg-muted)] font-mono cursor-pointer"
                                    onClick={() => selected ? (setACountry(''), setAFocusRaw('')) : handleFocusCountry(r.country_iso3)}>
                                    {i < 3 && aSortBy === 'mcda_rank' ? ['🥇','🥈','🥉'][i] : r.mcda_rank}
                                  </td>
                                  <td className="px-3 py-3 cursor-pointer"
                                    onClick={() => selected ? (setACountry(''), setAFocusRaw('')) : handleFocusCountry(r.country_iso3)}>
                                    <div className="flex items-center gap-1.5">
                                      <div>
                                        <div className="text-xs font-medium text-[color:var(--color-fg)]">{r.country_name}</div>
                                        <div className="text-xs text-[color:var(--color-fg-subtle)]">{r.continent}</div>
                                      </div>
                                      {!r.has_vulnerability && (
                                        <span title="Vulnerability scores estimated (no source data)" className="text-[color:var(--color-fg-subtle)] text-xs border border-[color:var(--color-border)] rounded px-1">~vuln</span>
                                      )}
                                    </div>
                                  </td>
                                  <td className="px-3 py-3 text-right cursor-pointer"
                                    onClick={() => selected ? (setACountry(''), setAFocusRaw('')) : handleFocusCountry(r.country_iso3)}>
                                    <div className="flex items-center justify-end gap-1.5">
                                      <div className="w-14 h-1.5 bg-[color:var(--color-surface-muted)] rounded-full overflow-hidden">
                                        <div className="h-full rounded-full" style={{ width:`${r.mcda_score*100}%`, background: scoreToColor(r.mcda_score) }} />
                                      </div>
                                      <span className="text-xs font-mono font-semibold" style={{ color: scoreToColor(r.mcda_score) }}>
                                        {r.mcda_score.toFixed(2)}
                                      </span>
                                    </div>
                                  </td>
                                  <td className="px-3 py-3 text-right text-xs font-mono">
                                    {delta == null ? <span className="text-[color:var(--color-fg-subtle)]">—</span>
                                      : delta > 0 ? <span className="text-[color:var(--color-accent)]">↑{Math.abs(Math.round(delta))}</span>
                                      : delta < 0 ? <span className="text-[color:var(--color-fg)]">↓{Math.abs(Math.round(delta))}</span>
                                      : <span className="text-[color:var(--color-fg-subtle)]">→</span>}
                                  </td>
                                  {topDims.map(k => {
                                    const v = r[k] as number | null
                                    return (
                                      <td key={k} className="px-3 py-3 text-right">
                                        {v != null
                                          ? <span className={`text-xs font-mono ${v >= 0.7 ? 'text-[color:var(--color-fg)]' : v >= 0.4 ? 'text-[color:var(--color-fg)]' : 'text-[color:var(--color-fg-muted)]'}`}>{(v*10).toFixed(1)}</span>
                                          : <span className="text-[color:var(--color-fg-subtle)] text-xs">—</span>}
                                      </td>
                                    )
                                  })}
                                </tr>
                                {selected && (
                                  <tr key={`${r.country_iso3}-loading`}>
                                    <td colSpan={5 + topDims.length} className="px-5 py-2 bg-[color:var(--color-surface-muted)] border-b border-[color:var(--color-border)]">
                                      <span className="text-xs text-[color:var(--color-accent)]">
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
                    <div className="bg-[color:var(--color-surface)] border border-[color:var(--color-border)] rounded-[20px] overflow-hidden">
                      <div className="px-5 py-3 border-b border-[color:var(--color-border)] flex items-center justify-between">
                        <p className="text-sm font-semibold text-[color:var(--color-fg)]">Side-by-side comparison</p>
                        <button onClick={() => setACompare(new Set())} className="text-xs text-[color:var(--color-fg-subtle)] hover:text-[color:var(--color-fg-muted)]">Clear</button>
                      </div>
                      <div className="overflow-x-auto">
                        <table className="w-full text-xs">
                          <thead className="border-b border-[color:var(--color-border)]">
                            <tr>
                              <th className="px-4 py-2.5 text-left text-[color:var(--color-fg-subtle)] font-normal w-40">Dimension</th>
                              {compared.map(r => (
                                <th key={r.country_iso3} className="px-4 py-2.5 text-center font-semibold text-[color:var(--color-fg)]">
                                  {r.country_name}
                                  <div className="text-[color:var(--color-fg-subtle)] font-normal">#{r.mcda_rank}</div>
                                </th>
                              ))}
                            </tr>
                          </thead>
                          <tbody className="divide-y divide-[color:var(--color-border)]/40">
                            <tr className="bg-[color:var(--color-surface-muted)]/30">
                              <td className="px-4 py-2 text-[color:var(--color-fg-muted)] font-medium">Overall Score</td>
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
                                <tr key={dim} className={importance >= 7 ? 'bg-[color:var(--color-surface-muted)]' : ''}>
                                  <td className="px-4 py-2 text-[color:var(--color-fg-muted)]">
                                    {dimLabel(dim)}
                                    {importance >= 7 && <span className="ml-1 text-[color:var(--color-accent)] text-xs">★</span>}
                                  </td>
                                  {compared.map((r, ci) => {
                                    const v = vals[ci]
                                    const isMax = v != null && v === maxVal
                                    return (
                                      <td key={r.country_iso3} className="px-4 py-2 text-center">
                                        {v != null ? (
                                          <span className={`font-mono ${isMax ? 'text-[color:var(--color-fg)] font-bold' : 'text-[color:var(--color-fg-muted)]'}`}>
                                            {(v * 10).toFixed(1)}
                                          </span>
                                        ) : <span className="text-[color:var(--color-fg-subtle)]">—</span>}
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
                  <div className="bg-[color:var(--color-surface)] border border-[color:var(--color-border)] rounded-[20px] overflow-hidden">
                    <div className="px-5 py-3 border-b border-[color:var(--color-border)]">
                      <p className="text-sm font-semibold text-[color:var(--color-fg)]">Who leads each metric</p>
                      <p className="text-xs text-[color:var(--color-fg-subtle)] mt-0.5">Click to expand pro/con for that country</p>
                    </div>
                    <table className="w-full text-xs">
                      <thead className="text-[color:var(--color-fg-subtle)] border-b border-[color:var(--color-border)]">
                        <tr>
                          <th className="px-4 py-2 text-left font-normal">Metric</th>
                          <th className="px-4 py-2 text-left font-normal">Top country</th>
                          <th className="px-4 py-2 text-right font-normal">Score</th>
                          <th className="px-4 py-2 text-right font-normal">Overall rank</th>
                          <th className="px-4 py-2 text-right font-normal">Weight</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-[color:var(--color-border)]/40">
                        {[...aResult.dimension_leaders]
                          .sort((a, b) => b.weight - a.weight)
                          .map(d => (
                            <tr key={d.dimension}
                              className="hover:bg-[color:var(--color-surface-muted)] cursor-pointer transition-colors"
                              onClick={() => setACountry(d.country_iso3)}>
                              <td className="px-4 py-2.5">
                                <span className="text-[color:var(--color-fg)] font-medium">{d.label}</span>
                                <span className="text-[color:var(--color-fg-subtle)] ml-2">{d.group}</span>
                              </td>
                              <td className="px-4 py-2.5 text-[color:var(--color-fg)]">{d.country_name}</td>
                              <td className="px-4 py-2.5 text-right font-mono text-[color:var(--color-fg)]">
                                {d.value != null ? (d.value*10).toFixed(1) : '—'}
                              </td>
                              <td className="px-4 py-2.5 text-right text-[color:var(--color-fg-muted)] font-mono">#{d.mcda_rank}</td>
                              <td className="px-4 py-2.5 text-right">
                                <span className={`font-mono ${d.weight >= 0.1 ? 'text-[color:var(--color-accent)]' : 'text-[color:var(--color-fg-subtle)]'}`}>
                                  {(d.weight*100).toFixed(0)}%
                                </span>
                              </td>
                            </tr>
                          ))}
                      </tbody>
                    </table>
                  </div>
                )}

                {/* ── Severity vs Funding scatter ──────────────────────── */}
                {(() => {
                  const CONT_COLOR: Record<string, string> = {
                    'Africa':               '#f97316',
                    'Middle East':          '#ef4444',
                    'Asia':                 '#3b82f6',
                    'Latin America':        '#10b981',
                    'Europe & Central Asia':'#a855f7',
                  }
                  const pts = aResult.ranked
                    .filter(r => r.inform_severity != null && r.coverage_ratio != null)
                    .map(r => ({
                      x:        Math.round((r.coverage_ratio as number) * 100),
                      y:        Math.round(((r.inform_severity as number) * 5) * 10) / 10,
                      z:        r.pin != null ? Math.max(4, Math.sqrt((r.pin as number) / 1e5) * 6) : 4,
                      name:     r.country_name,
                      iso3:     r.country_iso3,
                      continent:r.continent ?? '',
                      mismatch: r.mismatch_score,
                      rank:     r.mcda_rank,
                      pin:      r.pin,
                    }))

                  if (pts.length < 3) return null

                  // ── Dynamic "nice" axis domains ─────────────────────────
                  const xs = pts.map(p => p.x)
                  const ys = pts.map(p => p.y)
                  const xMinD = Math.min(...xs), xMaxD = Math.max(...xs)
                  const yMinD = Math.min(...ys), yMaxD = Math.max(...ys)
                  const xPad = Math.max((xMaxD - xMinD) * 0.08, 2)
                  const yPad = Math.max((yMaxD - yMinD) * 0.08, 0.1)
                  const xLo = Math.max(0,   Math.floor((xMinD - xPad) / 10) * 10)
                  const xHiRaw = Math.min(100, Math.ceil ((xMaxD + xPad) / 10) * 10)
                  const xHi = xHiRaw === xLo ? Math.min(100, xLo + 10) : xHiRaw
                  const yLo = Math.max(0, Math.floor((yMinD - yPad) * 2) / 2)
                  const yHiRaw = Math.min(5, Math.ceil ((yMaxD + yPad) * 2) / 2)
                  const yHi = yHiRaw === yLo ? Math.min(5, yLo + 0.5) : yHiRaw
                  const xDomain: [number, number] = [xLo, xHi]
                  const yDomain: [number, number] = [yLo, yHi]

                  const xTicks: number[] = []
                  for (let v = xLo; v <= xHi + 1e-9; v += 10) xTicks.push(Math.round(v))
                  const yStep = (yHi - yLo) <= 2 ? 0.5 : 1
                  const yTicks: number[] = []
                  for (let v = yLo; v <= yHi + 1e-9; v += yStep)
                    yTicks.push(Math.round(v * 10) / 10)

                  // Median-based quadrant dividers (data-driven)
                  const median = (arr: number[]) => {
                    const s = [...arr].sort((a, b) => a - b)
                    const m = Math.floor(s.length / 2)
                    return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2
                  }
                  const X_SPLIT = median(xs)
                  const Y_SPLIT = median(ys)

                  const CustomDot = (props: any) => {
                    const { cx, cy, payload } = props
                    const isSelected = payload.iso3 === aCountry
                    const r = Math.sqrt(payload.z / Math.PI) * 6
                    return (
                      <g>
                        {isSelected && (
                          <circle cx={cx} cy={cy} r={r + 5}
                            fill="none" stroke="#facc15" strokeWidth={2.5} />
                        )}
                        <circle cx={cx} cy={cy} r={r}
                          fill={CONT_COLOR[payload.continent] ?? '#94a3b8'}
                          fillOpacity={0.75}
                          stroke={isSelected ? '#facc15' : '#fff'}
                          strokeWidth={isSelected ? 2 : 0.8}
                          style={{ cursor: 'pointer' }}
                          onClick={() => setACountry(payload.iso3)}
                        />
                        {r > 10 && (
                          <text cx={cx} cy={cy} textAnchor="middle"
                            dominantBaseline="central"
                            fontSize={9} fill="#fff" fontWeight={600}
                            style={{ pointerEvents: 'none' }}>
                            {payload.iso3}
                          </text>
                        )}
                      </g>
                    )
                  }

                  const CustomTooltip = ({ active, payload }: any) => {
                    if (!active || !payload?.length) return null
                    const d = payload[0]?.payload
                    if (!d) return null
                    return (
                      <div className="bg-[color:var(--color-surface)] border border-[color:var(--color-border)] rounded-xl px-3 py-2.5 text-xs shadow-lg min-w-[160px] max-w-[220px]">
                        <p className="font-semibold text-[color:var(--color-fg)] mb-1 whitespace-nowrap overflow-hidden text-ellipsis">{d.name}</p>
                        <p className="text-[color:var(--color-fg-subtle)]">Severity: <span className="font-mono text-[color:var(--color-fg)]">{d.y}/5</span></p>
                        <p className="text-[color:var(--color-fg-subtle)]">Coverage: <span className="font-mono text-[color:var(--color-fg)]">{d.x}%</span></p>
                        {d.pin != null && (
                          <p className="text-[color:var(--color-fg-subtle)]">PIN: <span className="font-mono text-[color:var(--color-fg)]">{(d.pin/1e6).toFixed(1)}M</span></p>
                        )}
                        {d.mismatch != null && (
                          <p className="text-[color:var(--color-fg-subtle)]">Mismatch: <span className="font-mono text-[color:var(--color-fg)]">{(d.mismatch*100).toFixed(0)}%</span></p>
                        )}
                        <p className="text-[color:var(--color-fg-subtle)] mt-1">Rank: <span className="font-mono text-[color:var(--color-fg)]">#{d.rank}</span></p>
                      </div>
                    )
                  }

                  return (
                    <div className="bg-[color:var(--color-surface)] border border-[color:var(--color-border)] rounded-[20px] overflow-hidden">
                      <div className="px-5 py-3 border-b border-[color:var(--color-border)]">
                        <p className="text-sm font-semibold text-[color:var(--color-fg)]">Severity vs. Funding Coverage</p>
                        <p className="text-xs text-[color:var(--color-fg-subtle)] mt-0.5">
                          Top-left = most overlooked · bubble size = people in need · click to analyze a country
                        </p>
                      </div>
                      <div className="p-4">
                        {/* Quadrant labels — outside the plot, above the chart */}
                        <div className="flex justify-between items-center px-2 mb-1">
                          <span className="text-[10px] font-semibold text-red-400/80 bg-red-400/10 rounded px-1.5 py-0.5">
                            Severe &amp; Overlooked ★
                          </span>
                          <span className="text-[10px] font-semibold text-emerald-400/80 bg-emerald-400/10 rounded px-1.5 py-0.5">
                            Severe &amp; Funded
                          </span>
                        </div>

                        <ResponsiveContainer width="100%" height={340}>
                          <ScatterChart margin={{ top: 10, right: 24, bottom: 32, left: 24 }}>
                            <CartesianGrid strokeDasharray="3 3" stroke="var(--color-border)" opacity={0.4} />
                            <XAxis type="number" dataKey="x" domain={xDomain} ticks={xTicks} allowDataOverflow={false}
                              tick={{ fontSize: 10, fill: 'var(--color-fg-subtle)' }}
                              tickFormatter={v => `${v}%`}>
                              <Label value="Funding Coverage →" position="insideBottom" offset={-18}
                                style={{ fontSize: 10, fill: 'var(--color-fg-subtle)' }} />
                            </XAxis>
                            <YAxis type="number" dataKey="y" domain={yDomain} ticks={yTicks} allowDataOverflow={false}
                              tick={{ fontSize: 10, fill: 'var(--color-fg-subtle)' }}
                              tickFormatter={v => `${v}`}>
                              <Label value="INFORM Severity →" angle={-90} position="insideLeft" offset={10}
                                style={{ fontSize: 10, fill: 'var(--color-fg-subtle)' }} />
                            </YAxis>
                            <ZAxis dataKey="z" range={[40, 400]} />
                            <ReTooltip content={<CustomTooltip />}
                              allowEscapeViewBox={{ x: true, y: true }}
                              wrapperStyle={{ zIndex: 50, pointerEvents: 'none' }} />
                            <ReferenceLine x={X_SPLIT} stroke="var(--color-border)" strokeDasharray="4 4" strokeWidth={1.5} />
                            <ReferenceLine y={Y_SPLIT} stroke="var(--color-border)" strokeDasharray="4 4" strokeWidth={1.5} />
                            <Scatter data={pts} shape={<CustomDot />}>
                              {pts.map((p, i) => (
                                <Cell key={i} fill={CONT_COLOR[p.continent] ?? '#94a3b8'} />
                              ))}
                            </Scatter>
                          </ScatterChart>
                        </ResponsiveContainer>

                        {/* Quadrant labels — outside the plot, below the chart */}
                        <div className="flex justify-between items-center px-2 mt-1">
                          <span className="text-[10px] font-semibold text-[color:var(--color-fg-subtle)]/70 rounded px-1.5 py-0.5">
                            Lower Risk &amp; Overlooked
                          </span>
                          <span className="text-[10px] font-semibold text-[color:var(--color-fg-subtle)]/70 rounded px-1.5 py-0.5">
                            Lower Risk &amp; Funded
                          </span>
                        </div>

                        {/* Legend */}
                        <div className="flex flex-wrap gap-x-4 gap-y-1 justify-center mt-2">
                          {Object.entries(CONT_COLOR).map(([c, col]) => (
                            <span key={c} className="flex items-center gap-1 text-[10px] text-[color:var(--color-fg-subtle)]">
                              <span className="w-2 h-2 rounded-full inline-block" style={{ background: col }} />
                              {c}
                            </span>
                          ))}
                        </div>
                      </div>
                    </div>
                  )
                })()}
              </>
            )}

          </div>
          )
        })()}

        {/* ═══ QUERY TAB ═══════════════════════════════════════════════════ */}
        {tab === 'query' && (
          <div className="flex flex-col gap-6">
            {/* Glass prompt card over soft radial backdrop — mirrors the Prioritize tab */}
            <div className="relative rounded-[32px] p-2"
              style={{
                background: 'radial-gradient(ellipse 80% 100% at 50% 0%, rgba(13,148,136,0.10) 0%, rgba(13,148,136,0.04) 35%, transparent 75%)',
              }}>
              <div className="glass rounded-[28px] p-8 flex flex-col gap-4">
                <p className="text-[11px] font-semibold uppercase tracking-[0.08em] text-[color:var(--color-fg-muted)]">Ask a question</p>
                <form onSubmit={handleQuery} className="flex flex-col gap-4">
                  <textarea
                    className="w-full bg-white/60 border border-[color:var(--color-border)] rounded-[12px] px-4 py-3 text-[color:var(--color-fg)] placeholder:text-[color:var(--color-fg-subtle)] focus:outline-none focus:border-[color:var(--color-accent)] resize-none text-sm leading-relaxed"
                    rows={2}
                    placeholder={sqlTypewriter}
                    value={query}
                    onChange={e => setQuery(e.target.value)}
                  />
                  <div className="flex items-center gap-4 flex-wrap">
                    {['from','to'].map((label, idx) => (
                      <label key={label} className="text-sm text-[color:var(--color-fg-muted)] flex items-center gap-2">
                        Year {label}
                        <select className="bg-white/60 border border-[color:var(--color-border)] rounded-[10px] px-2 py-1 text-[color:var(--color-fg)] text-sm"
                          value={idx === 0 ? yearFrom : yearTo}
                          onChange={e => idx === 0 ? setYearFrom(Number(e.target.value)) : setYearTo(Number(e.target.value))}>
                          {YEARS.map(y => <option key={y}>{y}</option>)}
                        </select>
                      </label>
                    ))}
                    <label className="text-sm text-[color:var(--color-fg-muted)] flex items-center gap-2 ml-auto">
                      <input type="checkbox" checked={showSql} onChange={e => setShowSql(e.target.checked)} className="accent-[color:var(--color-accent)]" />
                      Show SQL
                    </label>
                    <button type="submit" disabled={qLoading || !query.trim()}
                      className="btn-primary px-6 py-2 rounded-[12px] text-sm font-medium">
                      {qLoading ? 'Analysing…' : 'Analyse'}
                    </button>
                  </div>
                </form>
              </div>
            </div>

            {qError && <div className="bg-[color:var(--color-surface-muted)] border border-[color:var(--color-border)] text-[color:var(--color-fg)] rounded-[12px] px-4 py-3 text-sm">{qError}</div>}

            {qResult && (
              <div className="flex flex-col gap-6">
                {showSql && (
                  <div className="bg-[color:var(--color-surface)] border border-[color:var(--color-border)] rounded-[20px] p-4">
                    <p className="text-xs text-[color:var(--color-fg-muted)] mb-2 uppercase tracking-wide font-medium">Generated SQL</p>
                    <pre className="text-xs font-mono tabular-nums text-[color:var(--color-fg)] bg-[color:var(--color-surface-muted)] border border-[color:var(--color-border)] rounded-[10px] p-4 overflow-x-auto whitespace-pre-wrap">{qResult.sql}</pre>
                  </div>
                )}
                <p className="text-sm text-[color:var(--color-fg-muted)]">{qResult.row_count} rows returned</p>

                {/* World map for query results */}
                {Object.keys(qIsoMap).length > 0 && (
                  <div className="bg-[color:var(--color-surface)] border border-[color:var(--color-border)] rounded-[20px] p-8">
                    <p className="text-xs font-semibold text-[color:var(--color-fg-muted)] mb-3 uppercase tracking-wide">Result Countries</p>
                    <ComposableMap projectionConfig={{ scale: 153 }}>
                      <Geographies geography={GEO_URL}>
                        {({ geographies }: { geographies: { rsmKey: string; properties: Record<string,unknown> }[] }) =>
                          geographies.map(geo => {
                            const iso = geoIso3(geo) ?? ''
                            const score = qIsoMap[iso]
                            return (
                              <Geography key={geo.rsmKey} geography={geo}
                                fill={score != null ? scoreToColor(score) : '#fafafa'}
                                stroke="#e4e4e7" strokeWidth={0.5}
                                style={{ default:{outline:'none'}, hover:{fill:'#0d9488',outline:'none'}, pressed:{outline:'none'} }}
                              />
                            )
                          })
                        }
                      </Geographies>
                    </ComposableMap>
                  </div>
                )}

                {/* Results table */}
                <div className="bg-[color:var(--color-surface)] border border-[color:var(--color-border)] rounded-[20px] overflow-hidden">
                  <div className="overflow-x-auto">
                    <table className="w-full text-sm">
                      <thead className="text-[color:var(--color-fg-muted)] text-xs uppercase border-b border-[color:var(--color-border)]">
                        <tr>{qResult.columns.map(col => (
                          <th key={col} className="px-4 py-3 text-left font-medium whitespace-nowrap">
                            {COLUMN_LABELS[col] ?? col.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase())}
                          </th>
                        ))}</tr>
                      </thead>
                      <tbody className="divide-y divide-[color:var(--color-border)]">
                        {qResult.rows.map((row, i) => (
                          <tr key={i} className="hover:bg-[color:var(--color-surface-muted)] transition-colors">
                            {qResult.columns.map(col => (
                              <td key={col} className="px-4 py-2.5 text-[color:var(--color-fg)] whitespace-nowrap font-mono text-xs">
                                {row[col] == null
                                  ? <span className="text-[color:var(--color-fg-subtle)]">—</span>
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

          </div>
        )}
      </main>
    </div>
  )
}
