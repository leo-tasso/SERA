const { useEffect, useMemo, useRef, useState } = React

const MAP_WIDTH = 920
const MAP_HEIGHT = 760
const DEFAULT_INDICATORS = [
  'gdp_per_capita',
  'income',
  'unemployment_rate',
  'life_expectancy',
]
const CHART_COLORS = ['#1d4ea4', '#0f766e', '#d97706', '#be123c', '#6d28d9', '#0891b2']
const OBJECTIVE_COLORS = {
  utilitarian: '#1d4ea4',
  rawlsian: '#d97706',
  egalitarian: '#6d28d9',
  wellbeing: '#0f766e',
}
const BASELINE_SERIES_COLOR = '#94a3b8'
const CANDIDATE_COLORS = {
  full: '#0f766e',
  moderate: '#d97706',
  baseline: BASELINE_SERIES_COLOR,
}

// How auditable each policy model's *result* is (mirrors PolicyModel.explainability).
const EXPLAINABILITY_META = {
  'white-box': {
    label: 'White box',
    color: '#0f766e',
    blurb: 'The policy itself is directly readable — weights, rules, or lever tables you can audit.',
  },
  'gray-box': {
    label: 'Gray box',
    color: '#d97706',
    blurb: 'A transparent surrogate explains the policy, including its own uncertainty.',
  },
  'black-box': {
    label: 'Black box + audit',
    color: '#be123c',
    blurb: 'Not directly interpretable; audited post hoc with importance scores and a distilled tree.',
  },
}

const DIVERGING_PALETTE = ['#b61c3e', '#e9a7b7', '#f1f5f9', '#7fc1b4', '#0f766e']

function featureDisplayName(key) {
  if (key === 'bias') return 'Base level'
  if (key === 'year_position') return 'Year in horizon'
  return formatLabel(key)
}

const SPENDING_PARAMS = new Set([
  'healthcare_spending_allocation',
  'education_spending_allocation',
  'infrastructure_investment_allocation',
  'social_welfare_spending_allocation',
  'rd_innovation_incentives',
  'green_energy_environment_investment',
  'pension_retirement_spending',
  'agriculture_support_level',
  'manufacturing_incentives',
  'tourism_support_level',
  'small_business_support',
  'public_sector_wage_levels',
  'housing_urban_development_support',
])

// Revenue levers: they fund the national pool (mirrors backend_bridge.TAX_PARAMS).
const TAX_PARAMS = new Set([
  'income_tax_rate',
  'corporate_tax_rate',
  'property_wealth_tax_rate',
  'vat_consumption_tax_rate',
])

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value))
}

function formatLabel(value) {
  return String(value || '')
    .split('_')
    .filter(Boolean)
    .map((part) => {
      if (part === 'gdp') return 'GDP'
      if (part === 'rd') return 'R&D'
      return part.charAt(0).toUpperCase() + part.slice(1)
    })
    .join(' ')
}

function formatNumber(value) {
  const numericValue = Number(value)
  if (!Number.isFinite(numericValue)) {
    return 'n/a'
  }

  return numericValue.toLocaleString(undefined, {
    maximumFractionDigits: Math.abs(numericValue) >= 100 ? 1 : 2,
  })
}

function mercatorProject(lon, lat) {
  const longitude = Number(lon) * Math.PI / 180
  const boundedLat = clamp(Number(lat), -85, 85) * Math.PI / 180
  const x = longitude
  const y = -Math.log(Math.tan(Math.PI / 4 + boundedLat / 2))
  return [x, y]
}

function collectCoordinates(geometry) {
  const coordinates = []
  const stack = geometry && geometry.coordinates ? [geometry.coordinates] : []

  while (stack.length) {
    const node = stack.pop()
    if (!Array.isArray(node)) {
      continue
    }

    if (node.length >= 2 && typeof node[0] === 'number' && typeof node[1] === 'number') {
      coordinates.push(node)
      continue
    }

    for (let index = node.length - 1; index >= 0; index -= 1) {
      stack.push(node[index])
    }
  }

  return coordinates
}

function buildProjection(features, width, height, padding = 24) {
  let minX = Infinity
  let maxX = -Infinity
  let minY = Infinity
  let maxY = -Infinity
  let hasPoints = false

  features.forEach((feature) => {
    collectCoordinates(feature.geometry).forEach(([lon, lat]) => {
      const [x, y] = mercatorProject(lon, lat)
      if (x < minX) minX = x
      if (x > maxX) maxX = x
      if (y < minY) minY = y
      if (y > maxY) maxY = y
      hasPoints = true
    })
  })

  if (!hasPoints) {
    return {
      project: (lon, lat) => mercatorProject(lon, lat),
    }
  }

  const rawWidth = Math.max(1e-9, maxX - minX)
  const rawHeight = Math.max(1e-9, maxY - minY)
  const scale = Math.min((width - padding * 2) / rawWidth, (height - padding * 2) / rawHeight)
  const scaledWidth = rawWidth * scale
  const scaledHeight = rawHeight * scale
  const offsetX = padding + (width - padding * 2 - scaledWidth) / 2 - minX * scale
  const offsetY = padding + (height - padding * 2 - scaledHeight) / 2 - minY * scale

  return {
    project: (lon, lat) => {
      const [x, y] = mercatorProject(lon, lat)
      return [x * scale + offsetX, y * scale + offsetY]
    },
  }
}

function ringToPath(ring, projection) {
  if (!ring || !ring.length) {
    return ''
  }

  return ring
    .map(([lon, lat], index) => {
      const [x, y] = projection(lon, lat)
      return `${index === 0 ? 'M' : 'L'}${x.toFixed(2)},${y.toFixed(2)}`
    })
    .join(' ') + ' Z'
}

function geometryToPath(geometry, projection) {
  if (!geometry) {
    return ''
  }

  if (geometry.type === 'Polygon') {
    return geometry.coordinates.map((ring) => ringToPath(ring, projection)).join(' ')
  }

  if (geometry.type === 'MultiPolygon') {
    return geometry.coordinates
      .map((polygon) => polygon.map((ring) => ringToPath(ring, projection)).join(' '))
      .join(' ')
  }

  return ''
}

function interpolateColor(stops, ratio) {
  const clampedRatio = clamp(ratio, 0, 1)
  const scaled = clampedRatio * (stops.length - 1)
  const index = Math.floor(scaled)
  const nextIndex = Math.min(stops.length - 1, index + 1)
  const t = scaled - index

  const toRgb = (hex) => {
    const value = hex.replace('#', '')
    return [
      parseInt(value.slice(0, 2), 16),
      parseInt(value.slice(2, 4), 16),
      parseInt(value.slice(4, 6), 16),
    ]
  }

  const [r1, g1, b1] = toRgb(stops[index])
  const [r2, g2, b2] = toRgb(stops[nextIndex])
  return `rgb(${Math.round(r1 + (r2 - r1) * t)}, ${Math.round(g1 + (g2 - g1) * t)}, ${Math.round(b1 + (b2 - b1) * t)})`
}

function provinceFillColor(score, minScore, maxScore) {
  if (!Number.isFinite(score) || !Number.isFinite(minScore) || !Number.isFinite(maxScore)) {
    return '#e8eef6'
  }

  const span = Math.max(maxScore - minScore, 1e-9)
  const ratio = (score - minScore) / span

  return interpolateColor(
    ['#f7ebee', '#f1cfd8', '#e9a7b7', '#dd758d', '#cc4763', '#b61c3e', '#8f0826'],
    ratio,
  )
}

function buildProvinceHistory(rows, provinceCode) {
  return rows
    .filter((row) => row.area_code === provinceCode)
    .sort((left, right) => Number(left.year) - Number(right.year))
}

function mergeHistoryRows(existingRows, incomingRows) {
  const index = new Map()
  ;[...(existingRows || []), ...(incomingRows || [])].forEach((row) => {
    const key = `${row.area_code}:${row.year}`
    index.set(key, { ...(index.get(key) || {}), ...row })
  })
  return Array.from(index.values()).sort((left, right) => Number(left.year) - Number(right.year))
}

function hasValue(value) {
  return value !== null && value !== undefined
}

// Stable fallback so effects depending on the per-province cache entry do not
// re-run on every render while the entry is still missing.
const EMPTY_PROVINCE_CACHE = { rows: [], keysLoaded: [] }

function computeResourceBudget(allocations, parameterMeta, latestStateRows, spendingIntensityPct, reservePool) {
  // Mirrors backend_bridge._budget_usage: spending levers consume the pool,
  // tax levers fund it (cutting taxes below baseline shrinks the budget).
  const spendingMeta = parameterMeta.filter((p) => SPENDING_PARAMS.has(p.key))
  const taxMeta = parameterMeta.filter((p) => TAX_PARAMS.has(p.key))
  const numSpending = spendingMeta.length
  const intensity = Number(spendingIntensityPct) || 19.0
  let totalBasePool = 0
  let totalUsed = 0
  const byProvince = {}

  const avgRatioFor = (metaList, provinceAllocs) => {
    if (!metaList.length) return 1
    const sum = metaList.reduce((acc, pm) => {
      const val = hasValue(provinceAllocs[pm.key]) ? provinceAllocs[pm.key] : pm.baseline
      return acc + Number(val) / Math.max(Number(pm.baseline), 1e-9)
    }, 0)
    return sum / metaList.length
  }

  latestStateRows.forEach((row) => {
    const code = String(row.area_code || '').trim().toUpperCase()
    const gdp = Number(row.gdp_per_capita)
    if (!Number.isFinite(gdp) || gdp <= 0) return

    const provinceAllocs = allocations[code] || {}
    const avgRatio = numSpending > 0 ? avgRatioFor(spendingMeta, provinceAllocs) : 0
    const revenueRatio = avgRatioFor(taxMeta, provinceAllocs)
    const baseLimit = (intensity / 100) * gdp
    const cost = avgRatio * baseLimit

    byProvince[code] = { cost }
    totalBasePool += revenueRatio * baseLimit
    totalUsed += cost
  })

  const totalPool = totalBasePool + Math.max(0, Number(reservePool) || 0)
  return { byProvince, totalUsed, totalBasePool, totalPool }
}

const ItalyMap = React.memo(function ItalyMap({ features, allocations, parameterMeta, selectedProvince, onSelectProvince }) {
  const projection = useMemo(() => buildProjection(features, MAP_WIDTH, MAP_HEIGHT, 26), [features])

  const mappedFeatures = useMemo(() => {
    return features.map((feature) => {
      const provinceCode = String(feature.properties.prov_acr || '').trim().toUpperCase()
      return {
        feature,
        provinceCode,
        path: geometryToPath(feature.geometry, projection.project),
      }
    })
  }, [features, projection])

  const provinceScores = useMemo(() => {
    const result = {}
    mappedFeatures.forEach(({ provinceCode }) => {
      const provinceAllocations = allocations[provinceCode] || {}
      const score = parameterMeta.reduce((total, parameter) => {
        const rawValue = hasValue(provinceAllocations[parameter.key])
          ? provinceAllocations[parameter.key]
          : parameter.baseline
        const value = Number(rawValue)
        const span = Math.max(1, Number(parameter.max) - Number(parameter.min))
        return total + (value - Number(parameter.min)) / span
      }, 0)
      result[provinceCode] = score
    })
    return result
  }, [allocations, mappedFeatures, parameterMeta])

  const scoreRange = useMemo(() => {
    const values = Object.values(provinceScores).filter((value) => Number.isFinite(value))
    if (!values.length) {
      return { min: 0, max: 0 }
    }

    return {
      min: values.reduce((minValue, currentValue) => Math.min(minValue, currentValue), values[0]),
      max: values.reduce((maxValue, currentValue) => Math.max(maxValue, currentValue), values[0]),
    }
  }, [provinceScores])

  return (
    <React.Fragment>
      <div className="map-frame">
        <svg className="italy-map" viewBox={`0 0 ${MAP_WIDTH} ${MAP_HEIGHT}`} role="img" aria-label="Map of Italian provinces">
          {mappedFeatures.map(({ feature, provinceCode, path }) => {
            return (
              <path
                key={provinceCode}
                className={`province${provinceCode === selectedProvince ? ' selected' : ''}`}
                d={path}
                fill={provinceFillColor(provinceScores[provinceCode], scoreRange.min, scoreRange.max)}
                stroke={provinceCode === selectedProvince ? '#153b92' : 'rgba(16, 32, 56, 0.22)'}
                strokeWidth={provinceCode === selectedProvince ? 1.7 : 0.8}
                onClick={() => onSelectProvince(provinceCode)}
              >
                <title>{`${feature.properties.prov_name} (${provinceCode})`}</title>
              </path>
            )
          })}
        </svg>
      </div>
      <div className="map-legend">
        <span>Allocator intensity</span>
        <div className="legend-bar" />
        <span>Low to high</span>
      </div>
    </React.Fragment>
  )
})

function ResourceMeter({ used, limit, label, reserve }) {
  const effectiveLimit = limit + Math.max(0, Number(reserve) || 0)
  const fraction = effectiveLimit > 0 ? used / effectiveLimit : 0
  const pct = Math.min(fraction * 100, 100)
  const over = fraction > 1
  const fillColor = fraction < 0.75 ? '#22c55e' : fraction < 0.92 ? '#f59e0b' : '#ef4444'
  const reserveAmount = Math.max(0, Number(reserve) || 0)

  return (
    <div className="resource-meter">
      <div className="resource-meter-header">
        <span className="resource-meter-label">{label}</span>
        <span className="resource-meter-pct" style={{ color: over ? '#ef4444' : undefined }}>
          {(fraction * 100).toFixed(1)}%{over ? ' ▲ Over limit' : ''}
        </span>
      </div>
      <div className="resource-meter-track">
        <div className="resource-meter-fill" style={{ width: `${pct}%`, background: fillColor }} />
      </div>
      <div className="resource-meter-sub">
        {formatNumber(used)} / {formatNumber(effectiveLimit)} EUR/cap
        {reserveAmount > 0 && ` (incl. ${formatNumber(reserveAmount)} reserve)`}
      </div>
    </div>
  )
}

function RunProgress({ progress }) {
  if (!progress) {
    return null
  }
  const percent = Math.min(Math.max(Number(progress.percent) || 0, 0), 100)
  return (
    <div className="run-progress">
      <div className="resource-meter-header">
        <span className="resource-meter-label">{progress.message || 'Working...'}</span>
        <span className="resource-meter-pct">{percent.toFixed(0)}%</span>
      </div>
      <div className="resource-meter-track">
        <div className="resource-meter-fill run-progress-fill" style={{ width: `${percent}%` }} />
      </div>
    </div>
  )
}

function TrendChart({ provinceName, provinceCode, rows, indicatorKeys, simulationStartYear }) {
  const canvasRef = useRef(null)
  const chartRef = useRef(null)

  useEffect(() => {
    if (!canvasRef.current || !rows.length || !indicatorKeys.length) {
      return undefined
    }

    const labels = rows.map((row) => row.year)
    const datasets = indicatorKeys.map((indicatorKey, index) => ({
      label: formatLabel(indicatorKey),
      data: rows.map((row) => (hasValue(row[indicatorKey]) ? row[indicatorKey] : null)),
      borderColor: CHART_COLORS[index % CHART_COLORS.length],
      backgroundColor: `${CHART_COLORS[index % CHART_COLORS.length]}22`,
      borderWidth: 2,
      pointRadius: 2,
      pointHoverRadius: 4,
      spanGaps: true,
      tension: 0.28,
    }))

    const simulationLinePlugin = simulationStartYear != null ? {
      id: 'simulationLine',
      afterDraw(chart) {
        const { ctx, scales } = chart
        const xScale = scales.x
        const yScale = scales.y
        const idx = chart.data.labels.findIndex((l) => String(l) === String(simulationStartYear))
        if (idx < 0) return
        const x = xScale.getPixelForValue(idx)
        ctx.save()
        ctx.beginPath()
        ctx.setLineDash([6, 4])
        ctx.moveTo(x, yScale.top)
        ctx.lineTo(x, yScale.bottom)
        ctx.strokeStyle = '#dc2626'
        ctx.lineWidth = 2
        ctx.stroke()
        ctx.setLineDash([])
        ctx.fillStyle = '#dc2626'
        ctx.font = 'bold 11px "Segoe UI", Helvetica, sans-serif'
        ctx.textAlign = 'left'
        ctx.fillText('Simulation start', x + 5, yScale.top + 14)
        ctx.restore()
      },
    } : null

    if (chartRef.current) {
      chartRef.current.destroy()
    }

    chartRef.current = new Chart(canvasRef.current, {
      type: 'line',
      data: { labels, datasets },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        interaction: { mode: 'index', intersect: false },
        plugins: {
          legend: { position: 'bottom' },
          title: {
            display: true,
            text: `${provinceName} (${provinceCode}) indicator trend`,
            color: '#102038',
            font: { size: 15, weight: '700' },
          },
        },
        scales: {
          x: {
            ticks: { color: '#5b6f89' },
            grid: { color: 'rgba(16, 32, 56, 0.06)' },
          },
          y: {
            ticks: { color: '#5b6f89' },
            grid: { color: 'rgba(16, 32, 56, 0.08)' },
          },
        },
      },
      plugins: simulationLinePlugin ? [simulationLinePlugin] : [],
    })

    return () => {
      if (chartRef.current) {
        chartRef.current.destroy()
        chartRef.current = null
      }
    }
  }, [indicatorKeys, provinceCode, provinceName, rows, simulationStartYear])

  if (!rows.length) {
    return <div className="empty-state">No historical data is available for this province.</div>
  }

  if (!indicatorKeys.length) {
    return <div className="empty-state">Select at least one indicator to render the trend plot.</div>
  }

  return (
    <div className="chart-canvas" style={{ height: 420 }}>
      <canvas ref={canvasRef} />
    </div>
  )
}

const BAND_LABEL = '::band::'

function MultiSeriesChart({ seriesList, title, height, bandSeries }) {
  const canvasRef = useRef(null)
  const chartRef = useRef(null)

  const validSeries = (seriesList || []).filter((series) => series.data && series.data.length)

  useEffect(() => {
    if (!canvasRef.current || !validSeries.length) {
      return undefined
    }

    const labels = validSeries[0].data.map((point) => point.year)
    const datasets = []
    if (bandSeries && bandSeries.data && bandSeries.data.length) {
      // Two invisible-border lines with the area between them filled: the
      // sensitivity band (same policy under weaker/stronger causal rules).
      datasets.push(
        {
          label: BAND_LABEL,
          data: bandSeries.data.map((point) => point.low),
          borderColor: 'transparent',
          backgroundColor: 'transparent',
          pointRadius: 0,
          fill: false,
          tension: 0.28,
        },
        {
          label: BAND_LABEL,
          data: bandSeries.data.map((point) => point.high),
          borderColor: 'transparent',
          backgroundColor: `${bandSeries.color}2e`,
          pointRadius: 0,
          fill: '-1',
          tension: 0.28,
        },
      )
    }
    validSeries.forEach((series) => {
      datasets.push({
        label: series.label,
        data: series.data.map((point) => point.value),
        borderColor: series.color,
        backgroundColor: `${series.color}22`,
        borderWidth: series.dashed ? 2 : 2.4,
        borderDash: series.dashed ? [6, 4] : undefined,
        pointRadius: series.dashed ? 0 : 1.6,
        tension: 0.28,
      })
    })

    if (chartRef.current) {
      chartRef.current.destroy()
    }

    chartRef.current = new Chart(canvasRef.current, {
      type: 'line',
      data: { labels, datasets },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        interaction: { mode: 'index', intersect: false },
        plugins: {
          legend: {
            position: 'bottom',
            labels: {
              boxWidth: 18,
              filter: (item) => item.text !== BAND_LABEL,
            },
          },
          tooltip: {
            filter: (item) => item.dataset.label !== BAND_LABEL,
          },
          title: {
            display: true,
            text: title,
            color: '#102038',
            font: { size: 14, weight: '700' },
          },
        },
        scales: {
          x: { ticks: { color: '#5b6f89' }, grid: { color: 'rgba(16, 32, 56, 0.06)' } },
          y: { ticks: { color: '#5b6f89' }, grid: { color: 'rgba(16, 32, 56, 0.08)' } },
        },
      },
    })

    return () => {
      if (chartRef.current) {
        chartRef.current.destroy()
        chartRef.current = null
      }
    }
  }, [seriesList, title, bandSeries])

  if (!validSeries.length) {
    return <div className="empty-state">No data for this chart yet.</div>
  }

  return (
    <div className="chart-canvas" style={{ height: height || 300 }}>
      <canvas ref={canvasRef} />
    </div>
  )
}

const DeltaMap = React.memo(function DeltaMap({ features, deltaByProvince, maxAbs }) {
  const projection = useMemo(() => buildProjection(features, MAP_WIDTH, MAP_HEIGHT, 26), [features])

  const mappedFeatures = useMemo(() => {
    return features.map((feature) => {
      const provinceCode = String(feature.properties.prov_acr || '').trim().toUpperCase()
      return {
        feature,
        provinceCode,
        path: geometryToPath(feature.geometry, projection.project),
      }
    })
  }, [features, projection])

  function deltaFillColor(delta) {
    if (!Number.isFinite(delta) || !(maxAbs > 0)) return '#e8eef6'
    const ratio = (delta + maxAbs) / (2 * maxAbs)
    return interpolateColor(['#b61c3e', '#e9a7b7', '#f1f5f9', '#7fc1b4', '#0f766e'], ratio)
  }

  return (
    <React.Fragment>
      <div className="map-frame">
        <svg className="italy-map" viewBox={`0 0 ${MAP_WIDTH} ${MAP_HEIGHT}`} role="img" aria-label="GDP change vs baseline by province">
          {mappedFeatures.map(({ feature, provinceCode, path }) => {
            const delta = deltaByProvince[provinceCode]
            return (
              <path
                key={provinceCode}
                className="province"
                d={path}
                fill={deltaFillColor(Number.isFinite(delta) ? delta : NaN)}
                stroke="rgba(16, 32, 56, 0.22)"
                strokeWidth={0.8}
              >
                <title>{`${feature.properties.prov_name} (${provinceCode}): ${Number.isFinite(delta) ? `${delta >= 0 ? '+' : ''}${delta.toFixed(1)}% vs baseline` : 'n/a'}`}</title>
              </path>
            )
          })}
        </svg>
      </div>
      <div className="map-legend">
        <span>-{formatNumber(maxAbs)}%</span>
        <div className="legend-bar legend-bar--diverging" />
        <span>+{formatNumber(maxAbs)}%</span>
      </div>
    </React.Fragment>
  )
})

const IndicatorMap = React.memo(function IndicatorMap({ features, latestStateRows, indicatorKey, selectedProvince, onSelectProvince }) {
  const projection = useMemo(() => buildProjection(features, MAP_WIDTH, MAP_HEIGHT, 26), [features])

  const mappedFeatures = useMemo(() => {
    return features.map((feature) => {
      const provinceCode = String(feature.properties.prov_acr || '').trim().toUpperCase()
      return {
        feature,
        provinceCode,
        path: geometryToPath(feature.geometry, projection.project),
      }
    })
  }, [features, projection])

  const provinceValues = useMemo(() => {
    const result = {}
    latestStateRows.forEach((row) => {
      if (row.area_code && indicatorKey) {
        const val = row[indicatorKey]
        if (val !== null && val !== undefined) {
          result[String(row.area_code).trim().toUpperCase()] = Number(val)
        }
      }
    })
    return result
  }, [latestStateRows, indicatorKey])

  const valueRange = useMemo(() => {
    const vals = Object.values(provinceValues).filter((v) => Number.isFinite(v))
    if (!vals.length) return { min: 0, max: 0 }
    return { min: Math.min(...vals), max: Math.max(...vals) }
  }, [provinceValues])

  function indicatorFillColor(value) {
    if (!Number.isFinite(value)) return '#e8eef6'
    const span = Math.max(valueRange.max - valueRange.min, 1e-9)
    const ratio = (value - valueRange.min) / span
    return interpolateColor(['#d1e4f5', '#a5c8ed', '#6aaad6', '#3b82b8', '#1d5c94', '#0f3d6b'], ratio)
  }

  return (
    <React.Fragment>
      <div className="map-frame">
        <svg className="italy-map" viewBox={`0 0 ${MAP_WIDTH} ${MAP_HEIGHT}`} role="img" aria-label={`${formatLabel(indicatorKey)} by province`}>
          {mappedFeatures.map(({ feature, provinceCode, path }) => {
            const value = provinceValues[provinceCode]
            return (
              <path
                key={provinceCode}
                className={`province${provinceCode === selectedProvince ? ' selected' : ''}`}
                d={path}
                fill={indicatorFillColor(Number.isFinite(value) ? value : NaN)}
                stroke={provinceCode === selectedProvince ? '#153b92' : 'rgba(16, 32, 56, 0.22)'}
                strokeWidth={provinceCode === selectedProvince ? 1.7 : 0.8}
                onClick={() => onSelectProvince(provinceCode)}
              >
                <title>{`${feature.properties.prov_name} (${provinceCode}): ${formatNumber(value)}`}</title>
              </path>
            )
          })}
        </svg>
      </div>
      <div className="map-legend">
        <span>{formatNumber(valueRange.min)}</span>
        <div className="legend-bar legend-bar--blue" />
        <span>{formatNumber(valueRange.max)}</span>
      </div>
    </React.Fragment>
  )
})

function ExplainBadge({ explainability }) {
  const meta = EXPLAINABILITY_META[explainability]
  if (!meta) return null
  return (
    <span className="explain-badge" style={{ background: `${meta.color}1a`, color: meta.color, borderColor: `${meta.color}55` }} title={meta.blurb}>
      {meta.label}
    </span>
  )
}

function LeverDeltaTable({ levers, limit }) {
  const rows = [...(levers || [])]
    .map((item) => ({
      ...item,
      shift: Math.abs(item.value - item.baseline) / Math.max(
        (Number.isFinite(item.max) ? item.max : item.value) - (Number.isFinite(item.min) ? item.min : item.baseline),
        1e-9,
      ),
    }))
    .sort((a, b) => b.shift - a.shift)
    .slice(0, limit || levers.length)
  return (
    <table className="explain-table">
      <thead>
        <tr><th>Lever</th><th>Chosen value</th><th>Baseline</th></tr>
      </thead>
      <tbody>
        {rows.map((item) => (
          <tr key={item.lever}>
            <td>{formatLabel(item.lever)}</td>
            <td>{formatNumber(item.value)}</td>
            <td>{formatNumber(item.baseline)}</td>
          </tr>
        ))}
      </tbody>
    </table>
  )
}

function LinearWeightsHeatmap({ explanation }) {
  const features = explanation.features || []
  const levers = explanation.levers || []
  let maxAbs = 0
  levers.forEach((row) => {
    features.forEach((feature) => {
      maxAbs = Math.max(maxAbs, Math.abs(Number(row.weights[feature]) || 0))
    })
  })
  const cellColor = (weight) => {
    if (!(maxAbs > 0) || !Number.isFinite(weight)) return '#f1f5f9'
    return interpolateColor(DIVERGING_PALETTE, (weight + maxAbs) / (2 * maxAbs))
  }
  return (
    <div className="explain-scroll">
      <table className="explain-table explain-heatmap">
        <thead>
          <tr>
            <th>Lever</th>
            {features.map((feature) => <th key={feature}>{featureDisplayName(feature)}</th>)}
          </tr>
        </thead>
        <tbody>
          {levers.map((row) => (
            <tr key={row.lever}>
              <td>{formatLabel(row.lever)}</td>
              {features.map((feature) => {
                const weight = Number(row.weights[feature])
                return (
                  <td key={feature} style={{ background: cellColor(weight) }} title={`${formatLabel(row.lever)} ← ${featureDisplayName(feature)}: ${weight.toFixed(3)}`}>
                    {Number.isFinite(weight) ? weight.toFixed(2) : 'n/a'}
                  </td>
                )
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

function DecisionRulesList({ explanation }) {
  const rules = explanation.rules || []
  return (
    <ol className="explain-rule-list">
      {rules.map((rule) => (
        <li key={rule.lever}>
          <strong>{formatLabel(rule.lever)}</strong>{': '}
          {rule.feature ? (
            <React.Fragment>
              IF <em>{formatLabel(rule.feature)}</em> is above the national reference {rule.threshold_pct >= 0 ? '+' : ''}{Number(rule.threshold_pct).toFixed(0)}%
              {Number.isFinite(rule.threshold_value) ? ` (≈ ${formatNumber(rule.threshold_value)})` : ''}
              {' → '}<strong>{formatNumber(rule.value_if_above)}</strong>, otherwise → <strong>{formatNumber(rule.value_if_below)}</strong>
            </React.Fragment>
          ) : (
            <React.Fragment>always <strong>{formatNumber(rule.value_if_below)}</strong></React.Fragment>
          )}
          {' '}(baseline {formatNumber(rule.baseline)})
        </li>
      ))}
    </ol>
  )
}

function ClusterPolicyCards({ explanation }) {
  const clusters = explanation.clusters || []
  return (
    <div className="explain-cluster-grid">
      {clusters.map((cluster, index) => (
        <div className="explain-cluster-card" key={cluster.id}>
          <h4>Group {index + 1} — {cluster.provinces.length} provinces</h4>
          <p className="control-text explain-provinces">{cluster.provinces.join(', ') || '—'}</p>
          {Object.keys(cluster.profile || {}).length ? (
            <p className="control-text">
              Profile: {Object.entries(cluster.profile).map(([key, value]) => `${formatLabel(key)} ${formatNumber(value)}`).join(' · ')}
            </p>
          ) : null}
          <LeverDeltaTable levers={cluster.levers} limit={8} />
        </div>
      ))}
    </div>
  )
}

function PartialDependenceTable({ explanation }) {
  const levers = explanation.levers || []
  return (
    <React.Fragment>
      <p className="control-text">
        Built from {explanation.evaluations} twin rollouts — the Gaussian-process surrogate estimates how the
        objective responds to each lever while the others stay at the optimum.
      </p>
      <div className="explain-scroll">
        <table className="explain-table">
          <thead>
            <tr><th>Lever</th><th>Optimised value</th><th>Baseline</th><th>Objective swing</th><th>GP uncertainty (±)</th></tr>
          </thead>
          <tbody>
            {levers.map((item) => (
              <tr key={item.lever}>
                <td>{formatLabel(item.lever)}</td>
                <td>{formatNumber(item.best_value)}</td>
                <td>{formatNumber(item.baseline)}</td>
                <td>{formatNumber(item.effect_range)}</td>
                <td>{formatNumber(item.mean_std)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </React.Fragment>
  )
}

function NeuralPosthocPanel({ explanation }) {
  const importances = explanation.importances || []
  const surrogate = explanation.surrogate
  const maxImportance = Math.max(...importances.map((item) => item.importance), 1e-9)
  return (
    <React.Fragment>
      <h4 className="explain-subtitle">Which indicators drive the network's decisions</h4>
      {importances.length ? (
        <div className="importance-list">
          {importances.map((item) => (
            <div className="importance-row" key={item.feature}>
              <span className="importance-label">{formatLabel(item.feature)}</span>
              <div className="importance-track">
                <div className="importance-bar" style={{ width: `${Math.max(2, (item.importance / maxImportance) * 100)}%` }} />
              </div>
              <span className="importance-value">{item.importance.toFixed(3)}</span>
            </div>
          ))}
        </div>
      ) : (
        <p className="control-text">No per-province features available to audit.</p>
      )}
      {surrogate ? (
        <React.Fragment>
          <h4 className="explain-subtitle">Distilled surrogate — province segments imitating the network</h4>
          <p className="control-text">
            A depth-{surrogate.max_depth} decision tree reproduces the network's lever choices with{' '}
            <strong>fidelity R² = {Number(surrogate.fidelity_r2).toFixed(2)}</strong> on held-out decisions.
            Read the segments as an honest approximation, not the network itself.
          </p>
          <div className="explain-cluster-grid">
            {(surrogate.segments || []).map((segment, index) => (
              <div className="explain-cluster-card" key={index}>
                <h4>{segment.conditions.join(' AND ')}</h4>
                <p className="control-text">{segment.n_samples} decision samples</p>
                <table className="explain-table">
                  <thead><tr><th>Lever</th><th>Value</th><th>Baseline</th></tr></thead>
                  <tbody>
                    {segment.levers.map((lever) => (
                      <tr key={lever.lever}>
                        <td>{formatLabel(lever.lever)}</td>
                        <td>{formatNumber(lever.value)}</td>
                        <td>{formatNumber(lever.baseline)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ))}
          </div>
        </React.Fragment>
      ) : null}
    </React.Fragment>
  )
}

function PolicyExplanation({ explanation, explainability }) {
  if (!explanation) return null
  let body = null
  if (explanation.type === 'linear_weights') body = <LinearWeightsHeatmap explanation={explanation} />
  else if (explanation.type === 'decision_rules') body = <DecisionRulesList explanation={explanation} />
  else if (explanation.type === 'cluster_policy') body = <ClusterPolicyCards explanation={explanation} />
  else if (explanation.type === 'partial_dependence') body = <PartialDependenceTable explanation={explanation} />
  else if (explanation.type === 'lever_table') body = <LeverDeltaTable levers={explanation.levers} />
  else if (explanation.type === 'neural_posthoc') body = <NeuralPosthocPanel explanation={explanation} />
  if (!body) return null
  return (
    <div className="explain-section">
      <h3 className="ethics-maps-title">
        Why these levers — model explanation <ExplainBadge explainability={explainability} />
      </h3>
      {explanation.note ? <p className="control-text">{explanation.note}</p> : null}
      {body}
    </div>
  )
}

function PolicyCandidates({ result, models, onAdopt, adoptedId, disabled }) {
  const candidates = result.candidates || []
  const baselineCandidate = candidates.find((candidate) => candidate.id === 'baseline') || {}
  const modelLabel = (models.find((m) => m.id === result.modelId) || {}).label || result.modelId
  const fullCandidate = candidates.find((candidate) => candidate.id === 'full')

  const gdpSeries = candidates.map((candidate) => ({
    label: candidate.label,
    data: candidate.gdpByYear,
    color: CANDIDATE_COLORS[candidate.id] || '#1d4ea4',
    dashed: candidate.id === 'baseline',
  }))
  const welfareSeries = candidates.map((candidate) => ({
    label: candidate.label,
    data: candidate.welfareByYear,
    color: CANDIDATE_COLORS[candidate.id] || '#1d4ea4',
    dashed: candidate.id === 'baseline',
  }))
  const band = fullCandidate && fullCandidate.gdpBandByYear && fullCandidate.gdpBandByYear.length
    ? { data: fullCandidate.gdpBandByYear, color: CANDIDATE_COLORS.full }
    : null

  const pctDelta = (value, base) => {
    if (!Number.isFinite(value) || !Number.isFinite(base) || base === 0) return null
    return (value / base - 1) * 100
  }
  const renderDelta = (delta, goodWhenPositive, suffix = '%') => {
    if (delta == null || !Number.isFinite(delta)) return null
    const good = goodWhenPositive ? delta >= 0 : delta < 0
    return (
      <span className={good ? 'delta-positive' : 'delta-negative'}>
        {' '}({delta >= 0 ? '+' : ''}{delta.toFixed(1)}{suffix})
      </span>
    )
  }

  return (
    <React.Fragment>
      {result.trainInfo && result.trainInfo.improvement_pct != null ? (
        <p className="control-text" style={{ marginBottom: 10 }}>
          <strong>{modelLabel}</strong> trained on <strong>{result.objectiveLabel}</strong> through {result.finalYear}
          {result.trainInfo.best_score != null ? `; best cumulative objective score ${formatNumber(result.trainInfo.best_score)}` : ''}
          {' '}(training lift {formatNumber(result.trainInfo.improvement_pct)}% over the model's first guess).
        </p>
      ) : null}
      <p className="uncertainty-note">
        Everything below is a model estimate, not data. The shaded band re-simulates the full intervention with the twin's hand-written
        causal rules at half and 1.5× strength — where the band is wide, the projection depends heavily on those assumptions; where it is
        narrow under an aggressive policy, the ±6%/year realism cap is usually what is driving the path instead.
        See ETHICS.md and docs/MODEL_CARD.md for the full limitations.
      </p>
      <div className="ethics-charts">
        <MultiSeriesChart seriesList={gdpSeries} title="Total national GDP — candidates vs baseline" bandSeries={band} />
        {result.objectiveId && result.objectiveId !== 'utilitarian' ? (
          <MultiSeriesChart seriesList={welfareSeries} title={`${result.objectiveLabel} welfare trajectory`} />
        ) : null}
      </div>
      <PolicyExplanation explanation={result.explanation} explainability={result.explainability} />
      <h3 className="ethics-maps-title">Candidate policies — the choice is yours, not the model's</h3>
      <div className="candidate-grid">
        {candidates.map((candidate) => {
          const isAdopted = adoptedId === candidate.id
          const isBaselineRow = candidate.id === 'baseline'
          return (
            <div className={`candidate-card${isAdopted ? ' adopted' : ''}`} key={candidate.id}>
              <h4>
                <span className="objective-swatch" style={{ background: CANDIDATE_COLORS[candidate.id] || '#1d4ea4' }} />
                {candidate.label}
              </h4>
              <p className="control-text">{candidate.description}</p>
              <ul className="candidate-metrics">
                <li>
                  <span>Final total GDP</span>
                  <span>
                    {formatNumber(candidate.finalGdpTotal)}
                    {!isBaselineRow ? renderDelta(pctDelta(candidate.finalGdpTotal, baselineCandidate.finalGdpTotal), true) : null}
                  </span>
                </li>
                <li>
                  <span>Inequality (Gini)</span>
                  <span>
                    {Number.isFinite(candidate.finalGini) ? candidate.finalGini.toFixed(3) : 'n/a'}
                    {!isBaselineRow && Number.isFinite(candidate.finalGini) && Number.isFinite(baselineCandidate.finalGini)
                      ? renderDelta((candidate.finalGini - baselineCandidate.finalGini) * 100, false, ' pts')
                      : null}
                  </span>
                </li>
                <li>
                  <span>Worst-off province GDP</span>
                  <span>
                    {formatNumber(candidate.worstProvinceGdp)}
                    {!isBaselineRow ? renderDelta(pctDelta(candidate.worstProvinceGdp, baselineCandidate.worstProvinceGdp), true) : null}
                  </span>
                </li>
                <li>
                  <span>Unspent reserve</span>
                  <span>{formatNumber(candidate.reservePool)}</span>
                </li>
              </ul>
              <button
                className="primary-button"
                onClick={() => onAdopt(candidate)}
                disabled={disabled || Boolean(adoptedId)}
              >
                {isAdopted ? 'Adopted ✓' : `Adopt: advance twin to ${result.finalYear}`}
              </button>
            </div>
          )
        })}
      </div>
      {adoptedId ? (
        <p className="control-text" style={{ marginTop: 10 }}>
          A candidate has been adopted and the twin advanced. Run the optimizer again from the new state to get fresh candidates.
        </p>
      ) : (
        <p className="control-text" style={{ marginTop: 10 }}>
          Nothing has been applied yet — adopting a candidate (including the baseline) is a human decision the model cannot make for you.
        </p>
      )}
    </React.Fragment>
  )
}

function EthicsComparison({ compareResult, models, features }) {
  const baseline = compareResult.baseline || {}
  const results = compareResult.results || []
  const modelLabel = (models.find((m) => m.id === compareResult.modelId) || {}).label || compareResult.modelId

  const chartSeries = useMemo(() => {
    const build = (key) => [
      ...results.map((result) => ({
        label: result.objectiveLabel,
        data: result[key],
        color: OBJECTIVE_COLORS[result.objectiveId] || '#1d4ea4',
      })),
      { label: 'Baseline (historical levers)', data: baseline[key], color: BASELINE_SERIES_COLOR, dashed: true },
    ]
    return {
      gdp: build('gdpByYear'),
      gini: build('giniByYear'),
      worst: build('worstGdpByYear'),
    }
  }, [compareResult])

  const mapData = useMemo(() => {
    const baseGdp = baseline.finalGdpByProvince || {}
    let maxAbs = 0
    const deltasByObjective = results.map((result) => {
      const deltas = {}
      Object.entries(result.finalGdpByProvince || {}).forEach(([code, value]) => {
        const base = Number(baseGdp[code])
        if (Number.isFinite(base) && base > 0 && Number.isFinite(Number(value))) {
          const delta = (Number(value) / base - 1) * 100
          deltas[code] = delta
          maxAbs = Math.max(maxAbs, Math.abs(delta))
        }
      })
      return { objectiveId: result.objectiveId, objectiveLabel: result.objectiveLabel, deltas }
    })
    return { deltasByObjective, maxAbs }
  }, [compareResult])

  const formatDelta = (value, suffix, decimals) => {
    if (!Number.isFinite(value)) return <span>n/a</span>
    const text = `${value >= 0 ? '+' : ''}${value.toFixed(decimals)}${suffix}`
    // For inequality, down is the "good" direction; callers pre-flip the sign convention via goodWhenNegative.
    return <span className={value >= 0 ? 'delta-positive' : 'delta-negative'}>{text}</span>
  }

  return (
    <React.Fragment>
      <p className="control-text" style={{ marginBottom: 12 }}>
        <strong>{modelLabel}</strong> trained separately under each framework, {compareResult.horizon} years from the current state (through {compareResult.finalYear}).
      </p>
      <table className="ethics-table">
        <thead>
          <tr>
            <th>Ethical framework</th>
            <th>Final total GDP</th>
            <th>Δ GDP vs baseline</th>
            <th>Final Gini</th>
            <th>Δ Gini (pts)</th>
            <th>Worst-off province GDP</th>
            <th>Δ worst-off</th>
          </tr>
        </thead>
        <tbody>
          <tr className="ethics-table-baseline">
            <td>Baseline (historical levers)</td>
            <td>{formatNumber(baseline.finalGdpTotal)}</td>
            <td>—</td>
            <td>{Number.isFinite(baseline.finalGini) ? baseline.finalGini.toFixed(3) : 'n/a'}</td>
            <td>—</td>
            <td>{formatNumber(baseline.worstProvinceGdp)}</td>
            <td>—</td>
          </tr>
          {results.map((result) => (
            <tr key={result.objectiveId}>
              <td>
                <span className="objective-swatch" style={{ background: OBJECTIVE_COLORS[result.objectiveId] || '#1d4ea4' }} />
                {result.objectiveLabel}
              </td>
              <td>{formatNumber(result.finalGdpTotal)}</td>
              <td>{formatDelta((result.finalGdpTotal / baseline.finalGdpTotal - 1) * 100, '%', 1)}</td>
              <td>{Number.isFinite(result.finalGini) ? result.finalGini.toFixed(3) : 'n/a'}</td>
              <td>{formatDelta(-(result.finalGini - baseline.finalGini) * 100, '', 2)}</td>
              <td>{formatNumber(result.worstProvinceGdp)}</td>
              <td>{formatDelta((result.worstProvinceGdp / baseline.worstProvinceGdp - 1) * 100, '%', 1)}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <p className="control-text" style={{ marginBottom: 14 }}>
        Δ Gini is shown so that positive (green) always means <em>less</em> inequality than baseline.
      </p>
      <div className="ethics-charts">
        <MultiSeriesChart seriesList={chartSeries.gdp} title="Total national GDP (efficiency)" />
        <MultiSeriesChart seriesList={chartSeries.gini} title="Inter-provincial Gini (inequality)" />
        <MultiSeriesChart seriesList={chartSeries.worst} title="Worst-off province GDP (the floor)" />
      </div>
      <h3 className="ethics-maps-title">Who gains, who loses — final-year provincial GDP vs baseline</h3>
      <div className="ethics-maps">
        {mapData.deltasByObjective.map((entry) => (
          <div className="ethics-map-card" key={entry.objectiveId}>
            <h4>
              <span className="objective-swatch" style={{ background: OBJECTIVE_COLORS[entry.objectiveId] || '#1d4ea4' }} />
              {entry.objectiveLabel}
            </h4>
            <DeltaMap features={features} deltaByProvince={entry.deltas} maxAbs={mapData.maxAbs} />
          </div>
        ))}
      </div>
    </React.Fragment>
  )
}

const TAG_LABELS = {
  utilitarian: 'Utilitarian corner (max total GDP)',
  egalitarian: 'Egalitarian corner (min Gini)',
  rawlsian: 'Rawlsian corner (max worst-off GDP)',
}

function ParetoScatter({ points, baseline, selectedIndex, onSelectPoint }) {
  const canvasRef = useRef(null)
  const chartRef = useRef(null)

  useEffect(() => {
    if (!canvasRef.current || !points.length) return undefined

    const frontier = {
      label: 'Pareto frontier',
      data: points.map((point, index) => ({ x: point.finalGini, y: point.finalGdpTotal, index })),
      backgroundColor: points.map((point, index) => {
        if (index === selectedIndex) return '#102038'
        const tag = (point.tags || [])[0]
        return tag ? OBJECTIVE_COLORS[tag] || '#0f766e' : '#0f766e'
      }),
      pointRadius: points.map((point, index) => ((point.tags || []).length || index === selectedIndex ? 6 : 4)),
      pointHoverRadius: 8,
    }
    const baselinePoint = {
      label: 'Baseline (historical levers)',
      data: baseline ? [{ x: baseline.finalGini, y: baseline.finalGdpTotal }] : [],
      backgroundColor: BASELINE_SERIES_COLOR,
      pointStyle: 'rectRot',
      pointRadius: 7,
    }

    if (chartRef.current) chartRef.current.destroy()
    chartRef.current = new Chart(canvasRef.current, {
      type: 'scatter',
      data: { datasets: [frontier, baselinePoint] },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        onClick: (_event, elements) => {
          if (!elements.length || !onSelectPoint) return
          const element = elements[0]
          if (element.datasetIndex === 0) onSelectPoint(element.index)
        },
        plugins: {
          legend: { position: 'bottom', labels: { boxWidth: 18 } },
          title: {
            display: true,
            text: 'Efficiency vs inequality — every point is a non-dominated policy',
            color: '#102038',
            font: { size: 14, weight: '700' },
          },
          tooltip: {
            callbacks: {
              label: (context) => {
                if (context.datasetIndex === 1) {
                  return `Baseline: GDP ${formatNumber(context.parsed.y)}, Gini ${context.parsed.x.toFixed(3)}`
                }
                const point = points[context.dataIndex]
                const tags = (point.tags || []).map((tag) => TAG_LABELS[tag] || tag).join('; ')
                return [
                  `GDP ${formatNumber(point.finalGdpTotal)}, Gini ${point.finalGini.toFixed(3)}`,
                  `Worst-off province GDP ${formatNumber(point.worstProvinceGdp)}`,
                  tags || null,
                ].filter(Boolean)
              },
            },
          },
        },
        scales: {
          x: {
            title: { display: true, text: 'Inter-provincial Gini (lower = more equal)', color: '#5b6f89' },
            ticks: { color: '#5b6f89' },
            grid: { color: 'rgba(16, 32, 56, 0.06)' },
          },
          y: {
            title: { display: true, text: 'Total national GDP (final year)', color: '#5b6f89' },
            ticks: { color: '#5b6f89' },
            grid: { color: 'rgba(16, 32, 56, 0.08)' },
          },
        },
      },
    })

    return () => {
      if (chartRef.current) {
        chartRef.current.destroy()
        chartRef.current = null
      }
    }
  }, [points, baseline, selectedIndex])

  if (!points.length) return <div className="empty-state">No frontier points yet.</div>
  return (
    <div className="chart-canvas" style={{ height: 360 }}>
      <canvas ref={canvasRef} />
    </div>
  )
}

function ParetoFrontier({ paretoResult, parameterMeta }) {
  const [selectedIndex, setSelectedIndex] = useState(null)
  const points = paretoResult.points || []
  const baseline = paretoResult.baseline || null
  const selected = selectedIndex != null ? points[selectedIndex] : null

  const baselineByKey = useMemo(() => {
    const map = {}
    ;(parameterMeta || []).forEach((item) => { map[item.key] = item })
    return map
  }, [parameterMeta])

  const selectedLevers = useMemo(() => {
    if (!selected) return []
    return Object.entries(selected.levers || {})
      .map(([key, value]) => {
        const meta = baselineByKey[key] || {}
        const span = Math.max(
          (Number.isFinite(meta.max) ? meta.max : Number(value)) - (Number.isFinite(meta.min) ? meta.min : 0),
          1e-9,
        )
        return {
          key,
          value: Number(value),
          baseline: Number(meta.baseline),
          shift: Number.isFinite(Number(meta.baseline)) ? Math.abs(Number(value) - Number(meta.baseline)) / span : 0,
        }
      })
      .sort((a, b) => b.shift - a.shift)
  }, [selected, baselineByKey])

  const nClusters = Number(paretoResult.nClusters) || 1
  const clustered = nClusters > 1

  return (
    <React.Fragment>
      <p className="control-text" style={{ marginBottom: 10 }}>
        NSGA-II evolved {paretoResult.evaluations}{' '}
        {clustered
          ? `per-cluster lever vectors (${nClusters} k-means regional packages)`
          : 'uniform national lever vectors'}{' '}
        over {paretoResult.generations} generations
        ({paretoResult.horizon} years, through {paretoResult.finalYear}). Every point below is non-dominated: improving one of
        total GDP, inequality, or the worst-off province necessarily worsens another.{' '}
        {clustered
          ? 'Because policy can differ across regions here, the frontier spans a real range of inequality — targeting, not national intensity, is what opens the trade-off.'
          : 'The named ethical frameworks are corners of this frontier, not separate truths.'}
      </p>
      <ParetoScatter points={points} baseline={baseline} selectedIndex={selectedIndex} onSelectPoint={setSelectedIndex} />
      <div className="explain-scroll" style={{ marginTop: 12 }}>
        <table className="explain-table pareto-table">
          <thead>
            <tr><th>#</th><th>Total GDP</th><th>Gini</th><th>Worst-off GDP</th><th>Character</th></tr>
          </thead>
          <tbody>
            <tr className="ethics-table-baseline">
              <td>—</td>
              <td>{formatNumber(baseline && baseline.finalGdpTotal)}</td>
              <td>{baseline && Number.isFinite(baseline.finalGini) ? baseline.finalGini.toFixed(3) : 'n/a'}</td>
              <td>{formatNumber(baseline && baseline.worstProvinceGdp)}</td>
              <td>Baseline (historical levers)</td>
            </tr>
            {points.map((point, index) => (
              <tr
                key={index}
                className={index === selectedIndex ? 'pareto-row-selected' : ''}
                onClick={() => setSelectedIndex(index === selectedIndex ? null : index)}
              >
                <td>{index + 1}</td>
                <td>{formatNumber(point.finalGdpTotal)}</td>
                <td>{point.finalGini.toFixed(3)}</td>
                <td>{formatNumber(point.worstProvinceGdp)}</td>
                <td>{(point.tags || []).map((tag) => TAG_LABELS[tag] || tag).join('; ') || 'Interior trade-off'}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {selected && selected.clusters ? (
        <div className="explain-section">
          <h3 className="ethics-maps-title">
            Regional packages of frontier point #{selectedIndex + 1} ({selected.clusters.length} clusters)
          </h3>
          {selected.clusters.map((cluster) => (
            <div key={cluster.id} style={{ marginBottom: 12 }}>
              <p className="control-text" style={{ marginBottom: 4 }}>
                <strong>Cluster {cluster.id + 1}</strong> — {cluster.provinces} province{cluster.provinces === 1 ? '' : 's'}
              </p>
              <LeverDeltaTable
                levers={Object.entries(cluster.levers || {})
                  .map(([key, value]) => {
                    const meta = baselineByKey[key] || {}
                    return { lever: key, value: Number(value), baseline: Number(meta.baseline), min: meta.min, max: meta.max }
                  })
                  .sort((a, b) => {
                    const sa = Number.isFinite(a.baseline) ? Math.abs(a.value - a.baseline) : 0
                    const sb = Number.isFinite(b.baseline) ? Math.abs(b.value - b.baseline) : 0
                    return sb - sa
                  })}
              />
            </div>
          ))}
        </div>
      ) : selected ? (
        <div className="explain-section">
          <h3 className="ethics-maps-title">Levers of frontier point #{selectedIndex + 1} (applied to every province)</h3>
          <LeverDeltaTable
            levers={selectedLevers.map((item) => {
              const meta = baselineByKey[item.key] || {}
              return { lever: item.key, value: item.value, baseline: item.baseline, min: meta.min, max: meta.max }
            })}
          />
        </div>
      ) : (
        <p className="control-text" style={{ marginTop: 8 }}>Click a point or a table row to inspect that policy's levers.</p>
      )}
    </React.Fragment>
  )
}

function App() {
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [mapPayload, setMapPayload] = useState(null)
  const [parameterMeta, setParameterMeta] = useState([])
  const [indicatorKeys, setIndicatorKeys] = useState([])
  const [provinceHistoryCache, setProvinceHistoryCache] = useState({})
  const [simulationRows, setSimulationRows] = useState([])
  const [latestStateRows, setLatestStateRows] = useState([])
  const [allocations, setAllocations] = useState({})
  const [selectedProvince, setSelectedProvince] = useState('')
  const [currentYear, setCurrentYear] = useState(null)
  const [selectedIndicator, setSelectedIndicator] = useState(DEFAULT_INDICATORS[0])
  const [baselineYear, setBaselineYear] = useState(null)
  const [simulationLog, setSimulationLog] = useState('')
  const [summary, setSummary] = useState(null)
  const [isSimulating, setIsSimulating] = useState(false)
  const [isLoadingTrends, setIsLoadingTrends] = useState(false)
  const [spendingIntensityPct, setSpendingIntensityPct] = useState(19.0)
  const [reservePool, setReservePool] = useState(0)
  const [models, setModels] = useState([])
  const [objectives, setObjectives] = useState([])
  const [provinces, setProvinces] = useState([])
  const [selectedModel, setSelectedModel] = useState('neural')
  const [selectedObjective, setSelectedObjective] = useState('utilitarian')
  // Per-objective tunable parameters (e.g. CVaR alpha, prioritarian rho),
  // keyed by objective id; seeded from each objective's declared defaults.
  const [objectiveParams, setObjectiveParams] = useState({})
  const [optimizeHorizon, setOptimizeHorizon] = useState(20)
  const [optimizeIterations, setOptimizeIterations] = useState(8)
  const [isOptimizing, setIsOptimizing] = useState(false)
  const [optimizeResult, setOptimizeResult] = useState(null)
  const [isComparing, setIsComparing] = useState(false)
  const [paretoResult, setParetoResult] = useState(null)
  const [isPareto, setIsPareto] = useState(false)
  // 1 = one uniform national lever vector; >1 = one lever vector per k-means
  // region (the clustered search that can actually target provinces).
  const [paretoClusters, setParetoClusters] = useState(1)
  const [compareResult, setCompareResult] = useState(null)
  const [adoptedCandidateId, setAdoptedCandidateId] = useState(null)
  const [runProgress, setRunProgress] = useState(null)
  const trendRequestsRef = useRef(new Set())

  useEffect(() => {
    const unsubscribe = window.seraApi.onSimulationLog((message) => {
      setSimulationLog((currentValue) => `${currentValue}${message}`)
    })
    return unsubscribe
  }, [])

  useEffect(() => {
    const unsubscribe = window.seraApi.onSimulationProgress((progress) => {
      setRunProgress(progress)
    })
    return unsubscribe
  }, [])

  useEffect(() => {
    let active = true

    async function load() {
      try {
        const [mapResponse, bootstrap] = await Promise.all([
          window.seraApi.loadProvinceMap(),
          window.seraApi.bootstrap(),
        ])

        if (!active) {
          return
        }

        const mapData = JSON.parse(mapResponse.raw)
        setMapPayload({ mapPath: mapResponse.mapPath, features: mapData.features || [] })
        setParameterMeta(bootstrap.parameterMeta || [])
        setIndicatorKeys(bootstrap.indicatorKeys || [])
        setLatestStateRows(bootstrap.latestStateRows || [])
        setAllocations(bootstrap.defaultAllocations || {})
        setCurrentYear(bootstrap.baselineYear)
        setBaselineYear(bootstrap.baselineYear)
        setSpendingIntensityPct(Number(bootstrap.spendingIntensityPct) || 19.0)
        setModels(bootstrap.models || [])
        setObjectives(bootstrap.objectives || [])
        const defaultObjectiveParams = {}
        for (const objective of bootstrap.objectives || []) {
          for (const param of objective.parameters || []) {
            defaultObjectiveParams[objective.id] = {
              ...(defaultObjectiveParams[objective.id] || {}),
              [param.id]: param.default,
            }
          }
        }
        setObjectiveParams(defaultObjectiveParams)
        setProvinces(bootstrap.provinces || [])
        if ((bootstrap.models || []).length) {
          const trainable = bootstrap.models.find((model) => model.trainable) || bootstrap.models[0]
          setSelectedModel(trainable.id)
        }
        if ((bootstrap.objectives || []).length) {
          setSelectedObjective(bootstrap.objectives[0].id)
        }

        const firstFeature = (mapData.features || [])[0]
        const firstProvince = (firstFeature && firstFeature.properties && firstFeature.properties.prov_acr)
          || ((bootstrap.provinces || [])[0])
        setSelectedProvince(String(firstProvince || '').trim().toUpperCase())
        setLoading(false)
      } catch (loadError) {
        if (!active) {
          return
        }
        setError(loadError.message || String(loadError))
        setLoading(false)
      }
    }

    load()
    return () => {
      active = false
    }
  }, [])

  const features = (mapPayload && mapPayload.features) || []

  const featureByProvince = useMemo(() => {
    const nextIndex = {}
    features.forEach((feature) => {
      const provinceCode = String(feature.properties.prov_acr || '').trim().toUpperCase()
      nextIndex[provinceCode] = feature
    })
    return nextIndex
  }, [features])

  const currentProvinceFeature = selectedProvince ? featureByProvince[selectedProvince] : null
  const selectedProvinceCache = provinceHistoryCache[selectedProvince] || EMPTY_PROVINCE_CACHE
  const historicalProvinceRows = selectedProvinceCache.rows || []
  const currentProvinceRows = useMemo(
    () => mergeHistoryRows(historicalProvinceRows, buildProvinceHistory(simulationRows, selectedProvince)),
    [historicalProvinceRows, selectedProvince, simulationRows],
  )

  const currentProvinceState = useMemo(
    () => latestStateRows.find((row) => row.area_code === selectedProvince) || null,
    [latestStateRows, selectedProvince],
  )

  const provinceAllocations = allocations[selectedProvince] || {}

  const mapVisibleIndicatorKeys = useMemo(() => {
    return indicatorKeys.length ? indicatorKeys : DEFAULT_INDICATORS
  }, [indicatorKeys])

  const topIndicators = useMemo(() => {
    return mapVisibleIndicatorKeys.slice(0, 4)
  }, [mapVisibleIndicatorKeys])

  const resourceBudget = useMemo(
    () => computeResourceBudget(allocations, parameterMeta, latestStateRows, spendingIntensityPct, reservePool),
    [allocations, parameterMeta, latestStateRows, spendingIntensityPct, reservePool],
  )

  useEffect(() => {
    let active = true

    async function loadProvinceTrendData() {
      if (!selectedProvince || !selectedIndicator || !currentYear) {
        return
      }

      const missingKeys = [selectedIndicator].filter((key) => !selectedProvinceCache.keysLoaded.includes(key))
      if (!missingKeys.length) {
        return
      }

      // Each bridge call spawns a Python process; skip if this exact request
      // is already in flight (e.g. the effect re-ran mid-fetch).
      const requestKey = `${selectedProvince}:${missingKeys.join(',')}`
      if (trendRequestsRef.current.has(requestKey)) {
        return
      }
      trendRequestsRef.current.add(requestKey)

      try {
        setIsLoadingTrends(true)
        const response = await window.seraApi.loadProvinceTrends({
          provinceCode: selectedProvince,
          indicatorKeys: missingKeys,
          startYear: 2016,
          endYear: currentYear,
        })

        if (!active) {
          return
        }

        setProvinceHistoryCache((currentValue) => {
          const existingEntry = currentValue[selectedProvince] || { rows: [], keysLoaded: [] }
          return {
            ...currentValue,
            [selectedProvince]: {
              rows: mergeHistoryRows(existingEntry.rows, response.rows || []),
              keysLoaded: Array.from(new Set([...(existingEntry.keysLoaded || []), ...(response.indicatorKeys || [])])),
            },
          }
        })
      } catch (trendError) {
        if (active) {
          setError(trendError.message || String(trendError))
        }
      } finally {
        trendRequestsRef.current.delete(requestKey)
        if (active) {
          setIsLoadingTrends(false)
        }
      }
    }

    loadProvinceTrendData()
    return () => {
      active = false
    }
  }, [currentYear, selectedIndicator, selectedProvince, selectedProvinceCache])

  function updateProvinceAllocation(parameterKey, nextValue) {
    setAllocations((currentValue) => ({
      ...currentValue,
      [selectedProvince]: {
        ...(currentValue[selectedProvince] || {}),
        [parameterKey]: nextValue,
      },
    }))
  }

  function resetProvinceAllocations() {
    setAllocations((currentValue) => ({
      ...currentValue,
      [selectedProvince]: Object.fromEntries(
        parameterMeta.map((parameter) => [parameter.key, parameter.baseline]),
      ),
    }))
  }

  async function runSimulation() {
    try {
      setIsSimulating(true)
      setSimulationLog('')
      setRunProgress({ percent: 0, message: 'Starting simulation...' })
      const response = await window.seraApi.simulateNextYear({
        currentYear,
        currentStateRows: latestStateRows,
        allocations,
        spendingIntensityPct,
        reservePool,
      })

      setLatestStateRows(response.nextStateRows || [])
      setSimulationRows((currentValue) => [...currentValue, ...(response.nextStateRows || [])])
      setSummary(response.summary || null)
      setCurrentYear(response.nextYear)
      if (hasValue(response.reservePool)) {
        setReservePool(Math.max(0, Number(response.reservePool) || 0))
      }
    } catch (simulationError) {
      setError(simulationError.message || String(simulationError))
    } finally {
      setIsSimulating(false)
      setRunProgress(null)
    }
  }

  async function runModel() {
    try {
      setIsOptimizing(true)
      setSimulationLog('')
      setOptimizeResult(null)
      setAdoptedCandidateId(null)
      setRunProgress({ percent: 0, message: 'Starting policy run...' })
      // The run only *proposes* candidates; nothing is applied to the twin
      // until the user explicitly adopts one (see adoptCandidate).
      const response = await window.seraApi.optimizePolicy({
        currentYear,
        currentStateRows: latestStateRows,
        modelId: selectedModel,
        objectiveId: selectedObjective,
        objectiveParams: objectiveParams[selectedObjective] || {},
        horizon: optimizeHorizon,
        iterations: optimizeIterations,
        spendingIntensityPct,
        reservePool,
      })
      setOptimizeResult(response)
    } catch (modelError) {
      setError(modelError.message || String(modelError))
    } finally {
      setIsOptimizing(false)
      setRunProgress(null)
    }
  }

  function adoptCandidate(candidate) {
    if (!optimizeResult || !candidate || adoptedCandidateId) {
      return
    }
    const trajectoryRows = candidate.trajectoryRows || []
    const finalRows = trajectoryRows.filter((row) => Number(row.year) === Number(optimizeResult.finalYear))

    setSimulationRows(trajectoryRows)
    if (finalRows.length) {
      setLatestStateRows(finalRows)
    }
    setCurrentYear(optimizeResult.finalYear)
    if (candidate.finalAllocations && Object.keys(candidate.finalAllocations).length) {
      setAllocations((prev) => ({ ...prev, ...candidate.finalAllocations }))
    }
    if (hasValue(candidate.reservePool)) {
      setReservePool(Math.max(0, Number(candidate.reservePool) || 0))
    }
    setSummary(candidate.summary || null)
    setAdoptedCandidateId(candidate.id)
  }

  async function runComparison() {
    try {
      setIsComparing(true)
      setSimulationLog('')
      setCompareResult(null)
      setRunProgress({ percent: 0, message: 'Starting ethics comparison...' })
      // Read-only what-if: the twin state, year, and allocations are untouched.
      const response = await window.seraApi.compareObjectives({
        currentYear,
        currentStateRows: latestStateRows,
        modelId: selectedModel,
        horizon: optimizeHorizon,
        iterations: optimizeIterations,
        spendingIntensityPct,
        reservePool,
      })
      setCompareResult(response)
    } catch (comparisonError) {
      setError(comparisonError.message || String(comparisonError))
    } finally {
      setIsComparing(false)
      setRunProgress(null)
    }
  }

  async function runPareto() {
    try {
      setIsPareto(true)
      setSimulationLog('')
      setParetoResult(null)
      setRunProgress({ percent: 0, message: 'Starting Pareto frontier search...' })
      // Read-only what-if: the twin state, year, and allocations are untouched.
      const response = await window.seraApi.paretoFront({
        currentYear,
        currentStateRows: latestStateRows,
        horizon: optimizeHorizon,
        iterations: optimizeIterations,
        spendingIntensityPct,
        reservePool,
        nClusters: paretoClusters,
      })
      setParetoResult(response)
    } catch (paretoError) {
      setError(paretoError.message || String(paretoError))
    } finally {
      setIsPareto(false)
      setRunProgress(null)
    }
  }

  if (loading) {
    return (
      <div className="loading">
        <div className="loading-card">
          <p className="eyebrow">SERA UI</p>
          <h2>Loading Italy map, provincial state, and allocator defaults</h2>
          <p className="control-text">
            The app is building the dashboard from the twin model and the reference province GeoJSON.
          </p>
        </div>
      </div>
    )
  }

  if (error) {
    return (
      <div className="loading">
        <div className="loading-card">
          <p className="eyebrow">SERA UI</p>
          <div className="error-box">{error}</div>
        </div>
      </div>
    )
  }

  return (
    <div className="app-shell">
      <section className="hero">
        <div className="hero-copy">
          <p className="eyebrow">React + Electron</p>
          <h1>SERA Provincial Control Room</h1>
          <p className="subtitle">
            Select any Italian province on the map, tune the annual allocators, and advance the digital twin one year at a time. The chart below keeps the historical series visible so each intervention can be read against the recent trend.
          </p>
          <div className="hero-guide">
            <h2>How to use the control room</h2>
            <ol>
              <li><strong>Pick a province</strong> on the map to open its allocator editor and current indicators.</li>
              <li><strong>Tune the allocators</strong> — spending levers draw from the shared national resource pool shown on the right.</li>
              <li><strong>Simulate the next year</strong> to advance the twin and read the result against the historical trend below.</li>
              <li><strong>Ask the AI for policies</strong>: choose a model and an ethical objective, then optimize. Candidates appear below with their trade-offs; nothing is applied until you adopt one.</li>
              <li><strong>Study the ethics</strong>: compare all frameworks side by side, or map the efficiency–equity frontier — both are what-if analyses that never touch the twin.</li>
              <li><strong>Watch the resource pool</strong>: the meter on the right shows the share of the national budget still available. Overspending is reined in automatically and can draw down the reserve before it is refused.</li>
              <li><strong>Read the chart</strong>: simulated years extend the historical series rather than replacing it, so every intervention stays legible against the long-run trend.</li>
            </ol>
          </div>
          <div className="hero-guide">
            <h2>Key terms</h2>
            <dl className="hero-glossary">
              <dt>Resource pool</dt>
              <dd>The shared national budget every province's spending levers draw from. The reserve absorbs moderate overspending; beyond it, allocations are constrained automatically.</dd>
              <dt>Ethical objective</dt>
              <dd>The fairness rule the optimizer maximises — from raw efficiency (utilitarian) to protecting the worst-off provinces (prioritarian, sufficientarian).</dd>
              <dt>Pareto frontier</dt>
              <dd>The set of policies where no province can gain without another losing — the efficiency–equity trade-off curve, mapped with NSGA-II.</dd>
            </dl>
          </div>
        </div>
        <div className="hero-controls">
          <div className="control-stack">
            <div>
              <div className="year-badge">Simulation year: {currentYear}</div>
            </div>
            <ResourceMeter
              used={resourceBudget.totalUsed}
              limit={resourceBudget.totalBasePool}
              reserve={reservePool}
              label="National resource pool"
            />
            <div>
              <p className="control-label">Selected province</p>
              <p className="control-text">
                {currentProvinceFeature
                  ? `${currentProvinceFeature.properties.prov_name} (${selectedProvince})`
                  : 'Select a province on the map.'}
              </p>
            </div>
            <button className="primary-button" onClick={runSimulation} disabled={isSimulating || isOptimizing || isComparing || isPareto || !latestStateRows.length}>
              {isSimulating ? 'Running simulation...' : `Simulate ${Number(currentYear) + 1}`}
            </button>
            {isSimulating && <RunProgress progress={runProgress} />}

            <div className="model-studio">
              <p className="control-label">AI policy model</p>
              <select
                className="indicator-select model-select"
                value={selectedModel}
                onChange={(event) => setSelectedModel(event.target.value)}
                disabled={isOptimizing || isComparing || isPareto}
              >
                {models.map((model) => (
                  <option key={model.id} value={model.id}>{model.label}</option>
                ))}
              </select>
              {(() => {
                const meta = models.find((model) => model.id === selectedModel)
                if (!meta) return null
                return (
                  <React.Fragment>
                    {meta.explainability ? (
                      <p className="model-desc"><ExplainBadge explainability={meta.explainability} /></p>
                    ) : null}
                    <p className="control-text model-desc">{meta.description}</p>
                  </React.Fragment>
                )
              })()}
              <p className="control-label">Ethical objective</p>
              <select
                className="indicator-select model-select"
                value={selectedObjective}
                onChange={(event) => setSelectedObjective(event.target.value)}
                disabled={isOptimizing || isComparing || isPareto}
              >
                {objectives.map((objective) => (
                  <option key={objective.id} value={objective.id}>{objective.label}</option>
                ))}
              </select>
              {(() => {
                const meta = objectives.find((objective) => objective.id === selectedObjective)
                return meta ? <p className="control-text model-desc">{meta.description}</p> : null
              })()}
              {(() => {
                const meta = objectives.find((objective) => objective.id === selectedObjective)
                const params = (meta && meta.parameters) || []
                if (!params.length) return null
                const current = objectiveParams[selectedObjective] || {}
                return (
                  <div className="model-inputs">
                    {params.map((param) => {
                      const value = hasValue(current[param.id]) ? current[param.id] : param.default
                      return (
                        <label className="model-field" key={param.id}>
                          <span>{param.label}: {Number(value).toFixed(2)}</span>
                          <input
                            type="range"
                            min={param.min}
                            max={param.max}
                            step={param.step}
                            value={value}
                            disabled={isOptimizing || isComparing || isPareto}
                            onChange={(event) => {
                              const next = Number(event.target.value)
                              setObjectiveParams((prev) => ({
                                ...prev,
                                [selectedObjective]: { ...(prev[selectedObjective] || {}), [param.id]: next },
                              }))
                            }}
                          />
                        </label>
                      )
                    })}
                  </div>
                )
              })()}
              <div className="model-inputs">
                <label className="model-field">
                  <span>Horizon (years)</span>
                  <input
                    type="number"
                    min={1}
                    max={50}
                    value={optimizeHorizon}
                    disabled={isOptimizing}
                    onChange={(event) => setOptimizeHorizon(clamp(Number(event.target.value) || 1, 1, 50))}
                  />
                </label>
                <label className="model-field">
                  <span>Training rounds</span>
                  <input
                    type="number"
                    min={1}
                    max={40}
                    value={optimizeIterations}
                    disabled={isOptimizing}
                    onChange={(event) => setOptimizeIterations(clamp(Number(event.target.value) || 1, 1, 40))}
                  />
                </label>
              </div>
              <button className="primary-button" onClick={runModel} disabled={isOptimizing || isSimulating || isComparing || isPareto || !latestStateRows.length}>
                {isOptimizing
                  ? 'Optimizing…'
                  : `Optimize ${(objectives.find((o) => o.id === selectedObjective) || {}).label || 'objective'} over ${optimizeHorizon}y`}
              </button>
              {isOptimizing && <RunProgress progress={runProgress} />}
              <button className="primary-button" onClick={runComparison} disabled={isOptimizing || isSimulating || isComparing || isPareto || !latestStateRows.length}>
                {isComparing ? 'Comparing frameworks…' : `Compare all ${objectives.length || 4} ethical frameworks`}
              </button>
              {isComparing && <RunProgress progress={runProgress} />}
              <div className="model-inputs">
                <label className="model-field">
                  <span>Pareto regions (1 = uniform national policy)</span>
                  <input
                    type="number"
                    min={1}
                    max={12}
                    value={paretoClusters}
                    disabled={isPareto}
                    onChange={(event) => setParetoClusters(clamp(Number(event.target.value) || 1, 1, 12))}
                  />
                </label>
              </div>
              <button className="primary-button" onClick={runPareto} disabled={isOptimizing || isSimulating || isComparing || isPareto || !latestStateRows.length}>
                {isPareto
                  ? 'Mapping the frontier…'
                  : (paretoClusters > 1
                    ? `Map the frontier — ${paretoClusters} regional packages`
                    : 'Map the efficiency–equity frontier (Pareto)')}
              </button>
              {isPareto && <RunProgress progress={runProgress} />}
            </div>
          </div>
        </div>
      </section>

      <section className="workspace">
        <div className="panel">
          <div className="panel-header">
            <div>
              <h2>Italy map</h2>
              <p>Province selection uses the same GeoJSON-driven SVG approach as your reference UI.</p>
            </div>
            <span className="status-pill active">{features.length} mapped provinces</span>
          </div>
          <ItalyMap
            features={features}
            allocations={allocations}
            parameterMeta={parameterMeta}
            selectedProvince={selectedProvince}
            onSelectProvince={setSelectedProvince}
          />
        </div>

        <aside className="panel">
          <div className="panel-header">
            <div>
              <h2>Allocator editor</h2>
              <p>Adjust the policy levers for the active province before running the next year.</p>
            </div>
            <span className="status-pill">{selectedProvince || 'No province'}</span>
          </div>

          {!selectedProvince || !currentProvinceFeature ? (
            <div className="empty-state">Choose a province on the map to inspect its allocators and indicator values.</div>
          ) : (
            <div className="editor-scroll">
              <div className="province-meta">
                <div>
                  <h3 className="province-title">{currentProvinceFeature.properties.prov_name}</h3>
                  <p className="province-region">{currentProvinceFeature.properties.reg_name} • {selectedProvince}</p>
                </div>

                <div className="kpi-grid">
                  {topIndicators.map((indicatorKey) => (
                    <div className="kpi-card" key={indicatorKey}>
                      <span className="kpi-label">{formatLabel(indicatorKey)}</span>
                      <span className="kpi-value">{formatNumber(currentProvinceState && currentProvinceState[indicatorKey])}</span>
                    </div>
                  ))}
                </div>
              </div>

              <div className="allocator-grid">
                {parameterMeta.map((parameter) => {
                  const rawValue = hasValue(provinceAllocations[parameter.key])
                    ? provinceAllocations[parameter.key]
                    : parameter.baseline
                  const value = Number(rawValue)
                  return (
                    <div className={`allocator-card${SPENDING_PARAMS.has(parameter.key) ? ' allocator-card--spending' : ''}`} key={parameter.key}>
                      <div className="allocator-head">
                        <span className="allocator-name">{parameter.label}</span>
                        <span className="allocator-value">{formatNumber(value)}</span>
                      </div>
                      <input
                        className="allocator-range"
                        type="range"
                        min={parameter.min}
                        max={parameter.max}
                        step={parameter.step}
                        value={value}
                        onChange={(event) => updateProvinceAllocation(parameter.key, Number(event.target.value))}
                      />
                      <div className="allocator-meta">
                        <span>Min {formatNumber(parameter.min)}</span>
                        <span>Baseline {formatNumber(parameter.baseline)}</span>
                        <span>Max {formatNumber(parameter.max)}</span>
                      </div>
                    </div>
                  )
                })}
              </div>

              <div className="allocator-actions">
                <button className="secondary-button" onClick={resetProvinceAllocations}>Reset province</button>
                <button className="ghost-button" onClick={() => setSelectedIndicator(DEFAULT_INDICATORS[0])}>Reset chart indicator</button>
              </div>
            </div>
          )}
        </aside>
      </section>

      <section className="chart-panel">
        <h2>Indicator trend</h2>
        <p className="chart-subtitle">
          Historical series and simulated future values for the selected province are plotted together.
        </p>
        <div className="indicator-picker">
          <select
            className="indicator-select"
            value={selectedIndicator}
            onChange={(event) => setSelectedIndicator(event.target.value)}
          >
            {indicatorKeys.map((key) => (
              <option key={key} value={key}>{formatLabel(key)}</option>
            ))}
          </select>
        </div>

        <div className="chart-plot-row">
          <div className="chart-plot-col">
            <TrendChart
              provinceName={(currentProvinceFeature && currentProvinceFeature.properties && currentProvinceFeature.properties.prov_name) || 'Province'}
              provinceCode={selectedProvince}
              rows={currentProvinceRows}
              indicatorKeys={[selectedIndicator]}
              simulationStartYear={simulationRows.length > 0 ? Number(baselineYear) + 1 : null}
            />
            {isLoadingTrends ? <div className="empty-state" style={{ marginTop: 12 }}>Loading trend data for the selected province...</div> : null}
          </div>
          <div className="indicator-map-panel">
            <div className="indicator-map-header">
              <h3 className="indicator-map-title">{formatLabel(selectedIndicator)}</h3>
              <p className="indicator-map-subtitle">Latest value per province</p>
            </div>
            <IndicatorMap
              features={features}
              latestStateRows={latestStateRows}
              indicatorKey={selectedIndicator}
              selectedProvince={selectedProvince}
              onSelectProvince={setSelectedProvince}
            />
          </div>
        </div>
      </section>

      {optimizeResult ? (
        <section className="chart-panel">
          <h2>Policy candidates</h2>
          <p className="chart-subtitle">
            The selected model proposes graded policy candidates optimized for the selected ethical objective — what "best for Italy" means is your choice, not the model's. Nothing is applied to the twin until you adopt a candidate.
          </p>
          <PolicyCandidates
            result={optimizeResult}
            models={models}
            onAdopt={adoptCandidate}
            adoptedId={adoptedCandidateId}
            disabled={isSimulating || isOptimizing || isComparing || isPareto}
          />
        </section>
      ) : null}

      <section className="chart-panel">
        <h2>Ethics equity dashboard</h2>
        <p className="chart-subtitle">
          Train the same model once per ethical framework from the current state and compare where each one takes Italy: total wealth (utilitarian view), inter-provincial inequality (Gini), and the floor (the worst-off province). The comparison is a what-if analysis — it never advances the twin.
        </p>
        {compareResult ? (
          <EthicsComparison compareResult={compareResult} models={models} features={features} />
        ) : (
          <div className="empty-state">
            Use "Compare all ethical frameworks" in the control panel above to train the selected model under every objective and see the trade-offs side by side.
          </div>
        )}
      </section>

      <section className="chart-panel">
        <h2>Efficiency–equity frontier</h2>
        <p className="chart-subtitle">
          NSGA-II evolves national lever vectors against three objectives at once — total GDP, inter-provincial inequality, and the worst-off province — and shows the whole Pareto frontier. Instead of asking which ethical framework is right, see exactly how much efficiency each unit of equity costs. Read-only: nothing is applied to the twin.
        </p>
        {paretoResult ? (
          <ParetoFrontier paretoResult={paretoResult} parameterMeta={parameterMeta} />
        ) : (
          <div className="empty-state">
            Use "Map the efficiency–equity frontier" in the control panel above to chart the trade-off space the ethical objectives live in.
          </div>
        )}
      </section>

      <section className="log-panel">
        <h2>Simulation output</h2>
        <p className="log-subtitle">The last run summary and bridge log appear here.</p>
        {summary ? (
          <div className="empty-state" style={{ marginBottom: 12 }}>
            <strong>Latest summary:</strong>{' '}
            Mean GDP per capita {formatNumber(summary.gdp_per_capita)}, mean income {formatNumber(summary.income)}, mean unemployment {formatNumber(summary.unemployment_rate)}, mean life expectancy {formatNumber(summary.life_expectancy)}.
          </div>
        ) : null}
        <div className="log-box">{simulationLog || 'Run the simulation to see the next-year execution log.'}</div>
      </section>
    </div>
  )
}

ReactDOM.createRoot(document.getElementById('root')).render(<App />)