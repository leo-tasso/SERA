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
  const [optimizeHorizon, setOptimizeHorizon] = useState(20)
  const [optimizeIterations, setOptimizeIterations] = useState(8)
  const [isOptimizing, setIsOptimizing] = useState(false)
  const [optimizeResult, setOptimizeResult] = useState(null)
  const [isComparing, setIsComparing] = useState(false)
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
              <p className="control-label">Map source</p>
              <p className="control-text">{mapPayload && mapPayload.mapPath}</p>
            </div>
            <div>
              <p className="control-label">Selected province</p>
              <p className="control-text">
                {currentProvinceFeature
                  ? `${currentProvinceFeature.properties.prov_name} (${selectedProvince})`
                  : 'Select a province on the map.'}
              </p>
            </div>
            <button className="primary-button" onClick={runSimulation} disabled={isSimulating || isOptimizing || isComparing || !latestStateRows.length}>
              {isSimulating ? 'Running simulation...' : `Simulate ${Number(currentYear) + 1}`}
            </button>
            {isSimulating && <RunProgress progress={runProgress} />}

            <div className="model-studio">
              <p className="control-label">AI policy model</p>
              <select
                className="indicator-select model-select"
                value={selectedModel}
                onChange={(event) => setSelectedModel(event.target.value)}
                disabled={isOptimizing || isComparing}
              >
                {models.map((model) => (
                  <option key={model.id} value={model.id}>{model.label}</option>
                ))}
              </select>
              {(() => {
                const meta = models.find((model) => model.id === selectedModel)
                return meta ? <p className="control-text model-desc">{meta.description}</p> : null
              })()}
              <p className="control-label">Ethical objective</p>
              <select
                className="indicator-select model-select"
                value={selectedObjective}
                onChange={(event) => setSelectedObjective(event.target.value)}
                disabled={isOptimizing || isComparing}
              >
                {objectives.map((objective) => (
                  <option key={objective.id} value={objective.id}>{objective.label}</option>
                ))}
              </select>
              {(() => {
                const meta = objectives.find((objective) => objective.id === selectedObjective)
                return meta ? <p className="control-text model-desc">{meta.description}</p> : null
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
              <button className="primary-button" onClick={runModel} disabled={isOptimizing || isSimulating || isComparing || !latestStateRows.length}>
                {isOptimizing
                  ? 'Optimizing…'
                  : `Optimize ${(objectives.find((o) => o.id === selectedObjective) || {}).label || 'objective'} over ${optimizeHorizon}y`}
              </button>
              {isOptimizing && <RunProgress progress={runProgress} />}
              <button className="primary-button" onClick={runComparison} disabled={isOptimizing || isSimulating || isComparing || !latestStateRows.length}>
                {isComparing ? 'Comparing frameworks…' : `Compare all ${objectives.length || 4} ethical frameworks`}
              </button>
              {isComparing && <RunProgress progress={runProgress} />}
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

      <section className="chart-panel">
        <h2>AI policy studio</h2>
        <p className="chart-subtitle">
          The selected model proposes graded policy candidates optimized for the selected ethical objective — what "best for Italy" means is your choice, not the model's. Nothing is applied to the twin until you adopt a candidate.
        </p>
        {optimizeResult ? (
          <PolicyCandidates
            result={optimizeResult}
            models={models}
            onAdopt={adoptCandidate}
            adoptedId={adoptedCandidateId}
            disabled={isSimulating || isOptimizing || isComparing}
          />
        ) : (
          <div className="empty-state">Run a model to get candidate policies with their trade-offs.</div>
        )}
      </section>

      <section className="chart-panel">
        <h2>Ethics equity dashboard</h2>
        <p className="chart-subtitle">
          Train the same model once per ethical framework from the current state and compare where each one takes Italy: total wealth (utilitarian view), inter-provincial inequality (Gini), and the floor (the worst-off province). The comparison is a what-if analysis — it never advances the twin.
        </p>
        {compareResult ? (
          <EthicsComparison compareResult={compareResult} models={models} features={features} />
        ) : (
          <div className="empty-state">
            Use "Compare all ethical frameworks" in the AI policy studio to train the selected model under every objective and see the trade-offs side by side.
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