import { useState, ChangeEvent, FormEvent, useMemo, useCallback, useEffect } from 'react'
import { 
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  LineChart, Line, Legend 
} from 'recharts'
import './App.css'

interface LibraryOption {
  value: string;
  label: string;
  icon: string;
}

export interface ImageCharacteristics {
  type: string
  size: string
  contrast: number
  noise: number
  channels: number
  mean_intensity: number
  edge_density: number
  complexity: number
}

export interface SegmentationMetrics {
  accuracy: number
  iou: number
  dice: number
  precision: number
  recall: number
  f1_score: number
  pixel_accuracy: number
  mae: number
  hausdorff_distance?: number | null
  predicted_area: number
  ground_truth_area: number
  true_positive: number
  false_positive: number
  true_negative: number
  false_negative: number
}

export interface Recommendation {
  method: string
  score: number
  estimated_time_ms: number
  estimated_iou: number
  best_for: string[]
}

export interface AnalysisData {
  histogram: number[]
  edge_density: number
  edges_b64: string
}

interface SegmentationResponse {
  success: boolean
  method: string
  library: string
  confidence: number
  elapsed_ms: number
  mask_b64: string
  overlay_b64: string
  chars: ImageCharacteristics
  metrics: SegmentationMetrics | null
  recommendations: Recommendation[]
  analysis: AnalysisData
  examples: Record<string, string[]>
}

interface MethodInfo {
  name: string; library: string; avg_iou: number; avg_time_ms: number
  memory_mb: number; robustness: number; description: string
  best_for: string[]; defaults: Record<string, any>
  schema: Record<string, { type: string; min?: number; max?: number; step?: number; default: any; label?: string }>
}

type GoalType = 'balanced' | 'speed' | 'accuracy' | 'low_memory'
type Tab  = 'results' | 'metrics' | 'recommendations' | 'analysis'
type Mode = 'classical' | 'neural'
type NeuralTask = 'semantic' | 'instance' | 'panoptic'

const NEURAL_TASKS = [
  { value: 'semantic', label: '🎨 Семантическая' },
  { value: 'instance', label: '🎭 Инстанс' },
  { value: 'panoptic', label: '🌐 Паноптическая' }
] as const;

const NEURAL_MODELS: Record<NeuralTask, string[]> = {
  semantic: ['segformer_b0','segformer_b1','segformer_b2','segformer_b3','segformer_b4','segformer_b5',
    'mask2former_swin_base','mask2former_swin_large','oneformer_swin_large','dpt_large',
    'upernet_convnext_small','unet_resnet34','unet_resnet50','unet_mit_b5',
    'fpn_mit_b5','psp_mit_b5','deeplab_resnet101','fcn_resnet50','segnet_resnet34','mobile_sam','sam2_tiny'],
  instance: ['mask2former_coco_instance','maskformer_resnet50','yolov8n_seg','yolov8s_seg','yolov8m_seg',
    'maskrcnn_resnet50','maskrcnn_resnet50_v2','mobile_sam','sam2_tiny'],
  panoptic: ['mask2former_ade_panoptic','mask2former_coco_panoptic','oneformer_coco_panoptic'],
}
const LIBRARIES: LibraryOption[] = [
    { value: "opencv", label: "OpenCV", icon: "🟢" },
    { value: "sklearn", label: "Scikit-learn", icon: "🔵" },
    { value: "torch", label: "PyTorch", icon: "🔴" },
  ];
const LIBRARIES_List = ['opencv','sklearn','torch'] as const
const API = 'http://localhost:8000'

// ──────────────────────── Helpers ──────────────────────────────────────────
const pct  = (n: number | null | undefined) => n == null ? '—' : `${(n * 100).toFixed(1)}%`
const fmt2 = (n: number | null | undefined) => n == null ? '—' : n.toFixed(2)
const fmt3 = (n: number | null | undefined) => n == null ? '—' : n.toFixed(3)

function MetricCard({ label, value, color = '#3b82f6' }: { label: string; value: string; color?: string }) {
  return (
    <div style={{ background: '#f8fafc', border: '1px solid #e2e8f0', borderRadius: 10,
      padding: '0.75rem 1rem', textAlign: 'center', borderTop: `3px solid ${color}` }}>
      <div style={{ fontSize: '0.75rem', color: '#64748b', marginBottom: 2 }}>{label}</div>
      <div style={{ fontSize: '1.35rem', fontWeight: 700, color: '#1e293b', fontVariantNumeric: 'tabular-nums' }}>{value}</div>
    </div>
  )
}

function Badge({ text, color = '#e0f2fe', textColor = '#0369a1' }: { text: string; color?: string; textColor?: string }) {
  return (
    <span style={{ background: color, color: textColor, fontSize: '0.7rem', padding: '2px 7px',
      borderRadius: 9999, fontWeight: 600, letterSpacing: '0.02em' }}>{text}</span>
  )
}

// ──────────────────────── Main App ─────────────────────────────────────────
export default function App() {
  const [file, setFile] = useState<File | null>(null)
  const [preview, setPreview] = useState<string>('')
  const [gtFile, setGtFile] = useState<File | null>(null)
  const [loading, setLoading] = useState<boolean>(false)
  const [res, setRes] = useState<SegmentationResponse | null>(null)
  const [err, setErr] = useState<string>('')
  const [activeTab, setActiveTab] = useState<Tab>('results')
  const [goal, setGoal] = useState<GoalType>('balanced')
  const [mode, setMode] = useState<Mode>('classical');
  const [autoSelect, setAutoSelect] = useState<boolean>(true);

  // Classical
  const [selectedLibrary, setSelectedLibrary] = useState<string>("opencv");
  const [selectedMethod, setSelectedMethod] = useState<string>("");
  const [availableMethods, setAvailableMethods] = useState<Record<string, MethodInfo>>({});
  const [methodSchema, setMethodSchema] = useState<Record<string, any>>({});
  const [customParams, setCustomParams] = useState<Record<string, any>>({});

  // Neural
  const [neuralTask, setNeuralTask]     = useState<NeuralTask>('semantic')
  const [neuralModel, setNeuralModel]   = useState('segformer_b2')

  useEffect(() => {
    if (autoSelect || mode === 'neural') return
    if (!autoSelect && selectedLibrary) {
      fetch(`${API}/api/methods?library=${selectedLibrary}`)
        .then(r => r.json())
        .then(data => {
          setAvailableMethods(data.methods ?? {});
          const firstMethod = Object.keys(data.methods ?? {})[0];
          if (firstMethod) setSelectedMethod(firstMethod); setCustomParams(data.methods[firstMethod]?.defaults ?? {})
        })
      .catch(() => {});
    }
  }, [selectedLibrary, autoSelect, mode]);

  useEffect(() => {
    if (mode === 'neural') {
      setNeuralModel(NEURAL_MODELS[neuralTask][0]);
    }
  }, [neuralTask, mode]);

  const handleMethodChange = (e: ChangeEvent<HTMLSelectElement>) => {
    const methodName = e.target.value;
    setSelectedMethod(methodName);
    const methodInfo = availableMethods[methodName];
    if (methodInfo) {
        setMethodSchema(methodInfo.schema || {});
        setCustomParams(methodInfo.defaults || {});
    }
  };

  const onFile = useCallback((e: ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0]
    if (!f) return
    if (preview) URL.revokeObjectURL(preview)
    setFile(f); setPreview(URL.createObjectURL(f)); setRes(null); setErr('')
  }, [preview])

  const onSubmit = async (e: FormEvent<HTMLFormElement>) => {
    e.preventDefault()
    if (!file) return
    
    setLoading(true)
    setErr('')
    setRes(null)
    
    const fd = new FormData()
    fd.append('file', file)
    fd.append('mode', mode);
    fd.append('goal', goal)
    fd.append('auto_select', String(autoSelect));
    fd.append('custom_params', JSON.stringify(customParams));

    if (mode === 'classical') {
      fd.append('library', selectedLibrary);
      if (!autoSelect) fd.append('method', selectedMethod);
    } else {
      fd.append('task', neuralTask);
      if (!autoSelect) fd.append('model', neuralModel);
    }

    if (gtFile) {
      fd.append('gt_mask', gtFile)
    }

    try {
      const r = await fetch(`${API}/api/segment`, {
        method: 'POST',
        body: fd
      })
      const raw = await r.text()
      console.log('📦 Raw response:', raw)
      if (!r.ok) throw new Error(`Ошибка сервера: ${r.status}: ${raw}`)
      const data: SegmentationResponse = JSON.parse(raw)
      console.log('✅ Parsed data:', data)
      setRes(data); setActiveTab('results')
    } catch (e: any) {
      console.error('❌ Fetch error:', e)
      setErr(e instanceof Error ? e.message : 'Неизвестная ошибка')
    } finally {
      setLoading(false)
    }
  }

const infoPanel = res && (
    <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.5rem', alignItems: 'center',
      background: '#f1f5f9', borderRadius: 10, padding: '0.6rem 1rem', marginBottom: '1rem', fontSize: '0.85rem' }}>
      <Badge text={res.method.toUpperCase()} color="#dbeafe" textColor="#1d4ed8" />
      <Badge text={res.library} color="#f3f4f6" textColor="#374151" />
      {res.chars.type && <Badge text={res.chars.type} color="#fef3c7" textColor="#92400e" />}
      <span style={{ color: '#64748b' }}>Уверенность: <b>{pct(res.confidence)}</b></span>
      <span style={{ color: '#64748b' }}>Время: <b>{res.elapsed_ms}мс</b></span>
      <span style={{ color: '#64748b' }}>Размер: <b>{res.chars.size}</b></span>
      <span style={{ color: '#64748b' }}>Контраст: <b>{fmt3(res.chars.contrast)}</b></span>
      <span style={{ color: '#64748b' }}>Шум: <b>{fmt3(res.chars.noise)}</b></span>
    </div>
  )

  return (
    <div style={{ fontFamily: "'JetBrains Mono', 'Fira Code', monospace", background: '#f8fafc',
      minHeight: '100vh', padding: '1.5rem 2rem', maxWidth: 1600, margin: '0 auto' }}>

      {/* ── Header ── */}
      <header style={{ marginBottom: '1.5rem', borderBottom: '2px solid #e2e8f0', paddingBottom: '1rem' }}>
        <h1 style={{ margin: 0, fontSize: '1.4rem', fontWeight: 800, letterSpacing: '-0.02em', color: '#0f172a' }}>
          ⚡🧠 AutoSegmenter <span style={{ color: '#3b82f6' }}>Pro</span>
        </h1>
        <p style={{ margin: '0.25rem 0 0', color: '#64748b', fontSize: '0.8rem' }}>
          Классические и нейросетевые методы сегментации
        </p>
      </header>
      <main>
      {/* ── Controls ── */}
      <form onSubmit={onSubmit}>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))',
          gap: '0.75rem', marginBottom: '1rem' }}>

          {/* Upload */}
          <label style={labelStyle}>
            <span style={labelText}>📷 Изображение</span>
            <input type="file" accept="image/*" onChange={onFile} required style={inputStyle} />
          </label>
          <label style={labelStyle}>
            <span style={labelText}>🎯 Ground Truth (опц.)</span>
            <input type="file" accept="image/*"
              onChange={e => setGtFile(e.target.files?.[0] || null)} style={inputStyle} />
          </label>

          {/* Mode */}
          <div style={labelStyle}>
            <span style={labelText}>Режим</span>
            <div style={{ display: 'flex', gap: '0.5rem' }}>
              {(['classical','neural'] as Mode[]).map(m => (
                <button key={m} type="button" onClick={() => setMode(m)}
                  style={{ ...modeBtn, background: mode === m ? '#1d4ed8' : '#f1f5f9',
                    color: mode === m ? 'white' : '#374151' }}>
                  {m === 'classical' ? '🔬 Классик' : '🧠 Нейро'}
                </button>
              ))}
            </div>
          </div>

          {/* Goal */}
          <div style={labelStyle}>
            <span style={labelText}>🎯 Цель</span>
            <select value={goal} onChange={e => setGoal(e.target.value as GoalType)} style={selectStyle}>
              <option value="balanced">⚖️ Баланс</option>
              <option value="speed">⚡ Скорость</option>
              <option value="accuracy">🎯 Точность</option>
              <option value="low_memory">💾 Память</option>
            </select>
          </div>

          {/* Auto-select toggle */}
          <div style={labelStyle}>
            <span style={labelText}>Выбор метода</span>
            <div style={{ display: 'flex', gap: '0.5rem' }}>
              {[true, false].map(v => (
                <button key={String(v)} type="button" onClick={() => setAutoSelect(v)}
                  style={{ ...modeBtn, background: autoSelect === v ? '#16a34a' : '#f1f5f9',
                    color: autoSelect === v ? 'white' : '#374151' }}>
                  {v ? '🤖 Авто' : '✍️ Ручной'}
                </button>
              ))}
            </div>
          </div>
        </div>

        {err && <div className="err">{err}</div>}

        {/* Classical method selector */}
        {mode === 'classical' && !autoSelect && (
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
            gap: '0.75rem', background: '#f1f5f9', borderRadius: 10, padding: '1rem', marginBottom: '1rem' }}>
            <div style={labelStyle}>
              <span style={labelText}>📚 Библиотека</span>
              <select value={selectedLibrary} onChange={e => { setSelectedLibrary(e.target.value); setSelectedMethod('') }} style={selectStyle}>
                {LIBRARIES.map(lib => (
                  <option key={lib.value} value={lib.value}>
                    {lib.icon} {lib.label}
                  </option>
                ))}
              </select>
            </div>
            <div style={labelStyle}>
              <span style={labelText}>⚙️ Метод</span>
              <select value={selectedMethod}
                onChange={e => {
                  setSelectedMethod(e.target.value)
                  setCustomParams(availableMethods[e.target.value]?.defaults ?? {})
                }} style={selectStyle}>
                {Object.entries(availableMethods).map(([k, m]) => (
                  <option key={k} value={k}>{m.name} {m.avg_iou > 0.8 ? '⭐' : ''}</option>
                ))}
              </select>
            </div>
            {selectedMethod && availableMethods[selectedMethod] && (
              <div style={{ gridColumn: '1 / -1', fontSize: '0.78rem', color: '#475569',
                background: '#e0f2fe', borderRadius: 8, padding: '0.5rem 0.75rem' }}>
                ⏱ {availableMethods[selectedMethod].avg_time_ms}мс &nbsp;|&nbsp;
                🎯 IoU {pct(availableMethods[selectedMethod].avg_iou)} &nbsp;|&nbsp;
                💾 {availableMethods[selectedMethod].memory_mb}МБ &nbsp;|&nbsp;
                🛡 Устойч. {pct(availableMethods[selectedMethod].robustness)}<br/>
                {availableMethods[selectedMethod].description}
              </div>
            )}
            {/* Param sliders */}
            {selectedMethod && availableMethods[selectedMethod]?.schema && Object.keys(availableMethods[selectedMethod].schema).length > 0 && (
              <div style={{ gridColumn: '1 / -1', display: 'grid',
                gridTemplateColumns: 'repeat(auto-fill, minmax(180px, 1fr))', gap: '0.5rem' }}>
                <h5 style={{ margin: '0 0 0.6rem', fontSize: '0.85rem', color: '#475569' }}>⚙️ Настройка параметров</h5>
                {Object.entries(availableMethods[selectedMethod].schema).map(([pk, cfg]) => (
                  <div key={pk} style={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
                    <label style={{ fontSize: '0.72rem', fontWeight: 600, color: '#374151' }}>
                      {cfg.label ?? pk}
                    </label>
                    {cfg.min !== undefined ? (
                      <div style={{ display: 'flex', gap: '0.4rem', alignItems: 'center' }}>
                        <input type="range" min={cfg.min} max={cfg.max} step={cfg.step ?? 1}
                          value={customParams[pk] ?? cfg.default}
                          onChange={e => setCustomParams(p => ({ ...p,
                            [pk]: cfg.type === 'int' ? parseInt(e.target.value) : parseFloat(e.target.value) }))}
                          style={{ flex: 1 }} />
                        <span style={{ fontSize: '0.72rem', minWidth: 36, textAlign: 'right', color: '#1d4ed8', fontWeight: 700 }}>
                          {customParams[pk] ?? cfg.default}
                        </span>
                      </div>
                    ) : (
                      <input type="number" step={cfg.step ?? 'any'} value={customParams[pk] ?? ''}
                        onChange={e => setCustomParams(p => ({ ...p,
                          [pk]: cfg.type === 'int' ? parseInt(e.target.value) : parseFloat(e.target.value) }))}
                        style={{ ...inputStyle, padding: '0.3rem' }} />
                    )}
                  </div>
                ))}
              </div>
            )}
          </div>
        )}

        {/* Neural selectors */}
        {mode === 'neural' && (
          <div style={{ display: 'flex', gap: '0.75rem', flexWrap: 'wrap',
            background: '#f1f5f9', borderRadius: 10, padding: '1rem', marginBottom: '1rem' }}>
            <div style={labelStyle}>
              <span style={labelText}>🎨 Задача</span>
              <select value={neuralTask} onChange={e => setNeuralTask(e.target.value as NeuralTask)} style={selectStyle}>
                <option value="semantic">🎨 Семантическая</option>
                <option value="instance">🎭 Инстанс</option>
                <option value="panoptic">🌐 Паноптическая</option>
              </select>
            </div>
            {!autoSelect && (
              <div style={labelStyle}>
                <span style={labelText}>🤖 Модель</span>
                <select value={neuralModel} onChange={e => setNeuralModel(e.target.value)} style={selectStyle}>
                  {NEURAL_MODELS[neuralTask].map(m => <option key={m} value={m}>{m}</option>)}
                </select>
              </div>
            )}
          </div>
        )}

        <button type="submit" disabled={!file || loading}
          style={{ background: loading ? '#93c5fd' : '#1d4ed8', color: 'white', border: 'none',
            borderRadius: 8, padding: '0.65rem 2rem', fontSize: '0.9rem', fontWeight: 700,
            cursor: !file || loading ? 'not-allowed' : 'pointer', letterSpacing: '0.02em' }}>
          {loading ? '⏳ Обработка…' : '▶ Запустить сегментацию'}
        </button>
      </form>

      {err && (
        <div style={{ background: '#fee2e2', color: '#991b1b', borderRadius: 8,
          padding: '0.75rem 1rem', margin: '0.75rem 0', fontSize: '0.85rem', fontWeight: 500 }}>
          ❌ {err}
        </div>
      )}

      {/* ── Info bar ── */}
      {infoPanel}

      {/* ── Tabs ── */}
      {res && (
        <div style={{ display: 'flex', gap: '0.4rem', borderBottom: '2px solid #e2e8f0',
          paddingBottom: '0.5rem', marginBottom: '1rem' }}>
          {([
            ['results',       '🖼 Результат'],
            ['metrics',       '📊 Метрики'],
            ['recommendations','💡 Рекомендации'],
            ['analysis',      '🔍 Анализ'],
          ] as [Tab, string][]).map(([t, label]) => (
            <button key={t} onClick={() => setActiveTab(t)} style={{
              border: 'none', background: activeTab === t ? '#1d4ed8' : '#f1f5f9',
              color: activeTab === t ? 'white' : '#374151', borderRadius: 7,
              padding: '0.4rem 0.9rem', fontWeight: 600, cursor: 'pointer',
              fontSize: '0.82rem', fontFamily: 'inherit' }}>
              {label}
            </button>
          ))}
        </div>
      )}

      {/* ── Results ── */}
      {activeTab === 'results' && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(380px, 1fr))', gap: '1rem' }}>
          {preview && <ImgCard title="📥 Оригинал" src={preview} />}
          {gtFile && <ImgCard title="🎯 Ground Truth" src={URL.createObjectURL(gtFile)} />}
          {res && <ImgCard title="🎨 Наложение" src={res.overlay_b64} />}
          {res && <ImgCard title="🔲 Маска" src={res.mask_b64} grayscale />}
          {res?.analysis.edges_b64 && <ImgCard title="📐 Границы" src={res.analysis.edges_b64} grayscale />}
        </div>
      )}

      {/* ── Metrics ── */}
      {activeTab === 'metrics' && res && (
        <div>
          {res.metrics ? (
            <>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(130px, 1fr))', gap: '0.65rem', marginBottom: '1rem' }}>
                <MetricCard label="IoU / Jaccard" value={pct(res.metrics.iou)} color="#3b82f6" />
                <MetricCard label="Dice Coeff."   value={pct(res.metrics.dice)} color="#8b5cf6" />
                <MetricCard label="Accuracy"      value={pct(res.metrics.accuracy)} color="#10b981" />
                <MetricCard label="Precision"     value={pct(res.metrics.precision)} color="#f59e0b" />
                <MetricCard label="Recall"        value={pct(res.metrics.recall)} color="#ef4444" />
                <MetricCard label="F1 Score"      value={pct(res.metrics.f1_score)} color="#ec4899" />
                <MetricCard label="Pixel Acc."    value={pct(res.metrics.pixel_accuracy)} color="#06b6d4" />
                <MetricCard label="MAE"           value={fmt3(res.metrics.mae)} color="#64748b" />
                {res.metrics.hausdorff_distance != null &&
                  <MetricCard label="Hausdorff" value={fmt2(res.metrics.hausdorff_distance)} color="#7c3aed" />}
              </div>
              {/* Confusion matrix */}
              <div style={{ background: 'white', borderRadius: 10, padding: '1rem', boxShadow: '0 1px 4px rgba(0,0,0,.08)' }}>
                <h4 style={{ margin: '0 0 0.6rem', fontSize: '0.85rem', color: '#475569' }}>Матрица ошибок</h4>
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.4rem', maxWidth: 280 }}>
                  {[
                    { l: `TP: ${res.metrics.true_positive}`,  bg: '#dcfce7', fg: '#166534' },
                    { l: `FP: ${res.metrics.false_positive}`, bg: '#fee2e2', fg: '#991b1b' },
                    { l: `FN: ${res.metrics.false_negative}`, bg: '#ffedd5', fg: '#9a3412' },
                    { l: `TN: ${res.metrics.true_negative}`,  bg: '#e0e7ff', fg: '#3730a3' },
                  ].map(c => (
                    <div key={c.l} style={{ background: c.bg, color: c.fg, padding: '0.6rem',
                      borderRadius: 6, textAlign: 'center', fontWeight: 700, fontSize: '0.85rem' }}>{c.l}</div>
                  ))}
                </div>
              </div>
            </>
          ) : (
            <div style={{ background: '#fef9c3', color: '#92400e', borderRadius: 8,
              padding: '1rem', fontSize: '0.85rem' }}>
              ℹ️ Загрузите Ground Truth маску для вычисления метрик качества.
            </div>
          )}
        </div>
      )}

      {/* ── Recommendations ── */}
      {activeTab === 'recommendations' && res && (
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.82rem' }}>
            <thead>
              <tr style={{ background: '#f1f5f9', borderBottom: '2px solid #e2e8f0' }}>
                {['#','Метод','Score','Время (мс)','Est. IoU','Лучше всего для'].map(h => (
                  <th key={h} style={{ padding: '0.6rem 0.8rem', textAlign: 'left',
                    fontWeight: 700, color: '#374151' }}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {res.recommendations.map((r, i) => (
                <tr key={r.method}
                  style={{ background: r.method === res.method ? '#eff6ff' : i % 2 === 0 ? 'white' : '#f8fafc',
                    borderLeft: r.method === res.method ? '3px solid #3b82f6' : '3px solid transparent' }}>
                  <td style={tdStyle}>{i + 1}</td>
                  <td style={tdStyle}>
                    <b>{r.method}</b> {r.method === res.method && <span style={{ color: '#16a34a' }}>✓</span>}
                  </td>
                  <td style={tdStyle}>{pct(r.score)}</td>
                  <td style={tdStyle}>{r.estimated_time_ms.toFixed(0)}</td>
                  <td style={tdStyle}>{pct(r.estimated_iou)}</td>
                  <td style={tdStyle}>
                    <div style={{ display: 'flex', gap: '0.25rem', flexWrap: 'wrap' }}>
                      {(r.best_for ?? []).map(b => (
                        <Badge key={b} text={b} color="#f0fdf4" textColor="#166534" />
                      ))}
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* ── Analysis ── */}
      {activeTab === 'analysis' && res && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(380px, 1fr))', gap: '1rem' }}>
          <div style={cardStyle}>
            <h4 style={cardTitle}>📈 Гистограмма интенсивностей</h4>
            <ResponsiveContainer width="100%" height={200}>
              <BarChart data={res.analysis.histogram.map((v, i) => ({ bin: i * 4, count: v }))}>
                <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
                <XAxis dataKey="bin" tick={{ fontSize: 10 }} label={{ value: 'Яркость', position: 'insideBottom', offset: -2, fontSize: 11 }} />
                <YAxis label={{value: 'Частота', angle: -90, position: 'insideLeft', fontSize: 11}} tick={{ fontSize: 10 }} />
                <Tooltip 
                  formatter={(v: number | undefined) => [v ?? 0, 'Частота']} 
                />
                <Bar dataKey="count" fill="#3b82f6" radius={[2,2,0,0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
          <div style={cardStyle}>
            <h4 style={cardTitle}>🔍 Характеристики изображения</h4>
            <div style={{ display: 'grid', gap: '0.4rem' }}>
              {Object.entries(res.chars).map(([k, v]) => (
                <div key={k} style={{ display: 'flex', justifyContent: 'space-between',
                  borderBottom: '1px dashed #e2e8f0', paddingBottom: '0.25rem', fontSize: '0.82rem' }}>
                  <span style={{ color: '#64748b' }}>{k}</span>
                  <span style={{ fontWeight: 600, color: '#1e293b' }}>
                    {typeof v === 'number' ? v.toFixed(4) : String(v)}
                  </span>
                </div>
              ))}
            </div>
          </div>
          {/* Примеры */}
          <div style={{ ...cardStyle, gridColumn: '1 / -1' }}>
            <h4 style={cardTitle}>📚 Рекомендуемые методы по типу сцены</h4>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(160px, 1fr))', gap: '0.75rem' }}>
              {Object.entries(res.examples).map(([type, ms]) => (
                <div key={type} style={{ background: '#f8fafc', borderRadius: 8, padding: '0.75rem' }}>
                  <div style={{ fontWeight: 700, fontSize: '0.8rem', marginBottom: '0.4rem', color: '#1e293b' }}>
                    {{ medical: '🏥 Медицина', documents: '📄 Документы',
                       nature: '🌿 Природа', industrial: '🏭 Индустрия' }[type] ?? type}
                  </div>
                  <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.25rem' }}>
                    {ms.map(m => (
                      <Badge key={m} text={m}
                        color={m === res.method ? '#1d4ed8' : '#e2e8f0'}
                        textColor={m === res.method ? 'white' : '#374151'} />
                    ))}
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}
      </main>
    </div>
  )
}

// ──────────────────────── Sub-components ───────────────────────────────────
function ImgCard({ title, src, grayscale = false }: { title: string; src: string; grayscale?: boolean }) {
  return (
    <div style={cardStyle}>
      <h4 style={cardTitle}>{title}</h4>
      <img src={src} alt={title} style={{ width: '100%', borderRadius: 6, marginTop: '0.5rem',
        filter: grayscale ? 'grayscale(1)' : undefined }} />
    </div>
  )
}

// ──────────────────────── Style helpers ────────────────────────────────────
const cardStyle: React.CSSProperties = {
  background: 'white', borderRadius: 10, padding: '1rem',
  boxShadow: '0 1px 4px rgba(0,0,0,.08)',
}
const cardTitle: React.CSSProperties = {
  margin: '0 0 0.25rem', fontSize: '0.85rem', fontWeight: 700, color: '#374151',
}
const labelStyle: React.CSSProperties = {
  display: 'flex', flexDirection: 'column', gap: '0.3rem',
}
const labelText: React.CSSProperties = {
  fontSize: '0.75rem', fontWeight: 700, color: '#475569', textTransform: 'uppercase', letterSpacing: '0.04em',
}
const inputStyle: React.CSSProperties = {
  padding: '0.45rem 0.6rem', border: '1px solid #cbd5e1', borderRadius: 7,
  fontSize: '0.82rem', background: 'white', fontFamily: 'inherit',
}
const selectStyle: React.CSSProperties = {
  ...inputStyle, cursor: 'pointer',
}
const modeBtn: React.CSSProperties = {
  border: 'none', borderRadius: 7, padding: '0.4rem 0.8rem',
  fontSize: '0.8rem', fontWeight: 600, cursor: 'pointer', fontFamily: 'inherit',
}
const tdStyle: React.CSSProperties = {
  padding: '0.55rem 0.8rem', borderBottom: '1px solid #f1f5f9',
}

