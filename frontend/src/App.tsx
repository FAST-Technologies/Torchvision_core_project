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

interface ValidationMethodResult {
  method: string;
  success: boolean;
  error?: string;
  validation_status?: 'PASS' | 'WARNING' | 'FAIL';
  iou?: number | null;
  dice?: number | null;
  pixel_accuracy?: number | null;
  precision?: number | null;
  recall?: number | null;
  f1_score?: number | null;
  mae?: number | null;
  hausdorff_distance?: number | null;
  primary_time?: number;
  reference_time?: number;
  time_diff?: number;
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

interface ValidationResponse {
  success: boolean;
  elapsed_ms: number;
  primary_library: string;
  reference_library: string;
  methods_tested: number;
  passed: number;
  warning: number;
  failed: number;
  results: ValidationMethodResult[];
  report_dir: string;
}

interface MethodInfo {
  name: string; library: string; avg_iou: number; avg_time_ms: number
  memory_mb: number; robustness: number; description: string
  best_for: string[]; defaults: Record<string, any>
  schema: Record<string, { type: string; min?: number; max?: number; step?: number; default: any; label?: string }>
}

type GoalType = 'balanced' | 'speed' | 'accuracy' | 'low_memory'
type Tab  = 'results' | 'metrics' | 'recommendations' | 'analysis' | 'validation'
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
const API = 'http://localhost:8000'

// ──────────────────────── Helpers ──────────────────────────────────────────
const pct  = (n: number | null | undefined) => n == null ? '—' : `${(n * 100).toFixed(1)}%`
const fmt2 = (n: number | null | undefined) => n == null ? '—' : n.toFixed(2)
const fmt3 = (n: number | null | undefined) => n == null ? '—' : n.toFixed(3)

function MetricCard({ label, value, color = 'blue' }: { label: string; value: string; color?: string }) {
  return (
    <div className={`metric-card metric-card--${color}`}>
      <div className="metric-card__label">{label}</div>
      <div className="metric-card__value">{value}</div>
    </div>
  )
}

// После компонента MetricCard, добавьте:
function ValidationStatusBadge({ status }: { status: ValidationMethodResult['validation_status'] }) {
  const config = {
    PASS: { bg: 'bg-green-100', text: 'text-green-800', label: '✅ PASS' },
    WARNING: { bg: 'bg-yellow-100', text: 'text-yellow-800', label: '⚠️ WARNING' },
    FAIL: { bg: 'bg-red-100', text: 'text-red-800', label: '❌ FAIL' },
  }[status || 'FAIL'];
  
  return (
    <span className={`px-2 py-0.5 rounded text-xs font-medium ${config.bg} ${config.text}`}>
      {config.label}
    </span>
  );
}

function ValidationResultsTable({ results }: { results: ValidationMethodResult[] }) {
  return (
    <div className="overflow-x-auto">
      <table className="data-table">
        <thead>
          <tr>
            <th>Метод</th>
            <th>Статус</th>
            <th>IoU</th>
            <th>Dice</th>
            <th>F1</th>
            <th>MAE</th>
            <th>Время (перв./реф.)</th>
          </tr>
        </thead>
        <tbody>
          {results.map((r) => (
            <tr key={r.method} className={!r.success ? 'opacity-50' : ''}>
              <td>
                <b>{r.method}</b>
                {!r.success && <span className="text-red-500 ml-2">❌ {r.error}</span>}
              </td>
              <td>{r.success && r.validation_status ? <ValidationStatusBadge status={r.validation_status} /> : '—'}</td>
              <td>{r.iou != null ? pct(r.iou) : '—'}</td>
              <td>{r.dice != null ? pct(r.dice) : '—'}</td>
              <td>{r.f1_score != null ? pct(r.f1_score) : '—'}</td>
              <td>{r.mae != null ? fmt3(r.mae) : '—'}</td>
              <td>
                {r.primary_time != null && r.reference_time != null 
                  ? `${r.primary_time.toFixed(2)}s / ${r.reference_time.toFixed(2)}s`
                  : '—'}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function Badge({ text, variant = 'info' }: { text: string; variant?: 'info' | 'success' | 'warning' | 'default' }) {
  return <span className={`badge badge--${variant}`}>{text}</span>
}

function ImgCard({ title, src, grayscale = false }: { title: string; src: string; grayscale?: boolean }) {
  return (
    <div className="img-card">
      <h4 className="img-card__title">{title}</h4>
      <img src={src} alt={title} className={grayscale ? 'img-card__image img-card__image--grayscale' : 'img-card__image'} />
    </div>
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

  // Validation
  const [validationLoading, setValidationLoading] = useState(false);
  const [validationResult, setValidationResult] = useState<ValidationResponse | null>(null);
  const [validationPrimaryLib, setValidationPrimaryLib] = useState('torch');
  const [validationReferenceLib, setValidationReferenceLib] = useState('opencv');
  const [validationFilter, setValidationFilter] = useState('all');

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

  useEffect(() => {
    if (mode === 'neural' && activeTab === 'validation') {
      setActiveTab('results');
      setValidationResult(null);
    }
  }, [mode, activeTab]);

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

  const onValidate = async () => {
    if (!file) {
      setErr('Сначала загрузите изображение');
      return;
    }
    
    setValidationLoading(true);
    setErr('');
    setValidationResult(null);
    
    const fd = new FormData();
    fd.append('file', file);
    fd.append('primary_library', validationPrimaryLib);
    fd.append('reference_library', validationReferenceLib);
    fd.append('methods_filter', validationFilter);
    
    try {
      const r = await fetch(`${API}/api/validate`, {
        method: 'POST',
        body: fd,
      });
      
      const raw = await r.text();
      if (!r.ok) throw new Error(`Ошибка валидации: ${r.status}: ${raw}`);
      
      const data: ValidationResponse = JSON.parse(raw);
      setValidationResult(data);
      
    } catch (e: any) {
      console.error('❌ Validation error:', e);
      setErr(e.message || 'Неизвестная ошибка валидации');
    } finally {
      setValidationLoading(false);
    }
  };

  const infoPanel = res && (
    <div className="info-panel">
      <Badge text={res.method.toUpperCase()} variant="info" />
      <Badge text={res.library} variant="default" />
      {res.chars.type && <Badge text={res.chars.type} variant="warning" />}
      <span className="info-panel__item">Уверенность: <b>{pct(res.confidence)}</b></span>
      <span className="info-panel__item">Время: <b>{res.elapsed_ms}мс</b></span>
      <span className="info-panel__item">Размер: <b>{res.chars.size}</b></span>
      <span className="info-panel__item">Контраст: <b>{fmt3(res.chars.contrast)}</b></span>
      <span className="info-panel__item">Шум: <b>{fmt3(res.chars.noise)}</b></span>
    </div>
  )

  return (
    <div className="app">
      {/* ── Header ── */}
      <header className="app-header">
        <h1 className="app-header__title">⚡🧠 AutoSegmenter <span className="app-header__accent">Pro</span></h1>
        <p className="app-header__subtitle">Классические и нейросетевые методы сегментации</p>
      </header>

      <main className="app-main">
        {/* ── Controls ── */}
        <form onSubmit={onSubmit} className="controls">
          <div className="controls__grid">
            <label className="control-group">
              <span className="control-group__label">📷 Изображение</span>
              <input type="file" accept="image/*" onChange={onFile} required className="control-input" />
            </label>
            <label className="control-group">
              <span className="control-group__label">🎯 Ground Truth (опц.)</span>
              <input type="file" accept="image/*"
                onChange={e => setGtFile(e.target.files?.[0] || null)} className="control-input" />
            </label>

            {/* Mode */}
            <div className="control-group">
              <span className="control-group__label">Режим</span>
              <div className="toggle-group">
                {(['classical','neural'] as Mode[]).map(m => (
                  <button key={m} type="button" onClick={() => setMode(m)}
                    className={`toggle-btn ${mode === m ? 'toggle-btn--active' : ''}`}>
                    {m === 'classical' ? '🔬 Классик' : '🧠 Нейро'}
                  </button>
                ))}
              </div>
            </div>

            {/* Goal */}
            <div className="control-group">
              <span className="control-group__label">🎯 Цель</span>
              <select value={goal} onChange={e => setGoal(e.target.value as GoalType)} className="control-select">
                <option value="balanced">⚖️ Баланс</option>
                <option value="speed">⚡ Скорость</option>
                <option value="accuracy">🎯 Точность</option>
                <option value="low_memory">💾 Память</option>
              </select>
            </div>

            {/* Auto-select */}
            <div className="control-group">
              <span className="control-group__label">Выбор метода</span>
              <div className="toggle-group">
                {[true, false].map(v => (
                  <button key={String(v)} type="button" onClick={() => setAutoSelect(v)}
                    className={`toggle-btn ${autoSelect === v ? 'toggle-btn--active toggle-btn--success' : ''}`}>
                    {v ? '🤖 Авто' : '✍️ Ручной'}
                  </button>
                ))}
              </div>
            </div>
          </div>

          {/* Classical method selector */}
          {mode === 'classical' && !autoSelect && (
            <div className="method-selector">
              <div className="controls__grid">
                <div className="control-group">
                  <span className="control-group__label">📚 Библиотека</span>
                  <select value={selectedLibrary} 
                    onChange={e => { setSelectedLibrary(e.target.value); setSelectedMethod('') }} 
                    className="control-select">
                    {LIBRARIES.map(lib => (
                      <option key={lib.value} value={lib.value}>{lib.icon} {lib.label}</option>
                    ))}
                  </select>
                </div>
                <div className="control-group">
                  <span className="control-group__label">⚙️ Метод</span>
                  <select value={selectedMethod} onChange={handleMethodChange} className="control-select">
                    {Object.entries(availableMethods).map(([k, m]) => (
                      <option key={k} value={k}>{m.name} {m.avg_iou > 0.8 ? '⭐' : ''}</option>
                    ))}
                  </select>
                </div>
                {selectedMethod && availableMethods[selectedMethod] && (
                  <div className="method-hint">
                    ⏱ {availableMethods[selectedMethod].avg_time_ms}мс &nbsp;|&nbsp;
                    🎯 IoU {pct(availableMethods[selectedMethod].avg_iou)} &nbsp;|&nbsp;
                    💾 {availableMethods[selectedMethod].memory_mb}МБ &nbsp;|&nbsp;
                    🛡 Устойч. {pct(availableMethods[selectedMethod].robustness)}<br/>
                    {availableMethods[selectedMethod].description}
                  </div>
                )}
                {/* Param sliders */}
                {selectedMethod && availableMethods[selectedMethod]?.schema && Object.keys(availableMethods[selectedMethod].schema).length > 0 && (
                  <div className="params-grid">
                    <h5 className="params-grid__title">⚙️ Настройка параметров</h5>
                    {Object.entries(availableMethods[selectedMethod].schema).map(([pk, cfg]) => (
                      <div key={pk} className="param-item">
                        <label className="param-item__label">{cfg.label ?? pk}</label>
                        {cfg.min !== undefined ? (
                          <div className="param-item__slider">
                            <input type="range" min={cfg.min} max={cfg.max} step={cfg.step ?? 1}
                              value={customParams[pk] ?? cfg.default}
                              onChange={e => setCustomParams(p => ({ ...p,
                                [pk]: cfg.type === 'int' ? parseInt(e.target.value) : parseFloat(e.target.value) }))}
                              className="range-input" />
                            <span className="range-value">{customParams[pk] ?? cfg.default}</span>
                          </div>
                        ) : (
                          <input type="number" step={cfg.step ?? 'any'} value={customParams[pk] ?? ''}
                            onChange={e => setCustomParams(p => ({ ...p,
                              [pk]: cfg.type === 'int' ? parseInt(e.target.value) : parseFloat(e.target.value) }))}
                            className="control-input control-input--small" />
                        )}
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </div>
          )}

          {/* Neural selectors */}
          {mode === 'neural' && (
            <div className="method-selector">
              <div className="controls__row">
                <div className="control-group">
                  <span className="control-group__label">🎨 Задача</span>
                  <select value={neuralTask} onChange={e => setNeuralTask(e.target.value as NeuralTask)} className="control-select">
                    {NEURAL_TASKS.map(t => <option key={t.value} value={t.value}>{t.label}</option>)}
                  </select>
                </div>
                {!autoSelect && (
                  <div className="control-group">
                    <span className="control-group__label">🤖 Модель</span>
                    <select value={neuralModel} onChange={e => setNeuralModel(e.target.value)} className="control-select">
                      {NEURAL_MODELS[neuralTask].map(m => <option key={m} value={m}>{m}</option>)}
                    </select>
                  </div>
                )}
              </div>
            </div>
          )}

          <button type="submit" disabled={!file || loading} className="submit-btn">
            {loading ? '⏳ Обработка…' : '▶ Запустить сегментацию'}
          </button>
        </form>

        {err && <div className="error-banner">❌ {err}</div>}
        {infoPanel}

        {/* ── Tabs ── */}
        {res && (
          <div className="tabs">
            {([
              ['results', '🖼 Результат'],
              ['metrics', '📊 Метрики'],
              ['recommendations','💡 Рекомендации'],
              ['analysis', '🔍 Анализ'],
              ...(mode === 'classical' ? [['validation', '🔬 Валидация'] as [Tab, string]] : []),
            ] as [Tab, string][]).map(([t, label]) => (
              <button 
                key={t} 
                onClick={() => {
                  setActiveTab(t); 
                  setValidationResult(null);
                  if (mode === 'neural' && t === 'validation') setActiveTab('results');
                }} 
                className={`tab-btn ${activeTab === t ? 'tab-btn--active' : ''}`}>
                {label}
              </button>
            ))}
          </div>
        )}

        {activeTab === 'validation' && (
          <div className="validation-tab">
            <div className="card">
              <h3 className="card__title">🔬 Кросс-библиотечная валидация</h3>
              <p className="text-sm text-gray-600 mb-4">
                Сравните реализации методов сегментации между библиотеками. 
                Результаты сохраняются в <code className="bg-gray-100 px-1 rounded">{validationResult?.report_dir}</code>
              </p>
              
              {/* Настройки валидации */}
              <div className="validation-controls">
                <div className="controls__grid">
                  <div className="control-group">
                    <label className="control-group__label">🔹 Первичная библиотека</label>
                    <select 
                      value={validationPrimaryLib}
                      onChange={(e) => setValidationPrimaryLib(e.target.value)}
                      className="control-select"
                    >
                      <option value="torch">🔴 PyTorch</option>
                      <option value="opencv">🟢 OpenCV</option>
                      <option value="sklearn">🔵 Scikit-learn</option>
                    </select>
                  </div>
                  
                  <div className="control-group">
                    <label className="control-group__label">⚪ Референсная библиотека</label>
                    <select 
                      value={validationReferenceLib}
                      onChange={(e) => setValidationReferenceLib(e.target.value)}
                      className="control-select"
                    >
                      <option value="opencv">🟢 OpenCV</option>
                      <option value="sklearn">🔵 Scikit-learn</option>
                      <option value="torch">🔴 PyTorch</option>
                    </select>
                  </div>
                  
                  <div className="control-group">
                    <label className="control-group__label">🔍 Фильтр методов</label>
                    <select 
                      value={validationFilter}
                      onChange={(e) => setValidationFilter(e.target.value)}
                      className="control-select"
                    >
                      <option value="all">📦 Все методы</option>
                      <option value="threshold">🎚 Пороговые</option>
                      <option value="edge">✏️ Граничные</option>
                      <option value="region">🔷 Региональные</option>
                      <option value="clustering">🔵 Кластеризация</option>
                    </select>
                  </div>
                </div>
                
                <button 
                  onClick={onValidate}
                  disabled={!file || validationLoading}
                  className="submit-btn submit-btn--secondary mt-4"
                >
                  {validationLoading ? '⏳ Валидация…' : '▶ Запустить валидацию'}
                </button>
              </div>
            </div>
            
            {/* Результаты валидации */}
            {validationResult && (
              <div className="card mt-4">
                <div className="flex justify-between items-center mb-4">
                  <h4 className="card__title">📊 Результаты</h4>
                  <div className="text-sm text-gray-500">
                    ⏱ {validationResult.elapsed_ms}мс | 
                    📈 {validationResult.methods_tested} методов |
                    ✅ {validationResult.passed} PASS |
                    ⚠️ {validationResult.warning} WARNING |
                    ❌ {validationResult.failed} FAIL
                  </div>
                </div>
                
                {/* Сводная статистика */}
                <div className="metrics-grid mb-4">
                  <MetricCard label="Совпадение (IoU)" value={pct(validationResult.results.filter(r => r.iou != null && r.iou >= 0.8).length / validationResult.results.filter(r => r.iou != null).length || 0)} color="blue" />
                  <MetricCard label="Среднее время" value={`${(validationResult.results.reduce((a, b) => a + (b.primary_time || 0), 0) / validationResult.results.length).toFixed(2)}s`} color="green" />
                  <MetricCard label="Успешных" value={`${validationResult.passed}/${validationResult.methods_tested}`} color="success" />
                </div>
                
                {/* Таблица результатов */}
                <ValidationResultsTable results={validationResult.results} />
                
                {/* Ссылка на полный отчёт */}
                <div className="mt-4 p-3 bg-gray-50 rounded text-sm">
                  💾 Полный отчёт сохранён в: <code className="bg-white px-1 rounded">{validationResult.report_dir}</code>
                </div>
              </div>
            )}
          </div>
        )}

        {/* ── Results ── */}
        {activeTab === 'results' && (
          <div className="results-grid">
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
                <div className="metrics-grid">
                  <MetricCard label="IoU / Jaccard" value={pct(res.metrics.iou)} color="blue" />
                  <MetricCard label="Dice Coeff." value={pct(res.metrics.dice)} color="purple" />
                  <MetricCard label="Accuracy" value={pct(res.metrics.accuracy)} color="green" />
                  <MetricCard label="Precision" value={pct(res.metrics.precision)} color="amber" />
                  <MetricCard label="Recall" value={pct(res.metrics.recall)} color="red" />
                  <MetricCard label="F1 Score" value={pct(res.metrics.f1_score)} color="pink" />
                  <MetricCard label="Pixel Acc." value={pct(res.metrics.pixel_accuracy)} color="cyan" />
                  <MetricCard label="MAE" value={fmt3(res.metrics.mae)} color="slate" />
                  {res.metrics.hausdorff_distance != null &&
                    <MetricCard label="Hausdorff" value={fmt2(res.metrics.hausdorff_distance)} color="violet" />}
                </div>
                <div className="confusion-matrix">
                  <h4 className="confusion-matrix__title">Матрица ошибок</h4>
                  <div className="confusion-grid">
                    {[
                      { l: `TP: ${res.metrics.true_positive}`, variant: 'tp' },
                      { l: `FP: ${res.metrics.false_positive}`, variant: 'fp' },
                      { l: `FN: ${res.metrics.false_negative}`, variant: 'fn' },
                      { l: `TN: ${res.metrics.true_negative}`, variant: 'tn' },
                    ].map(c => (
                      <div key={c.l} className={`confusion-cell confusion-cell--${c.variant}`}>{c.l}</div>
                    ))}
                  </div>
                </div>
              </>
            ) : (
              <div className="info-banner info-banner--warning">
                ℹ️ Загрузите Ground Truth маску для вычисления метрик качества.
              </div>
            )}
          </div>
        )}

        {/* ── Recommendations ── */}
        {activeTab === 'recommendations' && res && (
          <div className="table-container">
            <table className="data-table">
              <thead>
                <tr>
                  {['#','Метод','Score','Время (мс)','Est. IoU','Лучше всего для'].map(h => (
                    <th key={h}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {res.recommendations.map((r, i) => (
                  <tr key={r.method} className={r.method === res.method ? 'data-table__row--selected' : ''}>
                    <td>{i + 1}</td>
                    <td><b>{r.method}</b> {r.method === res.method && <span className="text-success">✓</span>}</td>
                    <td>{pct(r.score)}</td>
                    <td>{r.estimated_time_ms.toFixed(0)}</td>
                    <td>{pct(r.estimated_iou)}</td>
                    <td>
                      <div className="badge-group">
                        {(r.best_for ?? []).map(b => (
                          <Badge key={b} text={b} variant="success" />
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
          <div className="analysis-grid">
            <div className="card">
              <h4 className="card__title">📈 Гистограмма интенсивностей</h4>
              <ResponsiveContainer width="100%" height={200}>
                <BarChart data={res.analysis.histogram.map((v, i) => ({ bin: i * 4, count: v }))}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
                  <XAxis dataKey="bin" tick={{ fontSize: 10 }} label={{ value: 'Яркость', position: 'insideBottom', offset: -2, fontSize: 11 }} />
                  <YAxis label={{value: 'Частота', angle: -90, position: 'insideLeft', fontSize: 11}} tick={{ fontSize: 10 }} />
                  <Tooltip formatter={(v: number | undefined) => [v ?? 0, 'Частота']} />
                  <Bar dataKey="count" fill="#3b82f6" radius={[2,2,0,0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
            <div className="card">
              <h4 className="card__title">🔍 Характеристики изображения</h4>
              <div className="chars-list">
                {Object.entries(res.chars).map(([k, v]) => (
                  <div key={k} className="char-item">
                    <span className="char-item__key">{k}</span>
                    <span className="char-item__value">
                      {typeof v === 'number' ? v.toFixed(4) : String(v)}
                    </span>
                  </div>
                ))}
              </div>
            </div>
            <div className="card card--full">
              <h4 className="card__title">📚 Рекомендуемые методы по типу сцены</h4>
              <div className="examples-grid">
                {Object.entries(res.examples).map(([type, ms]) => (
                  <div key={type} className="example-card">
                    <div className="example-card__title">
                      {{ medical: '🏥 Медицина', documents: '📄 Документы',
                         nature: '🌿 Природа', industrial: '🏭 Индустрия' }[type] ?? type}
                    </div>
                    <div className="badge-group">
                      {ms.map(m => (
                        <Badge key={m} text={m} variant={m === res.method ? 'info' : 'default'} />
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

