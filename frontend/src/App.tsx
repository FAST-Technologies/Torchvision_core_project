import { useState, ChangeEvent, FormEvent, useMemo, useEffect } from 'react'
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
  hausdorff_distance?: number
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
  confidence: number
  mask_b64: string
  overlay_b64: string
  chars: ImageCharacteristics
  metrics: SegmentationMetrics | null
  recommendations: Recommendation[]
  analysis: AnalysisData
  examples: Record<string, string[]>
}

type GoalType = 'balanced' | 'speed' | 'accuracy' | 'low_memory'

const NEURAL_TASKS = [
  { value: 'semantic', label: '🎨 Семантическая' },
  { value: 'instance', label: '🎭 Инстанс' },
  { value: 'panoptic', label: '🌐 Паноптическая' }
] as const;

const NEURAL_MODELS = {
  semantic: [
    // SegFormer
    'segformer_b0', 'segformer_b1', 'segformer_b2', 
    'segformer_b3', 'segformer_b4', 'segformer_b5',
    // Mask2Former
    'mask2former_swin_base', 'mask2former_swin_large',
    // OneFormer
    'oneformer_swin_large',
    // DPT
    'dpt_large',
    // UPerNet
    'upernet_convnext_small',
    // SMP U-Net
    'unet_resnet34', 'unet_resnet50', 'unet_efficientnet_b0', 'unet_mit_b5',
    // SMP FPN
    'fpn_mit_b5', 'fpn_efficientnet',
    // SMP PSPNet
    'psp_mit_b5', 'psp_resnet50',
    // DeepLab
    'deeplab_resnet101',
    // FCN
    'fcn_resnet50', 'fcn_resnet101',
    // SegNet
    'segnet_resnet34', 'segnet_resnet50',
    // SAM (конвертирует instance → semantic)
    'mobile_sam', 'sam2_tiny',
  ],
  instance: [
    // Mask2Former
    'mask2former_coco_instance',
    // MaskFormer
    'maskformer_resnet50',
    // YOLOv8
    'yolov8n_seg', 'yolov8s_seg', 'yolov8m_seg',
    // Mask R-CNN
    'maskrcnn_resnet50', 'maskrcnn_resnet50_v2',
    // SAM для инстанс-сегментации
    'mobile_sam', 'sam2_tiny',
  ],
  panoptic: [
    // Mask2Former
    'mask2former_ade_panoptic', 'mask2former_coco_panoptic',
    // OneFormer
    'oneformer_coco_panoptic',
  ]
} as const;

function App() {
  const LIBRARIES: LibraryOption[] = [
    { value: "opencv", label: "OpenCV", icon: "🟢" },
    { value: "sklearn", label: "Scikit-learn", icon: "🔵" },
    { value: "torch", label: "PyTorch", icon: "🔴" },
  ];
  const [file, setFile] = useState<File | null>(null)
  const [preview, setPreview] = useState<string | null>(null)
  const [goal, setGoal] = useState<GoalType>('balanced')
  const [loading, setLoading] = useState<boolean>(false)
  const [res, setRes] = useState<SegmentationResponse | null>(null)
  const [err, setErr] = useState<string>('')
  const [activeTab, setActiveTab] = useState<'results' | 'metrics' | 'recommendations' | 'analysis'>('results')
  const [gtFile, setGtFile] = useState<File | null>(null)
  const [autoSelect, setAutoSelect] = useState(true);
  const [selectedLibrary, setSelectedLibrary] = useState<string>("opencv");
  const [selectedMethod, setSelectedMethod] = useState<string>("");
  const [availableMethods, setAvailableMethods] = useState<Record<string, any>>({});
  const [methodSchema, setMethodSchema] = useState<Record<string, any>>({});
  const [customParams, setCustomParams] = useState<Record<string, any>>({});
  const [mode, setMode] = useState<'classical' | 'neural'>('classical');
  const [neuralTask, setNeuralTask] = useState<'semantic' | 'instance' | 'panoptic'>('semantic');
  const [neuralModel, setNeuralModel] = useState('segformer_b2');

  const fmt = (n: number, decimals = 3) => n.toFixed(decimals)
  const pct = (n: number) => `${(n * 100).toFixed(1)}%`

  useEffect(() => {
    if (mode === 'neural') {
      setNeuralModel(NEURAL_MODELS[neuralTask][0]);
    }
  }, [mode, neuralTask]);

  useEffect(() => {
    if (!autoSelect && selectedLibrary) {
      fetch(`http://localhost:8000/api/methods?library=${selectedLibrary}`)
        .then(r => r.json())
        .then(data => {
          setAvailableMethods(data.methods);
          // Автовыбор первого метода
          const firstMethod = Object.keys(data.methods)[0];
          if (firstMethod) setSelectedMethod(firstMethod);
        });
    }
  }, [autoSelect, selectedLibrary]);

  const handleMethodChange = (e: ChangeEvent<HTMLSelectElement>) => {
    const methodName = e.target.value;
    setSelectedMethod(methodName);
    
    // Получаем схему и дефолты из загруженных методов
    const methodInfo = availableMethods[methodName];
    if (methodInfo) {
        setMethodSchema(methodInfo.schema || {});
        // Инициализируем стейт дефолтными значениями
        setCustomParams(methodInfo.defaults || {});
    }
  };

  const onChange = (e: ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0]
    if (f) {
      setFile(f)
      setPreview(URL.createObjectURL(f))
      setRes(null)
      setErr('')
    }
  }

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
      const r = await fetch('http://localhost:8000/api/segment', {
        method: 'POST',
        body: fd
      })
      const raw = await r.text()
      console.log('📦 Raw response:', raw)
      if (!r.ok) throw new Error(`Ошибка сервера: ${r.status}`)
      const data: SegmentationResponse = JSON.parse(raw)
      console.log('✅ Parsed data:', data)
      setRes(data)
    } catch (e: unknown) {
      console.error('❌ Fetch error:', e)
      setErr(e instanceof Error ? e.message : 'Неизвестная ошибка')
    } finally {
      setLoading(false)
    }
  }

  const cleanupPreview = () => {
    if (preview) URL.revokeObjectURL(preview)
  }

  return (
    <div className="app">
      <header>
        <h1>🧠 AutoSegmenter Pro</h1>
        <p>Production UI</p>
      </header>
      <main>
        <form onSubmit={onSubmit} className="controls">
          <div className="upload-group">
            <label>📷 Исходное изображение:</label>
            <input type="file" accept="image/*" onChange={onChange} required />
          </div>
          <div className="upload-group">
            <label>🎯 Ground Truth (опционально, для метрик):</label>
            <input type="file" accept="image/*" onChange={(e) => setGtFile(e.target.files?.[0] || null)} />
          </div>
          <div className="mode-toggle" style={{ margin: '1rem 0' }}>
            <label>
              <input
                type="checkbox"
                checked={mode === 'neural'}
                onChange={(e) => setMode(e.target.checked ? 'neural' : 'classical')}
              />
              🧠 Нейронный режим
            </label>
          </div>
          <div className="mode-toggle">
            <label>
              <input
                type="checkbox"
                checked={autoSelect}
                onChange={(e) => setAutoSelect(e.target.checked)}
              />
              🤖 Автовыбор метода
            </label>
          </div>
          <select 
            value={goal} 
            onChange={(e) => setGoal(e.target.value as GoalType)}
          >
            <option value="balanced">⚖️ Баланс</option>
            <option value="speed">⚡ Скорость</option>
            <option value="accuracy">🎯 Точность</option>
            <option value="low_memory">💾 Память</option>
          </select>
          <button type="submit" disabled={!file || loading}>
            {loading ? 'Обработка...' : 'Запуск'}
          </button>
        </form>

        {err && <div className="err">{err}</div>}

        {mode === 'neural' ? (
          // 🧠 Нейронный режим
          <div className="neural-selects" style={{ display: 'flex', gap: '1rem', margin: '1rem 0', flexWrap: 'wrap' }}>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.25rem' }}>
              <label style={{ fontSize: '0.85rem', fontWeight: 500 }}>🎯 Задача:</label>
              <select 
                value={neuralTask} 
                onChange={(e) => setNeuralTask(e.target.value as 'semantic' | 'instance' | 'panoptic')}
                style={{ padding: '0.5rem', borderRadius: '4px', border: '1px solid #ccc' }}
              >
                {NEURAL_TASKS.map(t => (
                  <option key={t.value} value={t.value}>{t.label}</option>
                ))}
              </select>
            </div>
            
            {!autoSelect && (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '0.25rem' }}>
                <label style={{ fontSize: '0.85rem', fontWeight: 500 }}>🤖 Модель:</label>
                <select 
                  value={neuralModel} 
                  onChange={(e) => setNeuralModel(e.target.value)}
                  style={{ padding: '0.5rem', borderRadius: '4px', border: '1px solid #ccc' }}
                >
                  {NEURAL_MODELS[neuralTask].map(m => (
                    <option key={m} value={m}>{m}</option>
                  ))}
                </select>
              </div>
            )}
          </div>
        ) : (
          !autoSelect && (
          <div className="method-select-group" style={{ display: 'flex', gap: '1rem', margin: '1rem 0', flexWrap: 'wrap' }}>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.25rem' }}>
              <label>📚 Библиотека:</label>
              <select 
                value={selectedLibrary}
                onChange={(e) => {
                  setSelectedLibrary(e.target.value);
                  setSelectedMethod("");
                }}
                className="library-select"
                style={{ padding: '0.5rem', borderRadius: '4px', border: '1px solid #ccc' }}
              >
                {LIBRARIES.map(lib => (
                  <option key={lib.value} value={lib.value}>
                    {lib.icon} {lib.label}
                  </option>
                ))}
              </select>
            </div>
            
            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.25rem' }}>
              <label>⚙️ Метод:</label>
              <select 
                value={selectedMethod}
                onChange={handleMethodChange}
                className="method-select"
                disabled={Object.keys(availableMethods).length === 0}
                style={{ padding: '0.5rem', borderRadius: '4px', border: '1px solid #ccc' }}
              >
                {Object.entries(availableMethods).map(([key, method]: [string, any]) => (
                  <option key={key} value={key}>
                    {method.name} {method.avg_iou > 0.8 && '⭐'}
                  </option>
                ))}
              </select>
            </div>
            
            {/* Подсказка по выбранному методу */}
            {availableMethods[selectedMethod] && (
              <div className="method-hint" style={{ fontSize: '0.85rem', color: '#666', marginTop: '0.5rem' }}>
                <small>
                  ⏱️ {availableMethods[selectedMethod].avg_time_ms}мс | 
                  🎯 IoU: {(availableMethods[selectedMethod].avg_iou * 100).toFixed(1)}% | 
                  💾 {availableMethods[selectedMethod].memory_mb}МБ
                </small>
                <p>{availableMethods[selectedMethod].description}</p>
              </div>
            )}
          </div>
        )
      )}

        {!autoSelect && methodSchema && Object.keys(methodSchema).length > 0 && (
          <div className="params-editor" style={{marginTop: '1rem', padding: '1rem', background: '#f0f4f8', borderRadius: '8px'}}>
            <h4>⚙️ Настройка параметров</h4>
            <div style={{display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(200px, 1fr))', gap: '1rem'}}>
              {Object.keys(methodSchema).map((paramKey) => {
                const config = methodSchema[paramKey];
                const isInt = config.type === 'int';
                
                return (
                  <div key={paramKey} style={{display: 'flex', flexDirection: 'column'}}>
                    <label style={{fontSize: '0.8rem', fontWeight: 'bold'}}>
                      {config.label || paramKey}
                    </label>
                    
                    {/* Слайдер для удобства или обычный инпут */}
                    {config.min !== undefined ? (
                      <div style={{display: 'flex', alignItems: 'center', gap: '0.5rem'}}>
                        <input
                          type="range"
                          min={config.min}
                          max={config.max}
                          step={config.step || 1}
                          value={customParams[paramKey] || config.default}
                          onChange={(e) => setCustomParams({
                            ...customParams, 
                            [paramKey]: isInt ? parseInt(e.target.value) : parseFloat(e.target.value)
                          })}
                          style={{flex: 1}}
                        />
                        <span style={{fontSize: '0.8rem', minWidth: '40px'}}>
                          {customParams[paramKey]}
                        </span>
                      </div>
                    ) : (
                      <input
                        type={isInt ? "number" : "number"}
                        step={config.step || "any"}
                        value={customParams[paramKey] || ""}
                        onChange={(e) => setCustomParams({
                          ...customParams, 
                          [paramKey]: isInt ? parseInt(e.target.value) : parseFloat(e.target.value)
                        })}
                      />
                    )}
                  </div>
                );
              })}
            </div>
          </div>
        )}

        {preview && (
          <div className="grid">
            <div className="card">
              <h3>📥 Оригинал</h3>
              <img src={preview} alt="original"/>
            </div>

            {res && (
              <div className="tabs">
                {['results', 'metrics', 'recommendations', 'analysis'].map(tab => (
                  <button
                    key={tab}
                    className={activeTab === tab ? 'active' : ''}
                    onClick={() => setActiveTab(tab as any)}
                  >
                    {{
                      results: '🎨 Результат',
                      metrics: '📊 Метрики',
                      recommendations: '💡 Рекомендации',
                      analysis: '🔍 Анализ'
                    }[tab]}
                  </button>
                ))}
              </div>
            )}
            
            {res && (
              <>
                <div className="card">
                  <h3>🎨 Overlay</h3>
                  <img src={res.overlay_b64} alt="overlay" />
                </div>
                <div className="card">
                  <h3>📊 Маска</h3>
                  <img src={res.mask_b64} alt="mask" />
                </div>
              </>
            )}
          </div>
        )}

        {gtFile && (
          <div className="card">
            <h3>🎯 Ground Truth</h3>
            <img 
              src={URL.createObjectURL(gtFile)} 
              alt="ground truth" 
              className="preview-img"
            />
          </div>
        )}

        {res && (
          <div className="meta">
            <h3>📈 Анализ</h3>
            <p>
              Метод: <b>{res.method.toUpperCase()}</b> | 
              Уверенность: {(res.confidence * 100).toFixed(1)}%
            </p>
            <p>
              Тип: {res.chars.type} | 
              Размер: {res.chars.size}
            </p>
            <p>
              Контраст: {res.chars.contrast.toFixed(3)} | 
              Шум: {res.chars.noise.toFixed(3)}
            </p>
          </div>
        )}

        {/* === ТАБ: Результаты === */}
        {activeTab === 'results' && res && (
          <div className="grid results-grid">
            <div className="card">
              <h3>📥 Оригинал</h3>
              <img src={preview!} alt="original" className="preview-img" />
            </div>
            <div className="card">
              <h3>🎨 Overlay</h3>
              <img src={res.overlay_b64} alt="overlay" className="preview-img" />
            </div>
            <div className="card">
              <h3>📊 Маска</h3>
              <img src={res.mask_b64} alt="mask" className="preview-img" />
              {res.analysis?.edges_b64 && (
                <>
                  <h4 style={{marginTop: '1rem'}}>🔲 Границы</h4>
                  <img src={res.analysis.edges_b64} alt="edges" className="preview-img" />
                </>
              )}
            </div>
          </div>
        )}

        {/* === ТАБ: Метрики === */}
        {activeTab === 'metrics' && res?.metrics && (
          <div className="metrics-section">
            <div className="metrics-grid">
              {[
                {label: 'IoU / Jaccard', value: res.metrics.iou, format: pct},
                {label: 'Dice Coefficient', value: res.metrics.dice, format: pct},
                {label: 'Accuracy', value: res.metrics.accuracy, format: pct},
                {label: 'Precision', value: res.metrics.precision, format: pct},
                {label: 'Recall', value: res.metrics.recall, format: pct},
                {label: 'F1 Score', value: res.metrics.f1_score, format: pct},
                {label: 'Pixel Accuracy', value: res.metrics.pixel_accuracy, format: pct},
                {label: 'MAE', value: res.metrics.mae, format: fmt},
                ...(res.metrics.hausdorff_distance && res.metrics.hausdorff_distance < Infinity 
                  ? [{label: 'Hausdorff', value: res.metrics.hausdorff_distance, format: (v:number)=>v.toFixed(2)}] 
                  : [])
              ].map(m => (
                <div key={m.label} className="metric-card">
                  <div className="metric-label">{m.label}</div>
                  <div className="metric-value">{m.format(m.value)}</div>
                </div>
              ))}
            </div>
            
            {/* Confusion Matrix Stats */}
            <div className="card" style={{marginTop: '1rem'}}>
              <h4>📋 Матрица ошибок</h4>
              <div className="confusion-grid">
                <div className="confusion-cell tp">TP: {res.metrics.true_positive}</div>
                <div className="confusion-cell fp">FP: {res.metrics.false_positive}</div>
                <div className="confusion-cell fn">FN: {res.metrics.false_negative}</div>
                <div className="confusion-cell tn">TN: {res.metrics.true_negative}</div>
              </div>
            </div>
          </div>
        )}

        {/* === ТАБ: Рекомендации === */}
        {activeTab === 'recommendations' && (res?.recommendations?.length ?? 0) > 0 && (
          <div className="recommendations-section">
            <table className="rec-table">
              <thead>
                <tr>
                  <th>Метод</th>
                  <th>Score</th>
                  <th>Время (мс)</th>
                  <th>Est. IoU</th>
                  <th>Лучше всего для</th>
                </tr>
              </thead>
              <tbody>
                {res?.recommendations?.map((rec, i) => (
                  <tr key={i} className={rec.method === res?.method ? 'selected' : ''}>
                    <td><b>{rec.method.toUpperCase()}</b>{rec.method === res?.method && ' ✅'}</td>
                    <td>{pct(rec.score)}</td>
                    <td>{rec.estimated_time_ms.toFixed(1)}</td>
                    <td>{pct(rec.estimated_iou)}</td>
                    <td>{rec.best_for?.join(', ')}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}

        {/* === ТАБ: Анализ === */}
        {activeTab === 'analysis' && res?.analysis && (
          <div className="analysis-section">
            <div className="grid">
              {/* Гистограмма */}
              <div className="card">
                <h4>📈 Распределение интенсивностей</h4>
                <ResponsiveContainer width="100%" height={250}>
                  <BarChart data={res.analysis.histogram.map((v, i) => ({bin: i, count: v}))}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="bin" label={{value: 'Интенсивность', position: 'insideBottom'}} />
                    <YAxis label={{value: 'Частота', angle: -90, position: 'insideLeft'}} />
                    <Tooltip />
                    <Bar dataKey="count" fill="#3b82f6" />
                  </BarChart>
                </ResponsiveContainer>
              </div>
              
              {/* Характеристики */}
              <div className="card">
                <h4>🔍 Характеристики изображения</h4>
                <div className="chars-list">
                  {Object.entries(res.chars).map(([k, v]) => (
                    <div key={k} className="char-item">
                      <span className="char-key">{k}:</span>
                      <span className="char-value">
                        {typeof v === 'number' ? (
                          k === 'size' ? v : 
                          k === 'contrast' || k === 'noise' || k === 'edge_density' || k === 'complexity' || k === 'mean_intensity' 
                            ? v.toFixed(3) 
                            : v
                        ) : v}
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            </div>
            
            {/* Примеры методов */}
            <div className="card" style={{marginTop: '1rem'}}>
              <h4>📚 Примеры использования методов</h4>
              <div className="examples-grid">
                {Object.entries(res.examples).map(([type, methods]) => (
                  <div key={type} className="example-card">
                    <h5>{type === 'medical' ? '🏥 Медицина' : 
                          type === 'documents' ? '📄 Документы' :
                          type === 'nature' ? '🌿 Природа' : '🏭 Индустрия'}</h5>
                    <div className="methods-list">
                      {methods.map(m => (
                        <span key={m} className={`method-tag ${m === res.method ? 'active' : ''}`}>
                          {m}
                        </span>
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
export default App

