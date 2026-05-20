import { Fragment, useState, ChangeEvent, FormEvent, useMemo, useCallback, useEffect, useRef } from 'react'
import { 
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  // LineChart, Line,
  Legend, Cell, ScatterChart, Scatter, ReferenceLine,
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
  original_b64?: string;
  primary_mask_b64?: string;
  reference_mask_b64?: string;
  difference_b64?: string;
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

export interface ValidationProgress {
  status: 'idle' | 'pending' | 'running' | 'completed' | 'failed';
  progress: number;        // 0-100
  processed: number;       // обработано методов
  total: number;           // всего методов
  elapsed_ms?: number;     // затраченное время
  error?: string;          // ошибка, если статус 'failed'
  error_details?: {
    error_type: string;
    failed_at: string;
    traceback?: string;
  };
}

export interface ValidationResponse {
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
  task_id?: string;
  progress?: ValidationProgress;
  benchmark?: BenchmarkSummary;
  benchmark_raw?: BenchmarkSummaryRaw;
}

export interface BenchmarkData {
  method: string;
  torch_time?: number;
  reference_time?: number;
  time_diff?: number;
  accuracy?: number | null;
  iou?: number | null;
  dice?: number | null;
  precision?: number | null;
  recall?: number | null;
  f1_score?: number | null;
  mae?: number | null;
  pixel_accuracy?: number | null;
  hausdorff_distance?: number | null;
  area_ratio?: number
  validation_status?: 'PASS' | 'WARNING' | 'FAIL';
  coverage_pct?: number
  predicted_area?: number;
  ground_truth_area?: number;
  area_difference?: number | null;
}

export interface BenchmarkSummary {
  methods_count: number;
  passed: number;
  warning: number;
  failed: number;
  avg_torch_time: number;
  avg_iou: number;
  data: BenchmarkData[];
}

export interface BenchmarkSummaryRaw {
  method: any;
  torch_time: number;
  reference_time: number;
  iou: number;
  status: string;
}

export interface BenchmarkProgress {
  status: 'idle' | 'pending' | 'running' | 'completed' | 'failed' | 'cancelled';
  progress: number;
  message: string;
  error_details?: {
    error_type: string;
    failed_at: string;
    traceback?: string;
  };
}

export interface BenchmarkConfig {
  // 🔹 Модели для запуска
  models_to_run?: string[]; // ['segformer', 'mask2former', ...]
  
  // 🔹 Метрики для расчёта
  metrics?: Array<'mIoU' | 'pixel_acc' | 'f1_weighted' | 'time_ms'>;
  
  // 🔹 Типы графиков
  plot_types?: Array<'bar' | 'scatter' | 'heatmap' | 'confusion'>;
  
  // 🔹 Параметры инференса
  inference?: {
    alpha?: number;          // Прозрачность наложения (0..1)
    batch_size?: number;     // Размер батча (если поддерживается)
    warmup_runs?: number;    // Прогревочные прогоны
  };
  
  // 🔹 Фильтры результатов
  filters?: {
    min_iou?: number;        // Минимальный IoU для отображения
    max_time_ms?: number;    // Максимальное время
    only_passed?: boolean;   // Только прошедшие валидацию
  };
  
  // 🔹 Визуализация
  visualization?: {
    show_overlay?: boolean;  // Показывать наложение маски
    show_gt?: boolean;       // Показывать ground truth
    color_palette?: 'ade' | 'coco' | 'cityscapes';
  };
}

export interface ComparatorMethod {
  name: string;
  library: "opencv" | "sklearn" | "torch" | "torch_v2";
  method: string;
  params?: Record<string, any>;
}

export interface ComparatorResult {
  method: string;
  library: string;
  f1_score?: number;
  jaccard?: number;
  accuracy?: number;
  test_time?: number;
  ref_time?: number;
  error?: string;
}

export interface ComparatorSummary {
  methods_count: number;
  successful: number;
  failed: number;
  top_by_f1: ComparatorResult[];
  avg_f1?: number;
}

export interface ComparatorResponse {
  success: boolean;
  elapsed_ms: number;
  summary: ComparatorSummary;
  results: ComparatorResult[];
  output_dir: string;
  charts: Record<string, string>; // base64
}

interface MethodInfo {
  name: string; library: string; avg_iou: number; avg_time_ms: number
  memory_mb: number; robustness: number; description: string
  best_for: string[]; defaults: Record<string, any>
  schema: Record<string, { type: string; min?: number; max?: number; step?: number; default: any; label?: string }>
}

type GoalType = 'balanced' | 'speed' | 'accuracy' | 'low_memory'
type Tab  = 'results' | 'metrics' | 'recommendations' | 'analysis' | 'validation' | 'benchmark' | 'comparator'
type Mode = 'classical' | 'neural'
type NeuralTask = 'semantic' | 'instance' | 'panoptic'

const NEURAL_TASKS = [
  { value: 'semantic', label: '🎨 Семантическая' },
  { value: 'instance', label: '🎭 Инстанс' },
  { value: 'panoptic', label: '🌐 Паноптическая' }
] as const;

export const DEFAULT_BENCHMARK_MODELS: string[] = [
  'segformer', 'segformer_b2', 'mask2former', 'maskformer', 'oneformer',
  'dpt', 'upernet', 
  // 'sam', 'sam2', 
  'yolov8n_seg', 'yolov8s_seg',
  'yolov8m_seg', 'unet_pretrained', 'deeplab_pretrained', 'fpn_mit_b5_pretrained',
  'psp_mit_b5_pretrained', 'fcn_resnet50_pretrained', 'segnet_resnet34_pretrained',
  'maskrcnn_pretrained',
];

export const DEFAULT_COMPARATOR_METHODS: Record<string, string[]> = {
  opencv: [
    "global_thresholding", "otsu_thresholding", "adaptive_thresholding",
    "canny_edge", "sobel_edge", "threshold_sauvola",
    "threshold_niblack", "threshold_bernsen", "prewitt_edge"
  ],
  sklearn: [
    "global_thresholding", "otsu_thresholding", "adaptive_thresholding",
    "canny_edge", "sobel_edge", "threshold_sauvola",
    "threshold_niblack", "threshold_bernsen", "prewitt_edge"
  ],
  torch: [
    "global_thresholding", "otsu_thresholding", "adaptive_thresholding",
    "canny_edge", "sobel_edge", "threshold_sauvola",
    "threshold_niblack", "threshold_bernsen", "prewitt_edge"
  ],
  torch_v2: [
    "global_thresholding", "otsu_thresholding", "adaptive_thresholding",
    "canny_edge", "sobel_edge", "threshold_sauvola",
    "threshold_niblack", "threshold_bernsen", "prewitt_edge"
  ]
};

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
    { value: "torch_v2", label: "PyTorch_v2", icon: "🟣" },
  ];
const API = 'http://localhost:8000'

// type AreaChartData = {
//   method: string;
//   coverage: number;
//   status?: 'PASS' | 'WARNING' | 'FAIL';
//   gt_area: number;
// };

const isValidAreaData = (d: BenchmarkData): d is BenchmarkData & { coverage_pct: number; ground_truth_area: number } => {
  return d.coverage_pct != null && d.ground_truth_area != null && d.ground_truth_area > 0;
};

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

function ValidationStatusBadge({ status }: { status: ValidationMethodResult['validation_status'] }) {
  const config = {
    PASS: { className: 'validation-status--pass', label: '✅ PASS' },
    WARNING: { className: 'validation-status--warning', label: '⚠️ WARNING' },
    FAIL: { className: 'validation-status--fail', label: '❌ FAIL' },
  }[status || 'FAIL'];
  
  return (
    <span className={`validation-status ${config.className}`}>
      {config.label}
    </span>
  );
}

function ValidationProgressBar({ progress }: { progress: ValidationProgress }) {
    if (progress.status === 'idle') return null;
    
    return (
      <div className="validation-progress">
        <div className="validation-progress__header">
          <span>
            {(progress.status === 'pending' || progress.status === 'running') && progress.total > 0 && 
              `🔄 Обработка: ${progress.processed}/${progress.total}`}
            {progress.status === 'pending' && progress.total === 0 && '⏳ Инициализация…'}
            {progress.status === 'completed' && '✅ Завершено'}
            {progress.status === 'failed' && '❌ Ошибка'}
          </span>
          {progress.elapsed_ms && <span>⏱ {progress.elapsed_ms}мс</span>}
        </div>
        
        {(progress.status === 'running' || progress.status === 'pending') && progress.total > 0 && (
          <>
            <div className="validation-progress__bar">
              <div 
                className="validation-progress__fill" 
                style={{ width: `${Math.min(progress.progress, 100)}%` }} 
              />
            </div>
            <div className="validation-progress__details">
              <span>Прогресс: {progress.progress}%</span>
              <span>Обработано: {progress.processed} из {progress.total}</span>
            </div>
          </>
        )}
      </div>
    );
  }

  function MaskComparisonGrid({ method, primaryLib, referenceLib }: { 
  method: ValidationMethodResult & { 
    original_b64?: string; 
    primary_mask_b64?: string; 
    reference_mask_b64?: string; 
    difference_b64?: string;
  };
  primaryLib: string;
  referenceLib: string;
}) {
  if (!method.original_b64) return null;
  
  // 🔹 Добавляем стейт для ошибок загрузки
  const [imgErrors, setImgErrors] = useState({
    original: false,
    primary: false,
    reference: false,
    difference: false,
  });

  // 🔹 Функция-хелпер для обработки ошибки
  const handleImgError = (key: keyof typeof imgErrors) => {
    console.error(`❌ Failed to load ${key} image`);
    setImgErrors(prev => ({ ...prev, [key]: true }));
  };

  return (
    <div className="mask-comparison">
      <div className="mask-comparison__grid">
        {/* Оригинальное изображение */}
        <div className="mask-comparison__item">
          <h5>Оригинал</h5>
          {imgErrors.original ? (
            <div className="img-error-placeholder">❌ Ошибка загрузки</div>
          ) : (
            <img 
              src={method.original_b64} 
              alt="original" 
              className="mask-img"
              onError={() => handleImgError('original')}
              onLoad={() => setImgErrors(prev => ({ ...prev, original: false }))}
            />
          )}
        </div>
        
        {/* Primary маска */}
        <div className="mask-comparison__item">
          <h5>{primaryLib}</h5>
          {imgErrors.primary ? (
            <div className="img-error-placeholder">❌ Ошибка загрузки</div>
          ) : (
            <img 
              src={method.primary_mask_b64} 
              alt="primary" 
              className="mask-img mask-img--grayscale"
              onError={() => handleImgError('primary')}
            />
          )}
        </div>
        
        {/* Reference маска */}
        <div className="mask-comparison__item">
          <h5>{referenceLib}</h5>
          {imgErrors.reference ? (
            <div className="img-error-placeholder">❌ Ошибка загрузки</div>
          ) : (
            <img 
              src={method.reference_mask_b64} 
              alt="reference" 
              className="mask-img mask-img--grayscale"
              onError={() => handleImgError('reference')}
            />
          )}
        </div>
        
        {/* Разность */}
        <div className="mask-comparison__item">
          <h5>Разность</h5>
          {imgErrors.difference ? (
            <div className="img-error-placeholder">❌ Ошибка загрузки</div>
          ) : (
            <img 
              src={method.difference_b64} 
              alt="difference" 
              className="mask-img mask-img--hot"
              onError={() => handleImgError('difference')}
            />
          )}
        </div>
      </div>
      
      {Object.values(imgErrors).every(v => v) && (
        <div className="text-red-500 text-sm mt-2">
          ⚠️ Не удалось загрузить изображения. Проверьте консоль для деталей.
        </div>
      )}
    </div>
  );
}

function ValidationResultsTable({ 
  results, 
  primaryLib, 
  referenceLib 
}: { 
  results: ValidationMethodResult[];
  primaryLib: string;
  referenceLib: string;
}) {
  const [expanded, setExpanded] = useState<string | null>(null);
  if (!results || results.length === 0) {
    return (
      <div className="text-center py-8 text-gray-500">
        📭 Нет данных для отображения
      </div>
    );
  }
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
            <Fragment key={r.method}>
              <tr className={!r.success ? 'opacity-50' : ''}>
                <td><b>{r.method}</b>{!r.success && <span className="text-red-500 ml-2">❌ {r.error}</span>}</td>
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
                <td>
                  {r.success && r.original_b64 && (
                    <button 
                      className="text-sm text-primary hover:underline"
                      onClick={() => setExpanded(expanded === r.method ? null : r.method)}
                    >
                      {expanded === r.method ? 'Скрыть 🔍' : 'Показать 🔍'}
                    </button>
                  )}
                </td>
              </tr>
              {expanded === r.method && r.success && (
                <tr>
                  <td colSpan={8}>
                    <MaskComparisonGrid 
                      method={r} 
                      primaryLib={primaryLib} 
                      referenceLib={referenceLib} 
                    />
                  </td>
                </tr>
              )}
            </Fragment>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function ValidationBenchmarkCharts({ data }: { data: BenchmarkSummary }) {
  console.log('📊 Benchmark data received:', data);
  // График 1: Время выполнения по методам
  const timeData = useMemo(() => 
    data.data
      .filter(d => d.torch_time != null)
      .sort((a, b) => (a.torch_time || 0) - (b.torch_time || 0))
      .map(d => ({ method: d.method, time: d.torch_time })),
    [data]
  );

  // График 2: IoU по методам
  const iouData = useMemo(() => 
    data.data
      .filter(d => d.iou != null)
      .sort((a, b) => (a.iou || 0) - (b.iou || 0))
      .map(d => ({ 
        method: d.method, 
        iou: d.iou,
        status: d.validation_status 
      })),
    [data]
  );

  // 🔹 График 3: Сравнение времени (Scatter)
  const compareData = useMemo(() => 
    data.data
      .filter(d => d.torch_time != null && d.reference_time != null && d.torch_time! > 0 && d.reference_time! > 0)
      .map(d => ({
        method: d.method,
        torch: d.torch_time,
        reference: d.reference_time,
        ratio: d.reference_time! > 0 ? d.torch_time! / d.reference_time! : 0,
      })),
    [data]
  );

  // 🔹 График 4: Покрытие масок  (пиксели)
  const coverageData = useMemo(() => 
    data.data
      .filter(d => d.predicted_area != null)
      .sort((a, b) => (a.predicted_area || 0) - (b.predicted_area || 0))
      .map(d => ({ method: d.method, coverage: d.predicted_area })),
    [data]
  );

  // 🔹 График 4: Покрытие масок  (проценты)
  const coverageData2 = useMemo(() => 
    data.data
      .filter(d => d.predicted_area != null && d.ground_truth_area != null && d.ground_truth_area > 0)
      .map(d => ({
        method: d.method,
        coverage: (d.predicted_area! / d.ground_truth_area! * 100),
        status: d.validation_status,
      })),
    [data]
  );

  const coverageData3 = useMemo(() => 
  data.data
    .filter(d => d.area_ratio != null && d.ground_truth_area != null && d.ground_truth_area > 0)
    .sort((a, b) => (a.area_ratio || 0) - (b.area_ratio || 0))
    .map(d => ({
      method: d.method,
      coverage: (d.area_ratio || 0) * 100,
      status: d.validation_status,
      pred_area: d.predicted_area,
      gt_area: d.ground_truth_area,
    })),
  [data]
);

  {/* 🔹 График: Покрытие относительно GT (%) */}
  const coveragePctData = useMemo(() => 
    data.data
      .filter(isValidAreaData)
      .map(d => ({
        method: d.method,
        coverage: d.coverage_pct,
        status: d.validation_status,
        gt_area: d.ground_truth_area,
      })),
    [data]
  );

  {/* 🔹 График: Сравнение площадей масок */}
  const areaComparisonData = useMemo(() => 
    data.data
      .filter(d => d.predicted_area != null && d.ground_truth_area != null)
      .map(d => ({
        method: d.method,
        predicted: d.predicted_area,
        ground_truth: d.ground_truth_area,
        ratio: d.ground_truth_area! > 0 ? (d.predicted_area! / d.ground_truth_area! * 100) : 0,
      }))
      .sort((a, b) => (b.ground_truth ?? 0) - (a.ground_truth ?? 0)),
    [data]
  );

  // 🔹 График 5: Матрица метрик (для heatmap)
  const heatmapData = useMemo(() => 
    data.data
      .filter(d => d.iou != null && d.f1_score != null)
      .slice(0, 20) // top-20 для читаемости
      .map(d => ({
        method: d.method,
        accuracy: d.accuracy,
        iou: d.iou,
        dice: d.dice,
        precision: d.precision,
        recall: d.recall,
        f1: d.f1_score,
        mae: d.mae,
        pixel_accuracy: d.pixel_accuracy,
        hausdorff_distance: d.hausdorff_distance,
        area_ratio: d.area_ratio,
        coverage_pct: d.coverage_pct,
        predicted_area: d.predicted_area,
        ground_truth_area: d.ground_truth_area,
        area_difference: d.area_difference,
      })),
    [data]
  );

  // 🔹 График 6: Trade-off время vs IoU
  const tradeoffData = useMemo(() => 
    data.data
      .filter(d => d.torch_time != null && d.iou != null && d.torch_time! > 0 && d.iou! > 0)
      .map(d => ({
        method: d.method,
        time: d.torch_time,
        iou: d.iou,
        status: d.validation_status,
      })),
    [data]
  );

  console.log('⏱ Time data:', timeData);
  console.log('🎯 IoU data:', iouData);

  return (
    <div className="benchmark-charts">
      {/* Статистика */}
      <div className="benchmark-summary">
        <div className="summary-card">
          <h4>📊 Сводка</h4>
          <p>Методов: <b>{data.methods_count}</b></p>
          <p>✅ PASS: <b>{data.passed}</b></p>
          <p>⚠️ WARNING: <b>{data.warning}</b></p>
          <p>❌ FAIL: <b>{data.failed}</b></p>
          <p>Средний IoU: <b>{(data.avg_iou * 100).toFixed(1)}%</b></p>
          <p>Среднее время: <b>{data.avg_torch_time.toFixed(3)}s</b></p>
        </div>
      </div>

      {/* График 1: Время выполнения */}
      <div className="chart-card">
        <h4>⏱ Время выполнения (Torch)</h4>
        <ResponsiveContainer width="100%" height={300}>
          <BarChart data={timeData} layout="vertical">
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis type="number" label={{ value: 'Время (с)', position: 'insideBottom' }} />
            <YAxis dataKey="method" type="category" width={150} tick={{ fontSize: 10 }} />
            <Tooltip 
              formatter={(v) => {
                const value = typeof v === 'number' ? v : 0;
                return [`${value.toFixed(3)}s`, 'Время'];
              }} 
            />
            <Bar dataKey="time" fill="#3b82f6" radius={[0, 2, 2, 0]} />
            <Legend />
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* График 2: IoU по методам */}
      <div className="chart-card">
        <h4>🎯 IoU по методам</h4>
        <ResponsiveContainer width="100%" height={300}>
          <BarChart data={iouData} layout="vertical">
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis type="number" domain={[0, 1]} label={{ value: 'IoU', position: 'insideBottom' }} />
            <YAxis dataKey="method" type="category" width={150} tick={{ fontSize: 10 }} />
            <Tooltip 
              formatter={(v, name) => {
                const value = typeof v === 'number' ? v : 0;
                return [`${(value * 100).toFixed(1)}%`, String(name)];
              }} 
            />
            <Bar dataKey="iou" radius={[0, 2, 2, 0]}>
              {iouData.map((entry, index) => {
                const color = 
                  entry.status === 'PASS' ? '#22c55e' :
                  entry.status === 'WARNING' ? '#f59e0b' :
                  '#ef4444';
                return <Cell key={`cell-${index}`} fill={color} />;
              })}
            </Bar>
            <Legend />
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* 🔹 График 3: Сравнение времени (Scatter) */}
      {compareData.length > 0 && (
        <div className="chart-card">
          <h4>⚖️ Сравнение: Torch vs Reference</h4>
          <ResponsiveContainer width="100%" height={300}>
            <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
              <CartesianGrid strokeDasharray="3 3" />
              
              {/* 🔹 Добавь domain для корректного масштабирования */}
              <XAxis 
                type="number" 
                dataKey="torch" 
                name="Torch" 
                label={{ value: 'Torch (с)', position: 'insideBottom', offset: -5 }}
                domain={[0, 'dataMax']}
              />
              <YAxis 
                type="number" 
                dataKey="reference" 
                name="Reference" 
                label={{ value: 'Reference (с)', angle: -90, position: 'insideLeft', offset: 0 }}
                domain={[0, 'dataMax']}
              />

              <Tooltip 
                formatter={(v, name) => {
                  const value = typeof v === 'number' ? v : 0;
                  return [`${value.toFixed(3)}s`, String(name ?? '')];
                }} 
                cursor={{ strokeDasharray: '3 3' }} 
              />
              
              {/* 🔹 Диагональная линия y=x через segment */}
              <ReferenceLine
                segment={[
                  { x: 0, y: 0 },
                  { x: 'dataMax' as any, y: 'dataMax' as any }
                ]}
                stroke="#888"
                strokeDasharray="3 3"
              />
              
              <Scatter name="Методы" data={compareData}>
                {compareData.map((entry, index) => {
                  const color = entry.ratio > 2 || entry.ratio < 0.5 ? '#ef4444' : '#3b82f6';
                  return <Cell key={`cell-${index}`} fill={color} />;
                })}
              </Scatter>
              <Legend />
            </ScatterChart>
          </ResponsiveContainer>
          <p className="text-xs text-gray-500 mt-2">
            🔴 Красные точки: методы, где разница во времени &gt;2×
          </p>
        </div>
      )}
      

      {/* 🔹 График 4: Покрытие масок */}
      {coverageData.length > 0 && (
        <div className="chart-card">
          <h4>📐 Покрытие масок (px)</h4>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={coverageData} layout="vertical">
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis type="number" domain={[0, 100]} label={{ value: 'Покрытие (px)', position: 'insideBottom' }} />
              <YAxis dataKey="method" type="category" width={150} tick={{ fontSize: 10 }} />
              <Tooltip 
                formatter={(v) => {
                  const value = typeof v === 'number' ? v : 0;
                  return [`${value.toFixed(1)}px`, 'Покрытие'];
                }} 
              />
              <Bar dataKey="coverage" fill="#14b8a6" radius={[0, 2, 2, 0]} />
              <Legend />
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}

      {coverageData2.length > 0 && (
        <div className="chart-card">
          <h4>📐 Покрытие масок (Ground Truth / Prediction, %)</h4>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={coverageData2} layout="vertical">
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis type="number" domain={[0, 200]} label={{ value: 'Покрытие (%)', position: 'insideBottom' }} />
              <YAxis dataKey="method" type="category" width={150} tick={{ fontSize: 10 }} />
              <Tooltip 
                formatter={(v) => {
                  const value = typeof v === 'number' ? v : 0;
                  return [`${value.toFixed(1)}%`, 'Покрытие'];
                }} 
              />
              <Bar dataKey="coverage" fill="#14b8a6" radius={[0, 2, 2, 0]} />
              <ReferenceLine x={100} stroke="#888" strokeDasharray="3 3" /> 
              <Legend />
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* 🔹 График: Отношение площадей маски (%) */}
      {coverageData3.length > 0 && (
        <div className="chart-card">
          <h4>📐 Отношение площадей: Prediction / Ground Truth</h4>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={coverageData3} layout="vertical">
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis 
                type="number" 
                domain={[0, 200]}  // ← 0-200%: видно пере- и недо-сегментацию
                label={{ value: 'Покрытие (%)', position: 'insideBottom' }} 
              />
              <YAxis dataKey="method" type="category" width={150} tick={{ fontSize: 10 }} />
              <Tooltip 
                formatter={(v, name) => {
                  const value = typeof v === 'number' ? v : 0;
                  if (name === 'coverage') return [`${value.toFixed(1)}%`, 'Отношение'];
                  if (name === 'pred_area') return [`${Math.round(value)} px`, 'Предсказание'];
                  if (name === 'gt_area') return [`${Math.round(value)} px`, 'Ground Truth'];
                  return [String(value), String(name ?? '')];
                }}
              />
              <Bar dataKey="coverage" radius={[0, 2, 2, 0]}>
                {coverageData3.map((entry, index) => {
                  // 🔹 Цвет по отклонению от 100%
                  const color = 
                    entry.coverage >= 95 && entry.coverage <= 105 ? '#22c55e' :    // ✅ Идеально
                    entry.coverage >= 80 && entry.coverage <= 120 ? '#f59e0b' :   // ⚠️ Нормально
                    '#ef4444';                                                     // ❌ Плохо
                  return <Cell key={`cell-${index}`} fill={color} />;
                })}
              </Bar>
              {/* 🔹 Линия идеального покрытия 100% */}
              <ReferenceLine 
                x={100} 
                stroke="#888" 
                strokeDasharray="3 3" 
                label={{ value: '100%', position: 'top', fill: '#666', fontSize: 10 }} 
              />
              <Legend />
            </BarChart>
          </ResponsiveContainer>
          <p className="text-xs text-gray-500 mt-2">
            🟢 95-105% | 🟠 80-120% | 🔴 &lt;80% или &gt;120%
          </p>
        </div>
      )}

      {coveragePctData.length > 0 && (
        <div className="chart-card">
          <h4>🎯 Покрытие маски (% от GT)</h4>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={coveragePctData} layout="vertical">
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis 
                type="number" 
                domain={[0, 200]}  // 0-200%, чтобы видеть пере- и недо-сегментацию
                label={{ value: 'Покрытие (%)', position: 'insideBottom' }} 
              />
              <YAxis dataKey="method" type="category" width={150} tick={{ fontSize: 10 }} />
              <Tooltip 
                formatter={(v, name) => {
                  const value = typeof v === 'number' ? v : 0;
                  if (name === 'coverage') return [`${value.toFixed(1)}%`, 'Покрытие'];
                  if (name === 'gt_area') return [`${Math.round(value)} px`, 'GT площадь'];
                  return [String(value), String(name ?? '')];
                }}
              />
              <Bar dataKey="coverage" radius={[0, 2, 2, 0]}>
                {coveragePctData.map((entry, index) => {
                  // Цвет по отклонению от 100%
                  if (entry.coverage == null) return null;
                  const color = 
                    entry.coverage >= 95 && entry.coverage <= 105 ? '#22c55e' :    // ✅ Хорошо
                    entry.coverage >= 80 && entry.coverage <= 120 ? '#f59e0b' :   // ⚠️ Нормально
                    '#ef4444';                                                     // ❌ Плохо
                  return <Cell key={`cell-${index}`} fill={color} />;
                })}
              </Bar>
              {/* 🔹 Линия идеального покрытия 100% */}
              <ReferenceLine x={100} stroke="#888" strokeDasharray="3 3" label={{ value: '100%', position: 'top', fill: '#666' }} />
              <Legend />
            </BarChart>
          </ResponsiveContainer>
          <p className="text-xs text-gray-500 mt-2">
            🟢 95-105% | 🟠 80-120% | 🔴 &lt;80% или &gt;120%
          </p>
        </div>
      )}

      {areaComparisonData.length > 0 && (
        <div className="chart-card">
          <h4>📐 Площади масок: Prediction vs Ground Truth</h4>
          <ResponsiveContainer width="100%" height={350}>
            <BarChart data={areaComparisonData} layout="vertical" margin={{ left: 20 }}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis type="number" label={{ value: 'Пиксели', position: 'insideBottom' }} />
              <YAxis dataKey="method" type="category" width={140} tick={{ fontSize: 9 }} />
              <Tooltip 
                formatter={(v, name) => {
                  const value = typeof v === 'number' ? v : 0;
                  if (name === 'predicted') return [`${Math.round(value)} px`, 'Предсказание'];
                  if (name === 'ground_truth') return [`${Math.round(value)} px`, 'Ground Truth'];
                  if (name === 'ratio') return [`${value.toFixed(1)}%`, 'Отношение'];
                  return [String(value), String(name ?? '')];
                }}
              />
              <Legend />
              <Bar dataKey="ground_truth" name="Ground Truth" fill="#94a3b8" radius={[2, 0, 0, 2]} />
              <Bar dataKey="predicted" name="Prediction" fill="#3b82f6" radius={[0, 2, 2, 0]} />
            </BarChart>
          </ResponsiveContainer>
          <p className="text-xs text-gray-500 mt-2">
            🔵 Предсказание | ⚪ Ground Truth | Чем ближе столбцы — тем лучше
          </p>
        </div>
      )}

      {/* 🔹 График 5: Матрица метрик (упрощённая таблица вместо heatmap) */}
      {heatmapData.length > 0 && (
        <div className="chart-card">
          <h4>🔥 Матрица метрик (top-{heatmapData.length})</h4>
          <div className="overflow-x-auto">
            <table className="data-table text-sm">
              <thead>
                <tr>
                  <th>Метод</th>
                  <th>Accuracy</th>
                  <th>IoU</th>
                  <th>Dice</th>
                  <th>Recall</th>
                  <th>F1</th>
                  <th>MAE</th>
                  <th>Pixel_Accuracy</th>
                  <th>Hausdorff_Distance</th>
                  <th>Area_Ratio</th>
                  <th>Predicted_Area</th>
                  <th>Ground_Truth_Area</th>
                  <th>Area_Difference</th>
                </tr>
              </thead>
              <tbody>
                {heatmapData.map((row, i) => (
                  <tr key={i}>
                    <td className="font-medium">{row.method}</td>
                    <td className={row.iou! >= 0.8 ? 'text-green-600' : 'text-red-600'}>{pct(row.iou)}</td>
                    <td>{pct(row.accuracy)}</td>
                    <td>{pct(row.dice)}</td>
                    <td>{pct(row.recall)}</td>
                    <td>{pct(row.f1)}</td>
                    <td>{pct(row.mae)}</td>
                    <td>{pct(row.pixel_accuracy)}</td>
                    <td>{row.hausdorff_distance}</td>
                    <td>{pct(row.area_ratio)}</td>
                    <td>{row.predicted_area}</td>
                    <td>{row.ground_truth_area}</td>
                    <td>{row.area_difference}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="text-xs text-gray-500 mt-2">
            🟢 IoU ≥ 80% | 🔴 IoU &lt; 80%
          </p>
        </div>
      )}

      {/* 🔹 График 6: Trade-off время vs IoU */}
      {tradeoffData.length > 0 && (
        <div className="chart-card">
          <h4>⚡ Trade-off: Скорость vs Точность</h4>
          <ResponsiveContainer width="100%" height={300}>
            <ScatterChart margin={{ top: 20, right: 20, bottom: 40, left: 20 }}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis type="number" dataKey="time" name="Время" label={{ value: 'Время (с)', position: 'insideBottom' }} />
              <YAxis type="number" dataKey="iou" name="IoU" label={{ value: 'IoU', angle: -90, position: 'insideLeft' }} />
              <Tooltip 
                formatter={(v, name) => {
                  const value = typeof v === 'number' ? v : 0;
                  if (name === 'iou') return [`${(value * 100).toFixed(1)}%`, 'IoU'];
                  return [`${value.toFixed(3)}s`, 'Время'];
                }} 
              />
              <Scatter name="Методы" data={tradeoffData}>
                {tradeoffData.map((entry, index) => {
                  // Цвет по статусу
                  const color = 
                    entry.status === 'PASS' ? '#22c55e' :
                    entry.status === 'WARNING' ? '#f59e0b' :
                    '#ef4444';
                  const isTop = index < 5; // упрощённо: первые 5
                  return (
                    <Cell 
                      key={`cell-${index}`} 
                      fill={color}
                      stroke={isTop ? '#000' : 'none'}
                      strokeWidth={isTop ? 1 : 0}
                    />
                  );
                })}
              </Scatter>
            </ScatterChart>
          </ResponsiveContainer>
          <p className="text-xs text-gray-500 mt-2">
            🟢 PASS | 🟠 WARNING | 🔴 FAIL | ⬛ Топ-5 компромиссов
          </p>
        </div>
      )}
    </div>
  );
}

function Badge({ text, variant = 'info' }: { text: string; variant?: 'info' | 'success' | 'warning' | 'default' }) {
  return <span className={`badge badge--${variant}`}>{text}</span>
}

function ImgCard({ title, src, grayscale = false }: { title: string; src: string; grayscale?: boolean }) {
  const [error, setError] = useState(false);
  return (
    <div className="img-card">
      <h4 className="img-card__title">{title}</h4>
      {error ? (
        <div className="img-card__error">
          ❌ Ошибка загрузки изображения
          <button 
            className="text-sm text-primary hover:underline mt-1"
            onClick={() => { setError(false); }}
          >
            Повторить
          </button>
        </div>
      ) : (
        <img 
          src={src} 
          alt={title} 
          className={grayscale ? 'img-card__image img-card__image--grayscale' : 'img-card__image'}
          onError={() => {
            console.error(`❌ Failed to load: ${title}`);
            setError(true);
          }}
        />
      )}
    </div>
  )
}

// ──────────────────────── Main App ─────────────────────────────────────────
export default function App() {
  type Timeout = ReturnType<typeof setTimeout>;
  const pollIntervalRef = useRef<Timeout | null>(null);
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
  const [_methodSchema, setMethodSchema] = useState<Record<string, any>>({});
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
  const [validationTaskId, setValidationTaskId] = useState<string | null>(null);
  const [validationProgress, setValidationProgress] = useState<ValidationProgress>({
    status: 'idle',
    progress: 0,
    processed: 0,
    total: 0,
  });
  const [benchmarkLoading, setBenchmarkLoading] = useState(false);
  const [benchmarkTaskId, setBenchmarkTaskId] = useState<string | null>(null);
  const [benchmarkProgress, setBenchmarkProgress] = useState<BenchmarkProgress>({
    status: 'idle',
    progress: 0,
    message: '',
    error_details: undefined,
  });
  const [benchmarkResult, setBenchmarkResult] = useState<any>(null);
  const [benchmarkGtFile, setBenchmarkGtFile] = useState<File | null>(null);
  const [selectedBenchmarkModels, setSelectedBenchmarkModels] = useState<string[]>(
    DEFAULT_BENCHMARK_MODELS
  );
  const [savedPresets, setSavedPresets] = useState<string[]>([]);

  // Comparator
  const [comparatorLoading, setComparatorLoading] = useState(false);
  const [_comparatorTaskId, setComparatorTaskId] = useState<string | null>(null);
  const [comparatorProgress, setComparatorProgress] = useState<BenchmarkProgress>({
    status: 'idle', progress: 0, message: '',
  });
  const [comparatorResult, setComparatorResult] = useState<ComparatorResponse | null>(null);
  const [selectedComparatorMethods, setSelectedComparatorMethods] = useState<ComparatorMethod[]>([
    { name: "Otsu_OpenCV", library: "opencv", method: "otsu_thresholding" },
    { name: "Otsu_Sklearn", library: "sklearn", method: "otsu_thresholding" },
    { name: "Otsu_Torch", library: "torch", method: "otsu_thresholding" },
    { name: "Otsu_Torch_v2", library: "torch_v2", method: "otsu_thresholding" },
  ]);
  const [comparatorReference, setComparatorReference] = useState<ComparatorMethod>({
    name: "Reference_Otsu", library: "sklearn", method: "otsu_thresholding"
  });
  useEffect(() => {
    setSavedPresets(getSavedPresets());
  }, []);

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

  useEffect(() => {
    console.log('🔄 validationProgress updated:', validationProgress);
    console.log('🔄 validationLoading:', validationLoading);
    console.log('🔄 validationResult:', !!validationResult);
  }, [validationProgress, validationLoading, validationResult]);

  useEffect(() => {
    if (!benchmarkTaskId) return;
    const poll = setInterval(async () => {
      try {
        const r = await fetch(`${API}/api/benchmark/status/${benchmarkTaskId}`);
        if (!r.ok) { console.error('Benchmark poll HTTP', r.status); return; }
        const data = await r.json();
        setBenchmarkProgress({
          status: data.status,
          progress: data.progress ?? 0,
          message: data.message ?? '',
          error_details: data.error_details,
        });
        if (['completed','failed','cancelled'].includes(data.status)) {
          clearInterval(poll);
          setBenchmarkLoading(false);
          setBenchmarkTaskId(null);
          if (data.status === 'completed') setBenchmarkResult(data.results);
          else setErr(data.message ?? 'Бенчмарк завершился с ошибкой');
          console.log(`✅ Benchmark ${data.status}:`, data);
        }
      } catch (e) { console.error('Benchmark poll error:', e); }
    }, 2000);
    return () => clearInterval(poll);
  }, [benchmarkTaskId]);

  useEffect(() => {
    if (benchmarkResult?.charts) {
      console.log('📊 benchmarkResult.charts:', benchmarkResult.charts);
    }
  }, [benchmarkResult]);

  const calculateEstimatedTime = (models: string[]): number => {
    const ESTIMATED_TIMES: Record<string, number> = {
      'segformer': 30000,
      'segformer_b2': 25000,
      'mask2former': 45000,
      'maskformer': 40000,
      'oneformer': 50000,
      'dpt': 35000,
      'upernet': 30000,
      'sam': 20000,
      'sam2': 25000,
      'yolov8n_seg': 10000,
      'yolov8s_seg': 15000,
      'yolov8m_seg': 20000,
      'unet_pretrained': 15000,
      'deeplab_pretrained': 20000,
      'fpn_mit_b5_pretrained': 18000,
      'psp_mit_b5_pretrained': 18000,
      'fcn_resnet50_pretrained': 12000,
      'segnet_resnet34_pretrained': 10000,
      'maskrcnn_pretrained': 25000,
    };
    
    return models.reduce((sum, model) => 
      sum + (ESTIMATED_TIMES[model] || 20000), 0);
  };

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

  const onBenchmarkStart = async () => {
    const health = await fetch(`${API}/api/benchmark/health`).then(r => r.json()).catch(() => null);
    if (health && health.cuda_available && health.vram_mb < 8000) {
      if (!window.confirm(
        `⚠️ Мало VRAM: ${health.vram_mb?.toFixed(0)} МБ (${health.device_name ?? 'GPU'}).\n` +
        `Свободно: ${health.vram_free_mb?.toFixed(0) ?? '?'} МБ. Некоторые модели могут не загрузиться.\n` +
        `Продолжить?`
      )) return;
    }
    setBenchmarkLoading(true);
    setBenchmarkTaskId(null);
    setBenchmarkProgress({
      status: 'pending',
      progress: 0,
      message: 'Запуск...',
      error_details: undefined,
    });
    setBenchmarkResult(null);
    setErr('');
    try {
      const fd = new FormData();
      if (file) fd.append('image', file);
      if (benchmarkGtFile) fd.append('gt_mask', benchmarkGtFile);
      fd.append('use_default_image', String(!file));
      if (!file) {
        fd.append('image_path', './data/ade20k_test_trained/original_image_0.jpg');
      }
      const config: BenchmarkConfig = {
        // Модели (все доступные по умолчанию)
        models_to_run: DEFAULT_BENCHMARK_MODELS,
        
        // Метрики
        metrics: ['mIoU', 'pixel_acc', 'f1_weighted', 'time_ms'],
        
        // Графики
        plot_types: ['bar', 'scatter', 'heatmap', 'confusion'],
        
        // Инференс
        inference: {
          alpha: 0.6,
          warmup_runs: 2,
        },
        
        // Фильтры
        filters: {
          min_iou: 0.0,
          only_passed: false,
        },
        
        // Визуализация
        visualization: {
          show_overlay: true,
          show_gt: true,
          color_palette: 'ade',
        },
      };
      fd.append('config', JSON.stringify(config));

      const res = await fetch(`${API}/api/benchmark/start`, {
        method: 'POST',
        body: fd,
      //   headers: { 'Content-Type': 'application/json' },
      //   body: JSON.stringify({ use_default_image: true })
      });
      if (!res.ok) throw new Error('Ошибка запуска');
      const { task_id } = await res.json();
      setBenchmarkTaskId(task_id);
    } catch (e: any) {
      setBenchmarkLoading(false);
      setErr(e.message);
    }
  };

  const onComparatorStart = async () => {
    if (!file) { setErr('Загрузите изображение'); return; }
    
    setComparatorLoading(true);
    setComparatorProgress({ status: 'pending', progress: 0, message: 'Запуск...' });
    setComparatorResult(null);
    
    try {
      const fd = new FormData();
      fd.append('image', file);
      fd.append('methods', JSON.stringify(selectedComparatorMethods));
      fd.append('reference', JSON.stringify(comparatorReference));
      fd.append('comparison_type', 'batch');
      
      const res = await fetch(`${API}/api/comparator/start`, { method: 'POST', body: fd });
      if (!res.ok) throw new Error('Ошибка запуска');
      const { task_id } = await res.json();
      setComparatorTaskId(task_id);
      
      // 🔹 Поллинг
      const poll = setInterval(async () => {
        try {
          const r = await fetch(`${API}/api/comparator/status/${task_id}`);
          const data = await r.json();
          setComparatorProgress({
            status: data.status, progress: data.progress ?? 0, message: data.message ?? ''
          });
          if (['completed', 'failed', 'cancelled'].includes(data.status)) {
            clearInterval(poll);
            setComparatorLoading(false);
            setComparatorTaskId(null);
            if (data.status === 'completed' && data.results) {
              setComparatorResult(data.results);
            } else if (data.status === 'failed') {
              setErr(data.message ?? 'Ошибка компаратора');
            }
          }
        } catch (e) { console.error('Comparator poll error:', e); }
      }, 2000);
      
    } catch (e: any) {
      setComparatorLoading(false);
      setErr(e.message);
    }
  };

  const onValidate = async () => {
    if (!file) { setErr('Сначала загрузите изображение'); return; }
    
    setValidationLoading(true);
    setErr('');
    setValidationResult(null);
    setValidationProgress({ status: 'pending', progress: 0, processed: 0, total: 0 });
    
    const fd = new FormData();
    fd.append('file', file);
    fd.append('primary_library', validationPrimaryLib);
    fd.append('reference_library', validationReferenceLib);
    fd.append('methods_filter', validationFilter);
    
    try {
      const startRes = await fetch(`${API}/api/validate/start`, { method: 'POST', body: fd });
      if (!startRes.ok) throw new Error(`Ошибка запуска: ${startRes.status}`);
      const { task_id } = await startRes.json();
      console.log('🔹 Validation started, task_id:', task_id);
      setValidationTaskId(task_id);
      const startTime = Date.now();
      console.log("Current time: ", startTime)
      
      await new Promise(resolve => setTimeout(resolve, 300));
      if (pollIntervalRef.current) {
        clearInterval(pollIntervalRef.current);
        pollIntervalRef.current = null;
      }
      pollIntervalRef.current = setInterval(async () => {
        try {
          const statusUrl = `${API}/api/validate/status/${task_id}`;
          console.log('🔄 Polling:', statusUrl);
          const statusRes = await fetch(statusUrl);
          if (!statusRes.ok) throw new Error(`Status error: ${statusRes.status}`);
          console.log('📊 statusRes:', statusRes);
          const status = await statusRes.json();
          console.log('📊 RAW server response:', status);
          
          setValidationProgress({
            status: status.status,
            progress: status.progress,
            processed: status.processed,
            total: status.total_methods,
            elapsed_ms: status.elapsed_ms,
          });
          console.log('📊 Status update:', status.status);
          console.log('📊 Progress update:', status.progress);
          console.log('📊 processed update:', status.processed);
          console.log('📊 total_methods update:', status.total_methods);
          console.log('📊 elapsed_ms update:', status.elapsed_ms);
          
          if (status.status === 'completed' || status.status === 'failed') {
            console.log('✅ Validation finished');
            if (pollIntervalRef.current) {
              clearInterval(pollIntervalRef.current);
              pollIntervalRef.current = null;
            }
            setValidationTaskId(null);
            if (status.status === 'completed') {
              console.log('🔹 status.benchmark:', status.benchmark);
              console.log('🔹 status.benchmark_raw:', status.benchmark_raw);
              console.log('🔹 status.results type:', typeof status.results);
              console.log('🔹 status.results isArray:', Array.isArray(status.results));
              console.log('🔹 status.results length:', Array.isArray(status.results) ? status.results.length : 'N/A');
              const resultsArray = Array.isArray(status.results) ? status.results : [];
              console.log('🔹 resultsArray length:', resultsArray.length);
              const summary = resultsArray.map((r: any) => ({
                method: r.method,
                success: r.success,
                validation_status: r.validation_status,
                iou: r.iou,
                dice: r.dice,
                pixel_accuracy: r.pixel_accuracy,
                precision: r.precision,
                recall: r.recall,
                f1_score: r.f1_score,
                mae: r.mae,
                hausdorff_distance: r.hausdorff_distance,
                primary_time: r.primary_time,
                reference_time: r.reference_time,
                time_diff: r.time_diff,
                original_b64: r.original_b64,
                primary_mask_b64: r.primary_mask_b64,
                reference_mask_b64: r.reference_mask_b64,
                difference_b64: r.difference_b64,
              })) || [];
              
              setValidationResult({
                success: true,
                elapsed_ms: status.elapsed_ms,
                primary_library: validationPrimaryLib,
                reference_library: validationReferenceLib,
                methods_tested: status.total_methods,
                passed: status.passed || 0,
                warning: status.warning || 0,
                failed: status.failed || 0,
                results: summary,
                report_dir: status.report_dir || './data/validation_web',
                benchmark: status.benchmark,
                benchmark_raw: status.benchmark_raw,
              });
            } else {
              setErr(status.error || 'Ошибка валидации');
            }
            setValidationLoading(false);
            return;
          }
        } catch (pollErr: any) {
          console.error('❌ Polling error:', pollErr); 
          if (pollErr.message?.includes('404') || pollErr.message?.includes('Not Found')) {
            console.log('🔹 Task not found on server, stopping polling');
            if (validationResult && validationResult.success) {
              console.log('✅ Results already received, ignoring 404');
              if (pollIntervalRef.current) {
                clearInterval(pollIntervalRef.current);
                pollIntervalRef.current = null;
              }
              setValidationTaskId(null);
              setValidationLoading(false);
              return;
            }
            if (pollIntervalRef.current) {
              clearInterval(pollIntervalRef.current);
              pollIntervalRef.current = null;
            }
            setValidationTaskId(null);
            setValidationLoading(false);
            
            if (validationProgress.processed === 0 && validationProgress.total === 0) {
              setValidationProgress(prev => ({ ...prev, status: 'failed', error: 'Task expired' }));
              setErr('Задача не найдена на сервере');
            }
            return;
          }
          setValidationLoading(false);
          setValidationProgress(prev => ({ ...prev, status: 'failed', error: 'Connection error' }));
          setErr(pollErr.message || 'Ошибка подключения');
        }
      }, 1000);
      
    } catch (e: any) {
      console.error('❌ Validation error:', e);
      setErr(e.message || 'Неизвестная ошибка');
      setValidationLoading(false);
      setValidationTaskId(null);
      setValidationProgress(prev => ({ ...prev, status: 'failed', error: e.message }));
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

  useEffect(() => {
    return () => {
      if (pollIntervalRef.current) {
        clearInterval(pollIntervalRef.current);
        pollIntervalRef.current = null;
      }
    };
  }, []);

  const saveBenchmarkPreset = useCallback((name: string, config: BenchmarkConfig) => {
    try {
      const presets = JSON.parse(localStorage.getItem('benchmark_presets') || '{}');
      presets[name] = config;
      localStorage.setItem('benchmark_presets', JSON.stringify(presets));
      setSavedPresets(Object.keys(presets));
      console.log(`✅ Preset "${name}" saved`);
      return true;
    } catch (e) {
      console.error('❌ Failed to save preset:', e);
      return false;
    }
  }, []);

  const loadBenchmarkPreset = useCallback((name: string): BenchmarkConfig | null => {
    try {
      const presets = JSON.parse(localStorage.getItem('benchmark_presets') || '{}');
      return presets[name] || null;
    } catch (e) {
      console.error('❌ Failed to load preset:', e);
      return null;
    }
  }, []);

  const getSavedPresets = useCallback((): string[] => {
    try {
      const presets = JSON.parse(localStorage.getItem('benchmark_presets') || '{}');
      return Object.keys(presets);
    } catch {
      return [];
    }
  }, []);

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
              ['benchmark', '📊 Бенчмарк'],
              ['comparator', '⚖️ Компаратор'],
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
                Результаты сохраняются в <code className="bg-gray-100 px-1 rounded">{validationResult?.report_dir || './data/validation_web'}</code>
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
                      disabled={validationLoading || validationProgress.status === 'running'}
                    >
                      <option value="torch">🔴 PyTorch</option>
                      <option value="torch_v2">🟣 PyTorch_v2</option>
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
                      disabled={validationLoading || validationProgress.status === 'running'}
                    >
                      <option value="opencv">🟢 OpenCV</option>
                      <option value="sklearn">🔵 Scikit-learn</option>
                      <option value="torch">🔴 PyTorch</option>
                      <option value="torch_v2">🟣 PyTorch_v2</option>
                    </select>
                  </div>
                  
                  <div className="control-group">
                    <label className="control-group__label">🔍 Фильтр методов</label>
                    <select 
                      value={validationFilter}
                      onChange={(e) => setValidationFilter(e.target.value)}
                      className="control-select"
                      disabled={validationLoading || validationProgress.status === 'running'}
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
                  disabled={!file || validationLoading || validationProgress.status === 'running'}
                  className="submit-btn submit-btn--secondary mt-4"
                >
                  {validationLoading || validationProgress.status === 'running' 
                    ? '⏳ Валидация…' 
                    : '▶ Запустить валидацию'}
                </button>
              </div>
              {validationResult?.benchmark ? (
                <div className="card mt-4">
                  <h4 className="card__title">📈 Бенчмарк-анализ</h4>
                  <ValidationBenchmarkCharts data={validationResult.benchmark} />
                </div>
              ) : (
                <div className="text-gray-500 text-sm mt-4">
                  ℹ️ Данные для бенчмарка ещё не загружены
                </div>
              )}
              {validationResult?.benchmark_raw && (
                <details>
                  <summary>🔍 Raw benchmark data</summary>
                  <pre>{JSON.stringify(validationResult.benchmark_raw, null, 2)}</pre>
                </details>
              )}
              <div className="card mt-4">
                <h4>📊 Benchmark debug</h4>
                <p>validationResult: {validationResult ? '✅' : '❌'}</p>
                <p>benchmark: {validationResult?.benchmark ? '✅' : '❌'}</p>
                <pre className="text-xs max-h-40 overflow-auto">
                  {JSON.stringify(validationResult?.benchmark, null, 2)}
                </pre>
              </div>
            </div>
            
            {/* 🔹 Прогресс-бар — показываем по статусу, а не по loading */}
            {(validationProgress.status === 'pending' || 
              validationProgress.status === 'running' ||
              (validationTaskId && validationProgress.status !== 'completed' && validationProgress.status !== 'failed')) && (
              <ValidationProgressBar progress={validationProgress} />
            )}
            
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
                  <MetricCard label="Совпадение (IoU)" value={pct(validationResult.results.filter(r => r.iou != null && r.iou >= 0.8).length / (validationResult.results.filter(r => r.iou != null).length || 1))} color="blue" />
                  <MetricCard label="Среднее время" value={`${(validationResult.results.reduce((a, b) => a + (b.primary_time || 0), 0) / (validationResult.results.length || 1)).toFixed(2)}s`} color="green" />
                  <MetricCard label="Успешных" value={`${validationResult.passed}/${validationResult.methods_tested}`} color="success" />
                </div>
                
                {/* Таблица результатов */}
                <ValidationResultsTable 
                  results={validationResult.results} 
                  primaryLib={validationPrimaryLib}
                  referenceLib={validationReferenceLib}
                />
                
                {/* Ссылка на полный отчёт */}
                <div className="mt-4 p-3 bg-gray-50 rounded text-sm">
                  💾 Полный отчёт сохранён в: <code className="bg-white px-1 rounded">{validationResult.report_dir}</code>
                </div>
              </div>
            )}
          </div>
        )}

        {/* ── Benchmark Tab ── */}
        {activeTab === 'benchmark' && (
          <div className="benchmark-tab">
            <div className="card">
              <h3 className="card__title">📊 Кросс-методный бенчмарк</h3>
              <p className="text-sm text-gray-600 mb-4">
                Сравните все доступные методы (классические + нейросетевые) на стандартном наборе. 
                Выполнение может занять 5-15 минут.
              </p>

              <div className="flex items-center gap-2 mb-4">
                <select 
                  className="control-select text-sm"
                  onChange={(e) => {
                    const preset = loadBenchmarkPreset(e.target.value);
                    if (preset) {
                      console.log('📥 Loaded preset:', preset);
                    }
                  }}
                  defaultValue=""
                >
                  <option value="" disabled>📁 Загрузить пресет...</option>
                  {savedPresets.map(name => (
                    <option key={name} value={name}>{name}</option>
                  ))}
                </select>
                
                <button
                  type="button"
                  onClick={() => {
                    const name = prompt('Название пресета:');
                    if (name) {
                      const config: BenchmarkConfig = {
                        models_to_run: ['segformer', 'mask2former', 'unet_pretrained'],
                        metrics: ['mIoU', 'pixel_acc', 'f1_weighted', 'time_ms'],
                        plot_types: ['bar', 'scatter', 'heatmap', 'confusion'],
                        inference: { alpha: 0.6, warmup_runs: 2 },
                        filters: { min_iou: 0.0, only_passed: false },
                        visualization: { show_overlay: true, show_gt: true, color_palette: 'ade' },
                      };
                      if (saveBenchmarkPreset(name, config)) {
                        alert(`✅ Пресет "${name}" сохранён`);
                      }
                    }
                  }}
                  className="text-sm text-primary hover:underline"
                >
                  💾 Сохранить пресет
                </button>
              </div>

              <button
                onClick={onBenchmarkStart}
                disabled={benchmarkLoading}
                className="submit-btn submit-btn--secondary"
              >
                {benchmarkLoading ? `⏳ ${benchmarkProgress.message} (${benchmarkProgress.progress}%)` : '▶ Запустить бенчмарк'}
              </button>

              <div className="control-group mt-4">
                <span className="control-group__label">🎯 GT-маска (опц.)</span>
                <input 
                  type="file" 
                  accept="image/*"
                  onChange={e => setBenchmarkGtFile(e.target.files?.[0] || null)} 
                  className="control-input" 
                />
                <p className="text-xs text-gray-500 mt-1">
                  Для расчёта метрик (mIoU, pixel_acc). Если не указано — используется дефолтная.
                </p>
              </div>

              {selectedBenchmarkModels.map(model => (
                <label key={model} className="flex items-center gap-1 text-sm">
                  <input
                    type="checkbox"
                    checked={selectedBenchmarkModels.includes(model)}
                    onChange={(e) => {
                      setSelectedBenchmarkModels(prev => 
                        e.target.checked 
                          ? [...prev, model] 
                          : prev.filter(m => m !== model)
                      );
                    }}
                  />
                  {model}
                </label>
              ))}

              {!benchmarkLoading && (
                <p className="text-xs text-gray-500 mt-2">
                  ⏱ Примерное время: {(calculateEstimatedTime(selectedBenchmarkModels) / 60000).toFixed(1)} мин
                </p>
              )}

              {/* Прогресс */}
              {benchmarkLoading && (
                <div className="validation-progress mt-4">
                  <div className="validation-progress__header">
                    <span>⚙️ {benchmarkProgress.message || 'Инициализация...'}</span>
                    <span>{benchmarkProgress.progress > 0 ? `${benchmarkProgress.progress.toFixed(0)}%` : ''}</span>
                  </div>
                  <div className={`validation-progress__bar ${benchmarkProgress.status === 'running' ? 'running' : ''}`}>
                    <div
                      className="validation-progress__fill"
                      style={{ width: `${Math.min(benchmarkProgress.progress, 100)}%` }}
                    />
                  </div>
                  <div className="validation-progress__details">
                    <span>Статус: {benchmarkProgress.status}</span>
                    {benchmarkProgress.progress > 0 && (
                      <span>Загружено: {benchmarkProgress.progress.toFixed(0)}%</span>
                    )}
                  </div>
                  {benchmarkProgress.status === 'running' && (
                    <div className="mt-3 p-4 bg-blue-50 rounded-lg border border-blue-200">
                      <h4 className="font-semibold text-blue-800 mb-2 text-sm">🔄 Бенчмарк в процессе</h4>
                      <ul className="text-sm text-blue-700 space-y-1">
                        <li>• Текущий этап: <b>{benchmarkProgress.message}</b></li>
                        <li>• Прогресс: <b>{benchmarkProgress.progress.toFixed(1)}%</b></li>
                        <li>• Осталось этапов: ~<b>{Math.max(0, Math.ceil((100 - benchmarkProgress.progress) / 5))}</b></li>
                      </ul>
                    </div>
                  )}
                </div>
              )}

              {benchmarkProgress.status === 'failed' && (
                <div className="error-banner">
                  ❌ {benchmarkProgress.message}
                  {benchmarkProgress.error_details && (
                    <details className="mt-2">
                      <summary className="cursor-pointer text-sm">Детали ошибки</summary>
                      <pre className="text-xs bg-red-50 p-2 rounded mt-1 overflow-auto">
                        {benchmarkProgress.error_details.traceback || JSON.stringify(benchmarkProgress.error_details, null, 2)}
                      </pre>
                    </details>
                  )}
                </div>
              )}

              {/* Результаты */}
              {benchmarkResult && (
                <div className="mt-6 space-y-6">
                  {/* Сводная таблица */}
                  <div className="overflow-x-auto">
                    <table className="data-table">
                      <thead>
                        <tr><th>Метод</th><th>mIoU</th><th>Pixel Acc</th><th>Time (ms)</th><th>Classes</th></tr>
                      </thead>
                      <tbody>
                        {Object.entries(benchmarkResult.summary).map(([method, data]: [string, any]) => (
                          <tr key={method}>
                            <td><b>{method}</b></td>
                            <td>{(data.mIoU * 100).toFixed(1)}%</td>
                            <td>{data.pixel_acc.toFixed(3)}</td>
                            <td>{data.time_ms.toFixed(1)}</td>
                            <td>{data.unique_classes}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>

                  {/* Графики */}
                  <div className="space-y-6">
                    <h4 className="font-semibold text-lg">📈 Графики метрик</h4>
                    {benchmarkResult?.charts?.metrics_plot_b64 && (
                      <div className="chart-card chart-card--tall">
                        <h5 className="font-medium mb-3">🎯 Mean IoU по моделям</h5>
                        <div className="chart-container">
                          <img
                            src={
                              String(benchmarkResult.charts.metrics_plot_b64).startsWith('data:')
                                ? benchmarkResult.charts.metrics_plot_b64
                                : `data:image/png;base64,${benchmarkResult.charts.metrics_plot_b64}`
                            }
                            alt="Benchmark metrics plot"
                            className="chart-img"
                            onError={e => { (e.target as HTMLImageElement).style.display='none'; }}
                          />
                        </div>
                      </div>
                    )}
                    {/* {benchmarkResult?.charts?.time_plot_b64 && (...)} */}
                  </div>
                </div>
              )}
            </div>
          </div>
        )}
        {benchmarkLoading && benchmarkTaskId && (
          <button
            onClick={async () => {
              try {
                const res = await fetch(`${API}/api/benchmark/${benchmarkTaskId}`, { method: 'DELETE' });
                const data = await res.json();
                // 200 = cancelled, 404 = not found, 400 = already finished
                if (res.ok || res.status === 400) {
                  setBenchmarkLoading(false);
                  setBenchmarkTaskId(null);
                  console.log('❌ Data Status:', data.status)
                  console.log('❌ Data Message:', data.message)
                  setBenchmarkProgress(prev => ({ ...prev, status: 'failed', message: 'Отменено пользователем' }));
                }
              } catch (e) { console.error('❌ Cancel error:', e); setBenchmarkLoading(false); setBenchmarkTaskId(null); }
            }}
            className="text-red-500 text-sm hover:underline ml-2"
            disabled={!benchmarkLoading || benchmarkProgress.status === 'cancelled'}
          >
            ✕ Отменить
          </button>
        )}

        {/* Вкладка компаратора */}
        {activeTab === 'comparator' && (
          <div className="comparator-tab">
            <div className="card">
              <h3 className="card__title">⚖️ Компаратор методов</h3>
              <p className="text-sm text-gray-600 mb-4">
                Сравните реализации одного метода в разных библиотеках (OpenCV / Sklearn / Torch / Torch_v2).
              </p>
              
              {/* Настройки */}
              <div className="comparator-controls">
                {/* Референсный метод */}
                <div className="control-group mb-4">
                  <label className="control-group__label">🎯 Референсный метод</label>
                  <select 
                    value={comparatorReference.name}
                    onChange={(e) => {
                      const ref = selectedComparatorMethods.find(m => m.name === e.target.value);
                      if (ref) setComparatorReference(ref);
                    }}
                    className="control-select"
                  >
                    {selectedComparatorMethods.map(m => (
                      <option key={m.name} value={m.name}>{m.name} ({m.library})</option>
                    ))}
                  </select>
                </div>
                
                {/* Выбор методов */}
                <div className="control-group mb-4">
                  <label className="control-group__label">📋 Методы для сравнения</label>
                  <div className="flex flex-wrap gap-2">
                    {DEFAULT_COMPARATOR_METHODS.opencv.map((method: string) => (
                      <label key={method} className="flex items-center gap-1 text-sm">
                        <input
                          type="checkbox"
                          checked={selectedComparatorMethods.some(m => m.method === method && m.library === 'opencv')}
                          onChange={(e) => {
                            if (e.target.checked) {
                              setSelectedComparatorMethods(prev => [...prev, {
                                name: `${method}_OpenCV`,
                                library: 'opencv' as const,  // ← 🔹 Явно указываем!
                                method: method,
                                params: {}
                              }]);
                            } else {
                              setSelectedComparatorMethods(prev => prev.filter(m => 
                                !(m.method === method && m.library === 'opencv')
                              ));
                            }
                          }}
                        />
                        🟢 {method}
                      </label>
                    ))}
                    {DEFAULT_COMPARATOR_METHODS.sklearn.map((method: string) => (
                      <label key={method} className="flex items-center gap-1 text-sm">
                        <input
                          type="checkbox"
                          checked={selectedComparatorMethods.some(m => m.method === method && m.library === 'sklearn')}
                          onChange={(e) => {
                            if (e.target.checked) {
                              setSelectedComparatorMethods(prev => [...prev, {
                                name: `${method}_Sklearn`, 
                                library: 'sklearn' as const, 
                                method: method,
                                params: {}
                              }]);
                            } else {
                              setSelectedComparatorMethods(prev => prev.filter(m => 
                                !(m.method === method && m.library === 'sklearn')
                              ));
                            }
                          }}
                        />
                        🔵 {method}
                      </label>
                    ))}
                    {DEFAULT_COMPARATOR_METHODS.torch.map((method: string) => (
                      <label key={method} className="flex items-center gap-1 text-sm">
                        <input
                          type="checkbox"
                          checked={selectedComparatorMethods.some(m => m.method === method && m.library === 'torch')}
                          onChange={(e) => {
                            if (e.target.checked) {
                              setSelectedComparatorMethods(prev => [...prev, {
                                name: `${method}_PyTorch`, 
                                library: 'torch' as const, 
                                method: method,
                                params: {}
                              }]);
                            } else {
                              setSelectedComparatorMethods(prev => prev.filter(m => 
                                !(m.method === method && m.library === 'torch')
                              ));
                            }
                          }}
                        />
                        🔴 {method}
                      </label>
                    ))}
                    {DEFAULT_COMPARATOR_METHODS.torch.map((method: string) => (
                      <label key={method} className="flex items-center gap-1 text-sm">
                        <input
                          type="checkbox"
                          checked={selectedComparatorMethods.some(m => m.method === method && m.library === 'torch_v2')}
                          onChange={(e) => {
                            if (e.target.checked) {
                              setSelectedComparatorMethods(prev => [...prev, {
                                name: `${method}_PyTorch_v2`, 
                                library: 'torch_v2' as const, 
                                method: method,
                                params: {}
                              }]);
                            } else {
                              setSelectedComparatorMethods(prev => prev.filter(m => 
                                !(m.method === method && m.library === 'torch_v2')
                              ));
                            }
                          }}
                        />
                        🟣 {method}
                      </label>
                    ))}
                  </div>
                </div>
                
                <button 
                  onClick={onComparatorStart}
                  disabled={comparatorLoading || selectedComparatorMethods.length === 0}
                  className="submit-btn submit-btn--secondary"
                >
                  {comparatorLoading ? `⏳ ${comparatorProgress.message}` : '▶ Запустить сравнение'}
                </button>
              </div>
              
              {/* Прогресс */}
              {comparatorLoading && (
                <div className="validation-progress mt-4">
                  <div className="validation-progress__header">
                    <span>⚙️ {comparatorProgress.message}</span>
                    <span>{comparatorProgress.progress > 0 ? `${comparatorProgress.progress.toFixed(0)}%` : ''}</span>
                  </div>
                  <div className="validation-progress__bar">
                    <div className="validation-progress__fill" style={{ width: `${Math.min(comparatorProgress.progress, 100)}%` }} />
                  </div>
                </div>
              )}
              
              {/* Результаты */}
              {comparatorResult && (
                <div className="mt-6 space-y-6">
                  {/* Сводка */}
                  <div className="metrics-grid">
                    <MetricCard label="Методов" value={comparatorResult.summary.methods_count.toString()} color="blue" />
                    <MetricCard label="Успешно" value={comparatorResult.summary.successful.toString()} color="success" />
                    <MetricCard label="Средний F1" value={comparatorResult.summary.avg_f1 ? pct(comparatorResult.summary.avg_f1) : '—'} color="purple" />
                  </div>
                  
                  {/* Топ по F1 */}
                  <div className="overflow-x-auto">
                    <table className="data-table">
                      <thead><tr><th>Метод</th><th>Библиотека</th><th>F1</th><th>IoU</th><th>Время (с)</th></tr></thead>
                      <tbody>
                        {comparatorResult.summary.top_by_f1.map((r, _) => (
                          <tr key={r.method}>
                            <td><b>{r.method}</b></td>
                            <td>{r.library}</td>
                            <td>{pct(r.f1_score)}</td>
                            <td>{pct(r.jaccard)}</td>
                            <td>{r.test_time?.toFixed(3)}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                  
                  {/* Графики */}
                  <div className="space-y-6">
                    <h4 className="font-semibold text-lg">📈 Сводная визуализация</h4>
                    {comparatorResult.charts?.['comparison_summary.jpg'] && (
                      <div className="chart-card chart-card--tall">
                        <h5 className="font-medium mb-3">🎯 Mean IoU по моделям</h5>
                        <div className="chart-container">
                          <img 
                            src={`data:image/jpeg;base64,${comparatorResult.charts['comparison_summary.jpg']}`}
                            alt="Summary comparator results"
                            className="chart-img"
                            onError={e => { (e.target as HTMLImageElement).style.display='none'; }}
                          />
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              )}
            </div>
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
                  <Tooltip 
                    formatter={(v, _) => {
                      const value = typeof v === 'number' ? v : 0;
                      return [String(Math.round(value)), 'Частота'];
                    }} 
                  />
                  <Bar dataKey="count" fill="#3b82f6" radius={[2,2,0,0]} />
                  <Legend />
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

