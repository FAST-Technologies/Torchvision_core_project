import { useState, ChangeEvent, FormEvent } from 'react'
import './App.css'

interface ImageCharacteristics {
  type: string
  size: string
  contrast: number
  noise: number
}

interface SegmentationResponse {
  success: boolean
  method: string
  confidence: number
  mask_b64: string
  overlay_b64: string
  chars: ImageCharacteristics
}

type GoalType = 'balanced' | 'speed' | 'accuracy' | 'low_memory'

function App() {
  const [file, setFile] = useState<File | null>(null)
  const [preview, setPreview] = useState<string | null>(null)
  const [goal, setGoal] = useState<GoalType>('balanced')
  const [loading, setLoading] = useState<boolean>(false)
  const [res, setRes] = useState<SegmentationResponse | null>(null)
  const [err, setErr] = useState<string>('')

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
    fd.append('goal', goal)

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
          <input 
            type="file" 
            accept="image/*" 
            onChange={onChange} 
            required 
          />
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

        {preview && (
          <div className="grid">
            <div className="card">
              <h3>📥 Оригинал</h3>
              <img src={preview} alt="original" onLoad={cleanupPreview} />
            </div>
            
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
      </main>
    </div>
  )
}
export default App

