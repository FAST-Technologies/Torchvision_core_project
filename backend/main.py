# backend/app.py
import os, sys, base64, io
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import numpy as np
from PIL import Image

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from segmenters.AutoSegmenter import AutoSegmenter, SegmentationGoal

app = FastAPI(title="AutoSegmenter API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

auto_seg = AutoSegmenter()


def to_base64(arr: np.ndarray) -> str:
    img = Image.fromarray(arr.astype(np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


@app.post("/api/segment")
async def segment(file: UploadFile = File(...), goal: str = Form("balanced")):
    try:
        # Чтение изображения
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        img_array = np.array(image)

        auto_seg.goal = (
            SegmentationGoal(goal)
            if goal in ["speed", "accuracy", "balanced", "low_memory"]
            else SegmentationGoal.BALANCED
        )

        # Сегментация
        mask, metadata = auto_seg.segment(
            img_array, auto_select=True, return_metadata=True
        )

        # Сохранение результата
        mask_img = Image.fromarray(mask)
        buf = io.BytesIO()
        mask_img.save(buf, format="PNG")
        buf.seek(0)

        if len(img_array.shape) == 2:
            img_rgb = np.stack([img_array] * 3, axis=-1)
        else:
            img_rgb = img_array.copy()

        mask_colored = np.zeros_like(img_rgb)
        mask_colored[mask > 0] = [255, 0, 0]  # Красный для объекта
        overlay = (img_rgb * 0.6 + mask_colored * 0.4).astype(np.uint8)

        # 🔹 Конвертация в base64 (универсальная функция)
        def arr_to_b64(arr: np.ndarray) -> str:
            if arr.dtype != np.uint8:
                arr = (
                    (arr * 255).astype(np.uint8)
                    if arr.max() <= 1.0
                    else arr.astype(np.uint8)
                )
            img = Image.fromarray(arr)
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            return base64.b64encode(buf.getvalue()).decode("utf-8")

        return {
            "success": True,
            "method": metadata["method"],
            "confidence": float(metadata["confidence"]),
            "mask_b64": f"data:image/png;base64,{arr_to_b64(mask)}",
            "overlay_b64": f"data:image/png;base64,{arr_to_b64(overlay)}",
            "chars": {
                "type": metadata["image_characteristics"].estimated_type.value,
                "size": f"{metadata['image_characteristics'].width}×{metadata['image_characteristics'].height}",
                "contrast": float(metadata["image_characteristics"].contrast),
                "noise": float(metadata["image_characteristics"].noise_level),
            },
        }
    except Exception as e:
        import traceback

        print(f"❌ Ошибка в /api/segment: {e}")
        traceback.print_exc()
        raise HTTPException(500, str(e))


@app.get("/recommendations/")
async def get_recommendations(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))
    img_array = np.array(image)

    recs = auto_seg.get_recommendations(img_array, top_k=5)
    return {"recommendations": recs}


if os.path.exists("../frontend/dist"):
    app.mount("/", StaticFiles(directory="../frontend/dist", html=True), name="static")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
