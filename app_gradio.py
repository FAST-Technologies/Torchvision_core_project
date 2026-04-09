# app_gradio.py
import gradio as gr
import numpy as np
from PIL import Image
import sys, os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from segmenters.AutoSegmenter import AutoSegmenter, SegmentationGoal


class SegmentationApp:
    def __init__(self):
        self.auto_seg = AutoSegmenter()

    def segment_image(self, image, goal, top_k):
        """Обработка изображения"""
        if image is None:
            return None, "Загрузите изображение", None

        img_array = np.array(image)
        self.auto_seg.goal = SegmentationGoal(goal)
        mask, metadata = self.auto_seg.segment(
            img_array, auto_select=True, return_metadata=True
        )
        overlay = self.create_overlay(img_array, mask)

        # Формирование отчета
        report = f"""
        ### 📊 Результаты анализа
        
        **Выбранный метод:** {metadata['method'].upper()}  
        **Уверенность:** {metadata['confidence']:.2%}  
        **Библиотека:** {metadata['library']}
        
        ### 🔍 Характеристики изображения:
        - Тип: {metadata['image_characteristics'].estimated_type.value}
        - Размер: {metadata['image_characteristics'].width}x{metadata['image_characteristics'].height}
        - Контраст: {metadata['image_characteristics'].contrast:.3f}
        - Уровень шума: {metadata['image_characteristics'].noise_level:.3f}
        - Плотность границ: {metadata['image_characteristics'].edge_density:.3f}
        
        ### 📋 Параметры метода:
        {', '.join(f'{k}={v}' for k, v in metadata['parameters'].items())}
        """

        return overlay, report, mask

    def create_overlay(self, img, mask):
        """Создание наложения маски на изображение"""
        if len(img.shape) == 2:
            img = np.stack([img] * 3, axis=-1)

        # Цветная маска
        mask_colored = np.zeros_like(img)
        mask_colored[mask > 0] = [255, 0, 0]  # Красный

        # Наложение с прозрачностью
        overlay = (img * 0.6 + mask_colored * 0.4).astype(np.uint8)
        return overlay


# Создание интерфейса
app = SegmentationApp()

demo = gr.Interface(
    fn=app.segment_image,
    inputs=[
        gr.Image(type="pil", label="📷 Загрузите изображение"),
        gr.Radio(
            choices=["speed", "accuracy", "balanced", "low_memory"],
            value="balanced",
            label="🎯 Цель оптимизации",
        ),
        gr.Slider(1, 5, value=3, label="📊 Количество рекомендаций"),
    ],
    outputs=[
        gr.Image(type="pil", label=" Результат сегментации"),
        gr.Markdown(label="📈 Анализ"),
        gr.Image(type="numpy", label=" Маска", visible=False),
    ],
    title=" AutoSegmenter - Интеллектуальная сегментация изображений",
    description="""
    Загрузите изображение, и система автоматически подберет оптимальный метод сегментации!
    
    **Поддерживаемые методы:**
    - Пороговые: Otsu, Adaptive, Niblack, Sauvola
    - Граничные: Canny, Sobel, Prewitt, Laplacian
    
    **Особенности:**
    - Автоматический анализ характеристик изображения
    - Выбор метода на основе бенчмарков
    - Оптимизация под скорость или точность
    """,
    examples=[
        ["examples/doc1.jpg", "balanced", 3],
        ["examples/medical1.jpg", "accuracy", 3],
        ["examples/nature1.jpg", "speed", 3],
    ],
)

if __name__ == "__main__":
    demo.launch(share=True, server_name="0.0.0.0", server_port=7860)  # Публичная ссылка
