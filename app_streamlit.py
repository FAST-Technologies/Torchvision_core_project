# app_streamlit.py

"""app_streamlit.py."""

import streamlit as st
import numpy as np
from PIL import Image
from segmenters.AutoSegmenter import AutoSegmenter, SegmentationGoal
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="AutoSegmenter", page_icon="🧠", layout="wide")

st.title("🧠 AutoSegmenter Pro")
st.markdown("---")

# Боковая панель
with st.sidebar:
    st.header("⚙️ Настройки")
    goal = st.selectbox(
        "Цель оптимизации", ["balanced", "speed", "accuracy", "low_memory"]
    )
    show_recommendations = st.checkbox("Показать рекомендации", value=True)
    top_k = st.slider("Количество рекомендаций", 1, 5, 3)

# Загрузка файла
uploaded_file = st.file_uploader(
    "📷 Загрузите изображение", type=["jpg", "jpeg", "png", "bmp"]
)

if uploaded_file is not None:
    # Колонки
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("📥 Оригинал")
        image = Image.open(uploaded_file)
        st.image(image, use_container_width=True)

    # Инициализация
    auto_seg = AutoSegmenter(goal=SegmentationGoal(goal))
    img_array = np.array(image)

    # Анализ
    with st.spinner("🔍 Анализирую изображение..."):
        characteristics = auto_seg.analyze_image(img_array)

        if show_recommendations:
            recommendations = auto_seg.get_recommendations(img_array, top_k=top_k)

    # Сегментация
    with st.spinner("🎨 Выполняю сегментацию..."):
        mask, metadata = auto_seg.segment(
            img_array, auto_select=True, return_metadata=True
        )

    # Результаты
    with col2:
        st.subheader("🎨 Результат")
        overlay = (
            img_array * 0.6 + np.stack([mask] * 3, axis=-1) * 0.4 * [255, 0, 0]
        ).astype(np.uint8)
        st.image(overlay, use_container_width=True)

    with col3:
        st.subheader("📊 Метрики")
        st.metric("Метод", metadata["method"].upper())
        st.metric("Уверенность", f"{metadata['confidence']:.2%}")
        st.metric("Время (оценка)", f"{metadata.get('estimated_time', 0):.1f}ms")

    # Детальная информация
    st.markdown("---")
    col4, col5 = st.columns(2)

    with col4:
        st.subheader("🔍 Характеристики изображения")
        st.json(
            {
                "Тип": characteristics.estimated_type.value,
                "Размер": f"{characteristics.width}x{characteristics.height}",
                "Каналы": characteristics.channels,
                "Средняя интенсивность": f"{characteristics.mean_intensity:.2f}",
                "Контраст": f"{characteristics.contrast:.3f}",
                "Уровень шума": f"{characteristics.noise_level:.3f}",
                "Плотность границ": f"{characteristics.edge_density:.3f}",
                "Комплексность": f"{characteristics.complexity_score:.3f}",
            }
        )

    with col5:
        if show_recommendations:
            st.subheader(" Топ рекомендаций")
            df_rec = pd.DataFrame(recommendations)
            st.dataframe(
                df_rec[["method", "score", "estimated_time_ms", "estimated_iou"]],
                use_container_width=True,
                hide_index=True,
            )

    # Графики
    st.markdown("---")
    st.subheader("📈 Визуализация анализа")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Гистограмма интенсивностей
    axes[0].hist(img_array.flatten(), bins=256, color="blue", alpha=0.7)
    axes[0].set_title("Распределение интенсивностей")

    # Границы
    edges = np.zeros_like(mask, dtype=float)
    edges[mask > 0] = 1
    axes[1].imshow(edges, cmap="gray")
    axes[1].set_title(f"Границы (плотность: {characteristics.edge_density:.3f})")

    # Маска
    axes[2].imshow(mask, cmap="gray")
    axes[2].set_title(f"Маска (покрытие: {np.mean(mask > 0) * 100:.1f}%)")

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    st.pyplot(fig)

else:
    st.info("👆 Загрузите изображение для начала работы")

# Примеры
st.markdown("---")
st.subheader("📚 Примеры использования")
st.markdown(
    """
- **Документы**: Otsu, Adaptive Thresholding
- **Медицинские изображения**: Sauvola, Adaptive
- **Природные сцены**: Canny, Sobel
- **Индустриальные**: Adaptive, Bernsen
"""
)
