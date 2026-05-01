#!/usr/bin/env python3
"""
Сравнение обучения с разными уровнями аугментаций
"""
import sys
import os

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from segmenters.ModelTrainer import ModelTrainer, TrainingConfig


# ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("СРАВНЕНИЕ АУГМЕНТАЦИЙ ДЛЯ ОБУЧЕНИЯ СЕГМЕНТАЦИИ")
    print("=" * 70)

    # Инициализация трейнера
    trainer = ModelTrainer(
        checkpoint_dir="./models/augmentation_experiments",
        root_dir="./data/ade20k",
        device="cuda",
    )

    # === ЭКСПЕРИМЕНТ 1: U-Net с разными аугментациями ===
    print("\n" + "=" * 70)
    print("ЭКСПЕРИМЕНТ 1: U-Net ResNet34")
    print("=" * 70)

    comparison_df_unet = trainer.compare_augmentations(
        model_type="unet_smp",
        augmentation_levels=["none", "basic", "medium"],
        base_config={
            "epochs": 20,
            "batch_size": 4,
            "lr": 1e-4,
            "encoder_name": "resnet34",
            "subset_fraction": 0.05,
        },
    )

    # === ЭКСПЕРИМЕНТ 2: FPN MiT-B5 с разными аугментациями ===
    print("\n" + "=" * 70)
    print("ЭКСПЕРИМЕНТ 2: FPN MiT-B5")
    print("=" * 70)

    comparison_df_fpn = trainer.compare_augmentations(
        model_type="fpn_smp",
        augmentation_levels=["none", "basic", "medium"],
        base_config={
            "epochs": 20,
            "batch_size": 4,
            "lr": 5e-5,
            "encoder_name": "resnet34",
            "variant": "b5",
            "subset_fraction": 0.05,
        },
    )

    # === ВИЗУАЛИЗАЦИЯ ===
    print("\n" + "=" * 70)
    print("ВИЗУАЛИЗАЦИЯ РЕЗУЛЬТАТОВ")
    print("=" * 70)

    trainer.plot_experiment_comparison(
        output_path="./models/augmentation_experiments/comparison_visualization.png"
    )

    # === ОЦЕНКА ЛУЧШИХ ЧЕКПОИНТОВ ===
    print("\n" + "=" * 70)
    print("ОЦЕНКА ЛУЧШИХ ЧЕКПОИНТОВ")
    print("=" * 70)

    # Собираем лучшие чекпоинты
    best_checkpoints = []
    for result in trainer.experiment_results:
        if result["augmentation_level"] == "medium":  # Или выбираем по best_miou
            best_checkpoints.append(result["checkpoint_path"])

    if best_checkpoints:
        trainer.evaluate_checkpoints(
            checkpoint_paths=best_checkpoints,
            model_type="unet_smp",
            encoder_name="resnet34",
        )

    print("\n" + "=" * 70)
    print("ВСЕ ЭКСПЕРИМЕНТЫ ЗАВЕРШЕНЫ")
    print("=" * 70)


if __name__ == "__main__":
    main()
