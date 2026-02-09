"""
Скрипт для копирования фото из Pictures в all_photos
"""

import os
import shutil
import glob


def copy_photos_to_project():
    """Копирует фото из папки Pictures в проект"""

    source_dir = r"C:\Users\admin\Pictures"
    target_dir = r"C:\Users\admin\opencv_face_recognition\pythonProject1\all_photos"

    # Создаем папку all_photos если её нет
    os.makedirs(target_dir, exist_ok=True)

    print(f"🔍 Ищу фото в: {source_dir}")

    # Поддерживаемые форматы
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.gif']
    all_photos = []

    # Находим все фото
    for ext in extensions:
        pattern = os.path.join(source_dir, '**', ext)  # ** значит искать во всех подпапках
        found = glob.glob(pattern, recursive=True)
        all_photos.extend(found)

    print(f"📸 Найдено {len(all_photos)} фото")

    # Ограничиваем количество (первые 100)
    photos_to_copy = all_photos[:100]

    # Копируем
    copied_count = 0
    for i, photo_path in enumerate(photos_to_copy, 1):
        filename = os.path.basename(photo_path)
        target_path = os.path.join(target_dir, filename)

        # Если файл с таким именем уже существует, добавляем номер
        if os.path.exists(target_path):
            name, ext = os.path.splitext(filename)
            counter = 1
            while os.path.exists(os.path.join(target_dir, f"{name}_{counter}{ext}")):
                counter += 1
            filename = f"{name}_{counter}{ext}"
            target_path = os.path.join(target_dir, filename)

        shutil.copy2(photo_path, target_path)
        copied_count += 1

        # Показываем прогресс
        if i % 10 == 0 or i == len(photos_to_copy):
            print(f"  Копировано {i}/{len(photos_to_copy)}...")

    print(f"\n✅ Готово! Скопировано {copied_count} фото")
    print(f"   Папка all_photos: {len(os.listdir(target_dir))} файлов")

    # Показываем несколько примеров
    print("\n📋 Примеры скопированных файлов:")
    files = os.listdir(target_dir)[:5]
    for file in files:
        print(f"   - {file}")


if __name__ == "__main__":
    copy_photos_to_project()