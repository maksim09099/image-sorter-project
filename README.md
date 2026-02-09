Скрипт для создания необходимых папок для проектов:
1. Распознавание лиц (main.py)
2. Автосортировщик фото (auto_sorter.py)
"""

import os
import sys


def create_project_structure():
    """Создает структуру папок для обоих проектов"""

    print("=" * 60)
    print("СОЗДАНИЕ СТРУКТУРЫ ПАПОК ДЛЯ ПРОЕКТОВ")
    print("=" * 60)

    # Папки для основного проекта (распознавание лиц)
    main_folders = [
        "data/faces",  # Фото с лицами
        "data/non_faces",  # Фото без лиц
        "models",  # Сохраненные модели ИИ
        "results",  # Результаты и графики
    ]

    # Папки для автосортировщика
    sorter_folders = [
        "all_photos",  # Исходные фото для сортировки
    ]

    all_folders = main_folders + sorter_folders

    print("\n📁 СОЗДАНИЕ ПАПОК:")
    created_count = 0
    existed_count = 0

    for folder in all_folders:
        try:
            os.makedirs(folder, exist_ok=True)
            if os.path.exists(folder):
                if len(os.listdir(folder)) == 0:
                    print(f"  ✓ Создана папка: {folder}/")
                    created_count += 1
                else:
                    print(f"  ✓ Папка уже существует: {folder}/ (не пустая)")
                    existed_count += 1
        except Exception as e:
            print(f"  ✗ Ошибка при создании {folder}: {e}")

    print(f"\n📊 РЕЗУЛЬТАТ:")
    print(f"  • Создано новых папок: {created_count}")
    print(f"  • Уже существовало: {existed_count}")

    # Проверяем существование файлов
    print("\n📄 ПРОВЕРКА ФАЙЛОВ:")
    required_files = ["main.py", "auto_sorter.py"]

    for file in required_files:
        if os.path.exists(file):
            print(f"  ✓ Файл существует: {file}")
        else:
            print(f"  ✗ Файл отсутствует: {file}")

    # Показываем содержимое каждой папки
    print("\n📂 СОДЕРЖИМОЕ ПАПОК:")
    for folder in all_folders:
        if os.path.exists(folder):
            try:
                files = os.listdir(folder)
                file_count = len(files)
                if file_count == 0:
                    print(f"  {folder}/: пусто")
                else:
                    # Показываем первые 3 файла
                    sample = files[:3]
                    if len(sample) == 3 and file_count > 3:
                        sample_text = ", ".join(sample) + f" ... (и еще {file_count - 3})"
                    else:
                        sample_text = ", ".join(files)
                    print(f"  {folder}/: {file_count} файлов [{sample_text}]")
            except:
                print(f"  {folder}/: ошибка чтения")

    # Создаем примеры файлов, если папки пустые
    print("\n🎯 СОВЕТЫ:")
    if os.path.exists("data/faces") and len(os.listdir("data/faces")) == 0:
        print("  • Папка data/faces/ пуста. Добавьте фото с лицами")
        print("  • Или запустите auto_sorter.py для автоматической сортировки")

    if os.path.exists("all_photos") and len(os.listdir("all_photos")) == 0:
        print("  • Папка all_photos/ пуста. Добавьте фото для сортировки")

    return True


def create_test_data():
    """Создает тестовые данные для проверки работы"""
    print("\n🤖 СОЗДАНИЕ ТЕСТОВЫХ ДАННЫХ")
    print("-" * 40)

    try:
        import cv2
        import numpy as np

        # Создаем несколько тестовых лиц
        for i in range(5):
            img = np.zeros((64, 64, 3), dtype=np.uint8)
            cv2.ellipse(img, (32, 32), (20, 25), 0, 0, 360, (255, 200, 150), -1)
            cv2.circle(img, (25, 25), 4, (0, 0, 0), -1)
            cv2.circle(img, (39, 25), 4, (0, 0, 0), -1)
            cv2.ellipse(img, (32, 40), (10, 5), 0, 0, 180, (0, 0, 0), 2)
            cv2.imwrite(f"data/faces/test_face_{i}.jpg", img)

        # Создаем несколько тестовых не-лиц
        for i in range(5):
            img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
            cv2.imwrite(f"data/non_faces/test_nonface_{i}.jpg", img)

        print("✅ Создано 10 тестовых изображений")
        print("   • 5 лиц в data/faces/")
        print("   • 5 не-лиц в data/non_faces/")

    except ImportError:
        print("⚠️  Для создания тестовых данных установите:")
        print("   pip install opencv-python numpy")


def main():
    """Главная функция"""
    print("\n" + "=" * 60)
    print("НАСТРОЙКА ПРОЕКТА РАСПОЗНАВАНИЯ ЛИЦ")
    print("=" * 60)

    # Создаем структуру
    create_project_structure()

    # Предлагаем создать тестовые данные
    response = input("\nСоздать тестовые данные? (да/нет): ").lower().strip()
    if response in ['да', 'yes', 'y', 'д']:
        create_test_data()

    # Показываем команды для запуска
    print("\n" + "=" * 60)
    print("🚀 КОМАНДЫ ДЛЯ ЗАПУСКА:")
    print("=" * 60)
    print("\n1. Запуск автосортировщика:")
    print("   python auto_sorter.py")
    print("   • Выберите пункт 1 в меню")

    print("\n2. Запуск основного ИИ:")
    print("   python main.py")
    print("   • Программа обучится на данных в data/faces и data/non_faces")

    print("\n3. Если возникают ошибки импорта:")
    print("   pip install opencv-python numpy matplotlib scikit-learn tensorflow tqdm")

    print("\n" + "=" * 60)
    print("✅ ВСЕ ПАПКИ СОЗДАНЫ!")
    print("=" * 60)


if __name__ == "__main__":
    main()



    pip install opencv-python
pip install numpy matplotlib scikit-learn tensorflow tqdm pillow
pip install pandas seaborn flask pyqt5
pip install opencv-python numpy matplotlib scikit-learn tensorflow pillow
pip install opencv-python numpy matplotlib scikit-learn tensorflow pillow
pip install pandas seaborn tqdm pathlib
pip install scikit-image imageio imutils
pip install albumentations opencv-contrib-python
pip install streamlit gradio plotly-dash
pip install pyqt5 customtkinter
pip install tqdm python-dotenv loguru
pip install joblib psutil
pip install cmake
pip install dlib
pip install opencv-python numpy matplotlib scikit-learn tensorflow pillow pandas seaborn tqdm scikit-image imutils
