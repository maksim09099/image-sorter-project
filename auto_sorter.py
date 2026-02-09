"""
АВТОМАТИЧЕСКИЙ СОРТИРОВЩИК ФОТО
Распределяет фото из all_photos на лица и не-лица
"""

import os
import shutil
import cv2
import numpy as np
from tqdm import tqdm  # для прогресс-бара


def setup_directories():
    """Создаёт необходимые папки"""
    directories = [
        "all_photos",  # исходные фото
        "data/faces",  # фото с лицами
        "data/non_faces",  # фото без лиц
        "models",  # модели ИИ
        "results"  # результаты
    ]

    for dir_path in directories:
        os.makedirs(dir_path, exist_ok=True)
        print(f"✓ Создана папка: {dir_path}")


def create_simple_faces(count=30):
    """Создаёт простые искусственные лица, если в all_photos нет фото"""
    print("\n🔄 Создаю искусственные лица для обучения...")

    for i in range(count):
        # Создаём изображение
        img = np.zeros((100, 100, 3), dtype=np.uint8)

        # Рисуем лицо
        cv2.ellipse(img, (50, 50), (30, 40), 0, 0, 360, (255, 200, 150), -1)  # лицо
        cv2.circle(img, (40, 40), 8, (0, 0, 0), -1)  # левый глаз
        cv2.circle(img, (60, 40), 8, (0, 0, 0), -1)  # правый глаз
        cv2.ellipse(img, (50, 65), (20, 10), 0, 0, 180, (0, 0, 0), 3)  # рот

        # Сохраняем
        cv2.imwrite(f"all_photos/artificial_face_{i:03d}.jpg", img)

    # Создаём не-лица
    for i in range(count):
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        cv2.imwrite(f"all_photos/artificial_nonface_{i:03d}.jpg", img)

    print(f"✅ Создано {count * 2} искусственных изображений")


def detect_faces_improved(image_path):
    """
    Улучшенный детектор лиц с несколькими методами
    """
    try:
        img = cv2.imread(image_path)
        if img is None:
            return False

        # Конвертируем в разные форматы для лучшего обнаружения
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Метод 1: Каскад Хаара (основной)
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )

        # Пробуем разные настройки чувствительности
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=3,  # уменьшили для большей чувствительности
            minSize=(30, 30)
        )

        if len(faces) > 0:
            return True

        # Метод 2: Детектор LBP (если Haar не сработал)
        lbp_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'lbpcascade_frontalface_improved.xml'
        )

        if lbp_cascade.empty():
            # Если LBP не загрузился, возвращаем результат Haar
            return len(faces) > 0

        lbp_faces = lbp_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=2,  # еще более чувствительный
            minSize=(20, 20)
        )

        return len(lbp_faces) > 0

    except Exception as e:
        print(f"⚠️ Ошибка при обработке {os.path.basename(image_path)}: {e}")
        return False


def sort_photos_automatically():
    """
    Основная функция сортировки
    """
    print("\n" + "=" * 60)
    print("🤖 НАЧИНАЮ АВТОМАТИЧЕСКУЮ СОРТИРОВКУ")
    print("=" * 60)

    # Проверяем, есть ли фото в all_photos
    photo_files = []
    for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
        photo_files.extend([f for f in os.listdir('all_photos') if f.lower().endswith(ext)])

    if not photo_files:
        print("📁 Папка all_photos пуста. Создаю тестовые данные...")
        create_simple_faces(15)
        # Обновляем список файлов
        photo_files = [f for f in os.listdir('all_photos') if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    print(f"📸 Найдено {len(photo_files)} фото для сортировки")

    # Статистика
    faces_found = 0
    non_faces_found = 0

    # Прогресс-бар
    print("\n🔍 Анализирую фото...")

    for filename in tqdm(photo_files, desc="Обработка"):
        file_path = os.path.join('all_photos', filename)

        # Определяем, есть ли лицо
        has_face = detect_faces_improved(file_path)

        # Копируем в соответствующую папку
        if has_face:
            shutil.copy2(file_path, os.path.join('data', 'faces', filename))
            faces_found += 1
        else:
            shutil.copy2(file_path, os.path.join('data', 'non_faces', filename))
            non_faces_found += 1

    # Результаты
    print("\n" + "=" * 60)
    print("📊 РЕЗУЛЬТАТЫ СОРТИРОВКИ:")
    print(f"   👤 Найдено лиц:    {faces_found} фото")
    print(f"   🚫 Не-лица:        {non_faces_found} фото")
    print(f"   📁 Всего обработано: {faces_found + non_faces_found} фото")
    print("=" * 60)

    # Если лиц слишком мало, добавляем искусственные
    if faces_found < 10:
        print("\n⚠️  Обнаружено мало лиц. Добавляю искусственные...")
        create_simple_faces(10)
        # Перезапускаем сортировку для новых файлов
        return sort_photos_automatically()

    return faces_found, non_faces_found


def show_sample_images():
    """Показывает примеры найденных лиц"""
    print("\n👀 ПРИМЕРЫ НАЙДЕННЫХ ЛИЦ:")

    faces_dir = "data/faces"
    if os.path.exists(faces_dir):
        face_files = [f for f in os.listdir(faces_dir) if f.lower().endswith(('.jpg', '.png'))][:3]

        for i, filename in enumerate(face_files, 1):
            print(f"   {i}. {filename}")

    print("\n💡 Совет: Если программа ошиблась, вы можете вручную")
    print("   переместить фото между папками data/faces и data/non_faces")


def main_menu():
    """Главное меню программы"""
    while True:
        print("\n" + "=" * 60)
        print("🏠 ГЛАВНОЕ МЕНЮ АВТОСОРТИРОВЩИКА")
        print("=" * 60)
        print("1. 📸 Автоматически отсортировать фото из all_photos/")
        print("2. 🖼️  Показать структуру папок")
        print("3. 🧹 Очистить папки data/faces и data/non_faces")
        print("4. 🤖 Запустить основной ИИ (обучение и тестирование)")
        print("5. 🚪 Выйти")

        choice = input("\nВыберите действие (1-5): ").strip()

        if choice == "1":
            # Создаем структуру папок
            setup_directories()

            # Запускаем сортировку
            faces, non_faces = sort_photos_automatically()

            # Показываем примеры
            show_sample_images()

            # Предлагаем запустить ИИ
            if faces > 0 and non_faces > 0:
                run_ai = input("\n✅ Данные готовы! Запустить обучение ИИ? (да/нет): ").strip().lower()
                if run_ai in ['да', 'yes', 'y', 'д']:
                    print("\n🚀 Запускаю основной ИИ...")
                    # Здесь можно вызвать основной код ИИ
                    # import main
                    # main.main()
                    print("Для запуска ИИ выполните: python main.py")

        elif choice == "2":
            print("\n📁 СТРУКТУРА ПАПОК:")
            print("   all_photos/    - исходные фото")
            print("   data/faces/    - фото с лицами")
            print("   data/non_faces - фото без лиц")

            # Показываем количество файлов
            if os.path.exists("all_photos"):
                count = len([f for f in os.listdir("all_photos") if f.lower().endswith(('.jpg', '.png'))])
                print(f"   all_photos: {count} фото")

            if os.path.exists("data/faces"):
                count = len([f for f in os.listdir("data/faces") if f.lower().endswith(('.jpg', '.png'))])
                print(f"   data/faces: {count} фото")

            if os.path.exists("data/non_faces"):
                count = len([f for f in os.listdir("data/non_faces") if f.lower().endswith(('.jpg', '.png'))])
                print(f"   data/non_faces: {count} фото")

        elif choice == "3":
            confirm = input("Удалить ВСЕ фото из data/faces и data/non_faces? (да/нет): ").strip().lower()
            if confirm in ['да', 'yes', 'y', 'д']:
                # Удаляем содержимое папок
                for folder in ["data/faces", "data/non_faces"]:
                    if os.path.exists(folder):
                        for file in os.listdir(folder):
                            try:
                                os.remove(os.path.join(folder, file))
                            except:
                                pass
                print("✅ Папки очищены!")

        elif choice == "4":
            print("\n🤖 Запуск основного ИИ...")
            print("Для запуска выполните в отдельном окне: python main.py")
            print("Или нажмите Ctrl+C чтобы выйти и запустить вручную")

        elif choice == "5":
            print("\n👋 До свидания!")
            break

        else:
            print("❌ Неверный выбор. Попробуйте снова.")


if __name__ == "__main__":
    # Создаем структуру папок при запуске
    setup_directories()

    # Запускаем меню
    main_menu()