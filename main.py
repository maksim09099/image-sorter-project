import os
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras import layers

print("=" * 60)
print(" ПРОСТОЙ ИИ ДЛЯ РАСПОЗНАВАНИЯ ЛИЦ")
print("=" * 60)

# ================== НАСТРОЙКИ ==================
DATA_PATH = "data"
IMG_SIZE = (64, 64)          # (width, height)
BATCH_SIZE = 32
EPOCHS = 20
MODELS_DIR = "models"
RESULTS_DIR = "results"
# ===============================================


def create_test_data():
    """Создаёт тестовые изображения если данных нет."""
    print(" Создаю тестовые данные...")

    faces_dir = os.path.join(DATA_PATH, "faces")
    non_faces_dir = os.path.join(DATA_PATH, "non_faces")
    os.makedirs(faces_dir, exist_ok=True)
    os.makedirs(non_faces_dir, exist_ok=True)

    # Лица
    for i in range(100):
        img = np.zeros((IMG_SIZE[1], IMG_SIZE[0], 3), dtype=np.uint8)

        cv2.ellipse(
            img,
            (IMG_SIZE[0] // 2, IMG_SIZE[1] // 2),
            (IMG_SIZE[0] // 4, IMG_SIZE[1] // 3),
            0, 0, 360,
            (255, 200, 150),
            -1
        )

        cv2.circle(img, (IMG_SIZE[0] // 2 - 15, IMG_SIZE[1] // 2 - 10), 5, (0, 0, 0), -1)
        cv2.circle(img, (IMG_SIZE[0] // 2 + 15, IMG_SIZE[1] // 2 - 10), 5, (0, 0, 0), -1)
        cv2.ellipse(img, (IMG_SIZE[0] // 2, IMG_SIZE[1] // 2 + 15), (10, 5), 0, 0, 180, (0, 0, 0), 2)

        out_path = os.path.join(faces_dir, f"face_{i:03d}.jpg")
        cv2.imwrite(out_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    # Не-лица
    for i in range(100):
        img = np.zeros((IMG_SIZE[1], IMG_SIZE[0], 3), dtype=np.uint8)
        shape = np.random.choice(["square", "triangle", "circle"])
        color = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))

        if shape == "square":
            cv2.rectangle(img, (10, 10), (IMG_SIZE[0] - 10, IMG_SIZE[1] - 10), color, -1)
        elif shape == "triangle":
            pts = np.array([
                [IMG_SIZE[0] // 2, 10],
                [10, IMG_SIZE[1] - 10],
                [IMG_SIZE[0] - 10, IMG_SIZE[1] - 10]
            ])
            cv2.fillPoly(img, [pts], color)
        else:
            cv2.circle(img, (IMG_SIZE[0] // 2, IMG_SIZE[1] // 2), IMG_SIZE[0] // 3, color, -1)

        out_path = os.path.join(non_faces_dir, f"nonface_{i:03d}.jpg")
        cv2.imwrite(out_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    print(" Тестовые данные созданы!")


def load_and_prepare_data():
    """Загружает данные из data/faces и data/non_faces, нормализует, делит на train/test."""
    print(" Загрузка данных...")

    faces_dir = os.path.join(DATA_PATH, "faces")
    non_faces_dir = os.path.join(DATA_PATH, "non_faces")

    if not os.path.exists(faces_dir) or not os.path.exists(non_faces_dir):
        print(" Папки с данными не найдены. Создаю тестовые данные...")
        create_test_data()
    images = []
    labels = []

    face_files = list(Path(faces_dir).glob("*.jpg")) + list(Path(faces_dir).glob("*.png"))
    non_face_files = list(Path(non_faces_dir).glob("*.jpg")) + list(Path(non_faces_dir).glob("*.png"))

    print(f"👤 Лиц: {len(face_files)} |  Не-лиц: {len(non_face_files)}")

    # Лица -> 1
    for img_path in face_files:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        img = cv2.resize(img, IMG_SIZE)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        images.append(img)
        labels.append(1)

# Не-лица -> 0
    for img_path in non_face_files:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        img = cv2.resize(img, IMG_SIZE)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        images.append(img)
        labels.append(0)

    if len(images) == 0:
        print(" Данные не загружены!")
        return None, None, None, None

    X = np.array(images, dtype="float32") / 255.0
    y = np.array(labels, dtype="float32")

    if len(np.unique(y)) < 2:
        print(" Недостаточно данных для 2 классов.")
        return None, None, None, None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print(f" Train: {len(X_train)} |  Test: {len(X_test)}")
    return X_train, X_test, y_train, y_test


def create_model(input_shape):
    """Создаёт CNN модель."""
    print("\n Создание модели...")

    model = tf.keras.Sequential([
        layers.Input(shape=input_shape),

        layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2)),

        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(64, activation="relu"),
        layers.Dense(1, activation="sigmoid")
    ])

    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    model.summary()
    return model


def train_model(model, X_train, y_train, X_test, y_test):
    """Обучает модель и сохраняет лучшую."""
    print("\n Начало обучения...")

    os.makedirs(MODELS_DIR, exist_ok=True)

    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(MODELS_DIR, "best_face_model.keras"),
        monitor="val_accuracy",
        save_best_only=True,
        mode="max",
        verbose=1
    )

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True
    )

    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[checkpoint, early_stop],
        verbose=1
    )

    return history, model


def evaluate_model(model, X_test, y_test):
    """Оценка качества на тесте."""
    print("\n Оценка модели...")
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f" Точность: {test_acc * 100:.2f}% | loss: {test_loss:.4f}")
    return test_acc


def plot_training_history(history):
    """Сохраняет графики accuracy/loss в results/."""
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Accuracy
    plt.figure(figsize=(8, 4))
    plt.plot(history.history["accuracy"], label="accuracy")
    plt.plot(history.history["val_accuracy"], label="val_accuracy")
    plt.title("Точность")
    plt.xlabel("Эпоха")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.legend()
    acc_path = os.path.join(RESULTS_DIR, "accuracy.png")
    plt.tight_layout()
    plt.savefig(acc_path, dpi=120)
    plt.show()
    print(f" График сохранен: {acc_path}")

    # Loss
    plt.figure(figsize=(8, 4))
    plt.plot(history.history["loss"], label="loss")
    plt.plot(history.history["val_loss"], label="val_loss")
    plt.title("Потери")
    plt.xlabel("Эпоха")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend()
    loss_path = os.path.join(RESULTS_DIR, "loss.png")
    plt.tight_layout()
    plt.savefig(loss_path, dpi=120)
    plt.show()
    print(f" График сохранен: {loss_path}")


def test_on_single_image(model, image_path):
    """Тестирует модель на одном изображении и показывает окно результата."""
    if not os.path.exists(image_path):
        print(f" Файл не найден: {image_path}")
        return None

    img = cv2.imread(image_path)
    if img is None:
        print(" Не удалось загрузить изображение")
        return None

    original = img.copy()

    # Подготовка изображения
    img_resized = cv2.resize(img, IMG_SIZE)
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_norm = img_rgb.astype("float32") / 255.0
    x = np.expand_dims(img_norm, axis=0)

    # Предсказание
    pred = float(model.predict(x, verbose=0)[0][0])

    # Текст результата
    if pred > 0.5:
        text = f"ЛИЦО ({pred * 100:.1f}%)"
        color = (0, 255, 0)
    else:
        text = f"НЕ ЛИЦО ({(1 - pred) * 100:.1f}%)"
        color = (0, 0, 255)

    # Показ результата
    cv2.putText(original, text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    h, w = original.shape[:2]
    scale = 700 / max(h, w)
    resized = cv2.resize(original, (int(w * scale), int(h * scale)))

    cv2.imshow("Результат", resized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return pred

def main():
    X_train, X_test, y_train, y_test = load_and_prepare_data()
    if X_train is None:
        return

    model = create_model((IMG_SIZE[1], IMG_SIZE[0], 3))

    history, trained_model = train_model(model, X_train, y_train, X_test, y_test)
    evaluate_model(trained_model, X_test, y_test)
    plot_training_history(history)

    os.makedirs(MODELS_DIR, exist_ok=True)
    final_path = os.path.join(MODELS_DIR, "face_recognition_model.keras")
    trained_model.save(final_path)
    print(f"\n Модель сохранена: {final_path}")

    while True:
        print("\nВыберите действие:")
        print("1) Проверить своё изображение")
        print("2) Проверить случайное тестовое изображение")
        print("3) Выйти")

        choice = input("Введите 1-3: ").strip()

        if choice == "1":
            img_path = input(r"Путь к файлу (пример: C:\Users\admin\Pictures\Camera Roll\photo.jpg): ").strip()
            test_on_single_image(trained_model, img_path)

        elif choice == "2":
            idx = np.random.randint(0, len(X_test))
            temp_img = (X_test[idx] * 255).astype("uint8")

            os.makedirs(RESULTS_DIR, exist_ok=True)
            temp_path = os.path.join(RESULTS_DIR, "temp_test.jpg")
            cv2.imwrite(temp_path, cv2.cvtColor(temp_img, cv2.COLOR_RGB2BGR))

            print(" Случайный тест...")
            test_on_single_image(trained_model, temp_path)

            if os.path.exists(temp_path):
                os.remove(temp_path)

        elif choice == "3":
            print(" До свидания!")
            break

        else:
            print(" Неверный выбор.")

if __name__ ==   "__main__":
    main()