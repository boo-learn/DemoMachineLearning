import joblib
import numpy as np
from PIL import Image

# 1. Загружаем нашу обученную модель
try:
    model = joblib.load("digits_model.pkl")
except FileNotFoundError:
    print("Ошибка: Сначала обучите модель, запустив скрипт 02_train.py!")
    exit()

# 2. Путь к вашему файлу
# TODO: Какое предсказание получим, если передадим полностью белое изображение? Почему?
# TODO: А если полностью черное?
image_path = "my_data/digit1.png"  # Убедитесь, что файл лежит в папке проекта

try:
    # Открываем изображение и переводим его в черно-белый режим (L - Luminance)
    img = Image.open(image_path).convert("L")

    # Сжимаем картинку до размера 8x8 пикселей, который ожидает модель
    img_resized = img.resize((8, 8), Image.Resampling.LANCZOS)

    # Превращаем картинку в массив NumPy
    img_array = np.array(img_resized)

    # В Pillow: 0 — черный цвет, 255 — белый.
    # В датасете load_digits: 0 — белый фон, 16 — максимальная чернота цифры.
    # Делаем инверсию и масштабируем диапазон 0..255 в диапазон 0..16
    img_scaled = (255 - img_array) / 255.0 * 16.0

    # Округляем до целых чисел, как в оригинальном датасете
    img_ready = np.round(img_scaled).astype(int)

    # Выпрямляем матрицу 8x8 в плоский вектор из 64 чисел
    img_vector = img_ready.flatten()

    # Посмотрим в консоли, как компьютер «увидел» ваш рисунок
    print("--- Ваша матрица 8x8 в представлении компьютера ---")
    print(img_ready)
    print("--------------------------------------------------")

    # 3. Передаем векторизированную картинку модели для предсказания
    # Оборачиваем в [img_vector], так как модель ждет список (батч) картинок
    # TODO: замените метод .predict() на .predict_proba() - он возвращает вероятности
    prediction = model.predict([img_vector])[0]

    print(f"\n[РЕЗУЛЬТАТ] Модель считает, что вы нарисовали цифру: {prediction}")

except FileNotFoundError:
    print(f"Ошибка: Не удалось найти файл {image_path}. Проверьте путь!")