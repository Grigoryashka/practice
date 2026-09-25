import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


np.random.seed(42)
N_USERS = 1200             # размер выборки
N_MODULES = 8              # количество модулей в курсе
AVG_WORDS_PER_SLIDE = 350  # средний объем текста на слайде (слов)

# Генерация синтетических xAPI-логов
# Реальные логи имеют формат JSON, но для анализа мы приводим их к таблице

# Базовые характеристики пользователей (уровень подготовки от 0 до 1)
user_skill = np.random.beta(2, 5, N_USERS)  # веса 2 и 5, 5 - вес низких значений (упор на большинство)

# Создаем DataFrame с построчными данными по каждому модулю
data = []

for user_id in range(N_USERS):
    skill = user_skill[user_id]

    for module_id in range(1, N_MODULES + 1):

        words_on_slide = np.random.randint(100, 600)  # Объем текста на слайде

        # Время на слайде зависит от объема текста и уровня подготовки
        # Чем больше текст и ниже навык — тем больше времени (или бросает)
        base_time = words_on_slide * 0.3  # базовое время чтения (сек)
        time_on_task = base_time / (0.5 + skill)  # опытные читают быстрее

        # Количество взаимодействий (кликов)
        interaction_rate = np.random.poisson(3 + skill * 5)

        # Вероятность завершения модуля зависит от объема текста и навыка
        # КЛЮЧЕВАЯ ЗАВИСИМОСТЬ: если слов > 300, вероятность резко падает
        threshold = 380  # порог слов, после которого начинается резкий отсев
        steepness = 45  # крутизна кривой (чем меньше, тем резче падение)

        # Формула сигмоиды: P = 1 / (1 + e^((x - threshold) / steepness))
        prob_text = 1 / (1 + np.exp((words_on_slide - threshold) / steepness))

        # Добавляем влияние скилла (опытные чуть лучше справляются с большими текстами)
        completion_prob = prob_text + (skill * 0.15)

        # Ограничиваем вероятность, чтобы не было абсолютных 0 и 1
        completion_prob = np.clip(completion_prob, 0.05, 0.95)

        # Факт завершения (1 — завершил, 0 — бросил) (рандом - человеческий фактор)
        completed = int(np.random.random() < completion_prob)

        # Балл за тест (если попытался пройти)
        if completed:
            test_score = np.clip(skill * 0.8 + 0.2 + np.random.normal(0, 0.1), 0, 1)
        else:
            test_score = np.nan

        data.append({
            'user_id': user_id,
            'module_id': module_id,
            'user_skill': round(skill, 3),
            'words_on_slide': words_on_slide,
            'time_on_task_sec': round(time_on_task, 1),
            'interaction_rate': interaction_rate,
            'completed': completed,
            'test_score': round(test_score, 3) if not np.isnan(test_score) else np.nan
        })


# Формируем итоговый DataFrame

df = pd.DataFrame(data)

print(f"Всего записей (xAPI-событий): {len(df)}")
print(f"Пользователей: {df['user_id'].nunique()}")
print(f"Модулей: {df['module_id'].nunique()}")
print(f"\nОбщая завершаемость (базовая, линейная модель): "
      f"{df['completed'].mean()*100:.1f}%")
print(f"\nПервые 5 записей:")
print(df.head())
print("\nСтатистика по объему текста на слайдах:")
print(df['words_on_slide'].describe())

