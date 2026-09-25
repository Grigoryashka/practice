import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from e_data_analysis import df
import load_and_preparation


print("БЛОК 3. СИМУЛЯЦИОННОЕ МОДЕЛИРОВАНИЕ")

# Симуляция адаптивного ветвления (Adaptive Branching)

print("\n1. ПРИМЕНЕНИЕ АЛГОРИТМА АДАПТИВНОГО ВЕТВЛЕНИЯ")
print("-" * 70)

np.random.seed(42)
df['pre_assessment_score'] = np.clip(df['user_skill'] + np.random.normal(0, 0.1, len(df)), 0, 1)

df['sim_completed'] = df['completed']
df['sim_time_on_task'] = df['time_on_task_sec']

for idx, row in df.iterrows():
    if row['pre_assessment_score'] >= 0.80:
        df.at[idx, 'sim_time_on_task'] = row['time_on_task_sec'] * 0.3
        df.at[idx, 'sim_completed'] = 1 if np.random.random() < 0.98 else 0
    else:
        if row['words_on_slide'] > 300:
            boosted_prob = np.clip(row['user_skill'] + 0.4, 0, 0.95)
            df.at[idx, 'sim_completed'] = 1 if np.random.random() < boosted_prob else 0
            df.at[idx, 'sim_time_on_task'] = row['time_on_task_sec'] * 0.9


# Симуляция интервального повторения (SM-2)

print("\n2. ПРИМЕНЕНИЕ АЛГОРИТМА ИНТЕРВАЛЬНОГО ПОВТОРЕНИЯ (SM-2)")
print("-" * 70)

df['baseline_retention'] = df.apply(lambda x: x['test_score'] * 0.20 if x['completed'] == 1 else np.nan, axis=1)
df['sim_retention'] = df.apply(lambda x: x['test_score'] * 0.75 if x['sim_completed'] == 1 else np.nan, axis=1)


# Расчет и сравнение метрик (ИСПРАВЛЕНО: mean вместо max)

print("\n3. СРАВНИТЕЛЬНЫЙ АНАЛИЗ МЕТРИК (БАЗИС vs СИМУЛЯЦИЯ)")
print("-" * 70)

user_baseline = df.groupby('user_id').agg({
    'completed': 'mean',
    'time_on_task_sec': 'sum',
    'baseline_retention': 'mean'
}).rename(columns={'completed': 'base_completed', 'time_on_task_sec': 'base_time', 'baseline_retention': 'base_retention'})

user_sim = df.groupby('user_id').agg({
    'sim_completed': 'mean',
    'sim_time_on_task': 'sum',
    'sim_retention': 'mean'
}).rename(columns={'sim_completed': 'sim_completed', 'sim_time_on_task': 'sim_time', 'sim_retention': 'sim_retention'})

comparison = user_baseline.join(user_sim)


metrics_comparison = pd.DataFrame({
    'Метрика': [
        'Completion Rate (средний % завершенных модулей)',
        'Time-to-competence (среднее время обучения, мин)',
        'Retention Rate (удержание знаний через 30 дней, %)'
    ],
    'Базовая модель (Линейная)': [
        f"{comparison['base_completed'].mean() * 100:.1f}%",
        f"{comparison['base_time'].mean() / 60:.1f}",
        f"{comparison['base_retention'].mean() * 100:.1f}%"
    ],
    'Симуляция (Адаптивная + SM-2)': [
        f"{comparison['sim_completed'].mean() * 100:.1f}%",
        f"{comparison['sim_time'].mean() / 60:.1f}",
        f"{comparison['sim_retention'].mean() * 100:.1f}%"
    ],
    'Дельта (Изменение)': [
        f"+{(comparison['sim_completed'].mean() - comparison['base_completed'].mean()) * 100:.1f} п.п.",
        f"-{((comparison['base_time'].mean() - comparison['sim_time'].mean()) / comparison['base_time'].mean() * 100):.1f}%",
        f"+{(comparison['sim_retention'].mean() - comparison['base_retention'].mean()) * 100:.1f} п.п."
    ]
})

print("\nРЕЗУЛЬТАТЫ СИМУЛЯЦИИ НА ВЫБОРКЕ (N=1200):")
print(comparison.head())
print("СВОДНАЯ ТАБЛИЦА ЭФФЕКТИВНОСТИ")
print(metrics_comparison.to_string(index=False))


# Визуализация результатов симуляции

print("\n4. ПОСТРОЕНИЕ ГРАФИКА СРАВНЕНИЯ")
print("-" * 70)

fig, ax = plt.subplots(1, 3, figsize=(18, 6))

bars1 = ax[0].bar(['Базовая модель', 'Адаптивная симуляция'],
                  [comparison['base_completed'].mean() * 100, comparison['sim_completed'].mean() * 100],
                  color=['#ff9999', '#66b3ff'], edgecolor='black', alpha=0.8)
ax[0].set_ylabel('Процент (%)', fontsize=12)
ax[0].set_title('Completion Rate', fontsize=14, fontweight='bold')
ax[0].bar_label(bars1, fmt='%.1f%%', padding=3)
ax[0].grid(axis='y', alpha=0.3)

bars2 = ax[1].bar(['Базовая модель', 'Адаптивная симуляция'],
                  [comparison['base_time'].mean() / 60, comparison['sim_time'].mean() / 60],
                  color=['#ff9999', '#66b3ff'], edgecolor='black', alpha=0.8)
ax[1].set_ylabel('Минуты', fontsize=12)
ax[1].set_title('Time-to-competence (время обучения)', fontsize=14, fontweight='bold')
ax[1].bar_label(bars2, fmt='%.1f мин', padding=3)
ax[1].grid(axis='y', alpha=0.3)

bars3 = ax[2].bar(['Базовая модель (Эббингауз)', 'Симуляция (SM-2)'],
                  [comparison['base_retention'].mean() * 100, comparison['sim_retention'].mean() * 100],
                  color=['#ff9999', '#66b3ff'], edgecolor='black', alpha=0.8)
ax[2].set_ylabel('Процент (%)', fontsize=12)
ax[2].set_title('Retention Rate (через 30 дней)', fontsize=14, fontweight='bold')
ax[2].bar_label(bars3, fmt='%.1f%%', padding=3)
ax[2].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('simulation_results.png', dpi=300, bbox_inches='tight')
plt.show()

print("✓ График сравнения метрик сохранен в файл 'simulation_results.png'")
