import pandas as pd
import os
from datetime import datetime
from load_and_preparation import df
from adapt_modeling import comparison, metrics_comparison


print("БЛОК 4. ЭКСПОРТ ДАННЫХ И ГЕНЕРАЦИЯ ОТЧЕТА")

output_dir = "practice_results"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Сохранение данных

df.to_csv(f"{output_dir}/xapi_logs_enriched.csv", index=False, encoding='utf-8-sig')
comparison.to_csv(f"{output_dir}/user_level_metrics.csv", index=True, encoding='utf-8-sig')

print(f"✓ Данные сохранены в папке '{output_dir}/'")

# Генерация текстового отчета

current_date = datetime.now().strftime("%d.%m.%Y")

report_text = f"""
ОТЧЕТ ПО РЕЗУЛЬТАТАМ СИМУЛЯЦИОННОГО МОДЕЛИРОВАНИЯ
ООО «Образовательные технологии ПРО» | Дата: {current_date}
Автор: Мусорин Г.М. (Д-Э341)

РЕЗУЛЬТАТЫ СИМУЛЯЦИИ:
{metrics_comparison.to_string(index=False)}

ВЫВОДЫ:
- Адаптивная модель повышает Completion Rate и Retention Rate
- Рекомендуется внедрение микролернинга и алгоритма SM-2
"""

report_filename = f"{output_dir}/simulation_report_{current_date.replace('.', '-')}.txt"
with open(report_filename, "w", encoding="utf-8") as f:
    f.write(report_text)

print(f"✓ Отчет сформирован: {report_filename}")

print("ВСЕ АРТЕФАКТЫ СОХРАНЕНЫ. ГОТОВО!")