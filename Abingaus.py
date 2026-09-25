import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from matplotlib.ticker import PercentFormatter

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

days = np.linspace(0, 50, 500)


def forgetting_curve(t, S=2.5):
    return np.exp(-t / S)


review_points = [0, 1, 7, 21, 45]
review_labels = ['Начальное\nобучение', '24 часа', '7 дней', '21 день', '45 дней']

fig, ax = plt.subplots(figsize=(12, 7))

ax.plot(days, forgetting_curve(days), 'r-', linewidth=2.5, label='Кривая забывания', zorder=1)

colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8']
for i, review_day in enumerate(review_points):

    segment_days = np.linspace(review_day, 50, 300)
    retention_after_review = 0.9 * np.exp(-(segment_days - review_day) / (2.5 + i * 0.5))

    if i < len(review_points) - 1:
        next_review = review_points[i + 1]
        mask = segment_days <= next_review
        segment_days_plot = segment_days[mask]
        retention_plot = retention_after_review[mask]
    else:
        segment_days_plot = segment_days
        retention_plot = retention_after_review

    ax.plot(segment_days_plot, retention_plot, color=colors[i], linewidth=2,
            alpha=0.7, zorder=2)

    if review_day > 0:
        ax.axvline(x=review_day, color=colors[i], linestyle='--', alpha=0.5, linewidth=1.5)

        arrow = FancyArrowPatch((review_day, forgetting_curve(review_day)),
                                (review_day, 0.9),
                                arrowstyle='->',
                                mutation_scale=20,
                                lw=2,
                                color=colors[i])
        ax.add_patch(arrow)

for i, (day, label) in enumerate(zip(review_points, review_labels)):
    if day == 0:
        y_pos = 1.0
    else:
        y_pos = forgetting_curve(day)

    ax.scatter([day], [y_pos], color=colors[i], s=150, zorder=5, edgecolors='white', linewidth=2)

    ax.annotate(label,
                xy=(day, y_pos),
                xytext=(day, y_pos + 0.15),
                ha='center',
                fontsize=9,
                fontweight='bold',
                color=colors[i])

ax.set_xlabel('Время (дни)', fontsize=12, fontweight='bold')
ax.set_ylabel('Удержание знаний (%)', fontsize=12, fontweight='bold')
ax.set_title('Кривая забывания Эббингауза и точки применения интервального повторения',
             fontsize=14, fontweight='bold', pad=20)

ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))
ax.set_ylim(0, 1.1)
ax.set_xlim(0, 50)

ax.grid(True, alpha=0.3, linestyle='--')

ax.legend(loc='upper right', fontsize=10, framealpha=0.9)

plt.figtext(0.5, 0.01,
            'После каждого повторения кривая забывания становится более пологой,\nчто обеспечивает лучшее удержание знаний на длительном промежутке времени',
            ha='center', fontsize=10, style='italic', alpha=0.7)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])

plt.savefig('ebbinghaus_forgetting_curve.png', dpi=300, bbox_inches='tight')
plt.savefig('ebbinghaus_forgetting_curve.pdf', bbox_inches='tight')

plt.show()

print("График сохранен в файлы: ebbinghaus_forgetting_curve.png и ebbinghaus_forgetting_curve.pdf")