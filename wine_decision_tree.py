import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import ListedColormap
from sklearn.datasets import load_wine
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.model_selection import cross_val_score, StratifiedKFold, learning_curve
from sklearn.metrics import (confusion_matrix, classification_report,
                              ConfusionMatrixDisplay, roc_curve, auc)
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# ── Colores del estilo ──────────────────────────────────────────────────────
WINE_RED   = '#8B1A1A'
WINE_GOLD  = '#C8A951'
WINE_CREAM = '#F5F0E8'
WINE_DARK  = '#2C1810'
WINE_ROSE  = '#D4826A'
ACCENT1    = '#5B7FA6'
ACCENT2    = '#6B9E6B'
ACCENT3    = '#9B6B9B'

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.facecolor': WINE_CREAM,
    'axes.facecolor': '#FBF7F0',
    'axes.labelcolor': WINE_DARK,
    'xtick.color': WINE_DARK,
    'ytick.color': WINE_DARK,
    'text.color': WINE_DARK,
})

# ── 1. Cargar dataset ───────────────────────────────────────────────────────
wine = load_wine()
X, y = wine.data, wine.target
feature_names = wine.feature_names
class_names = wine.target_names

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y)

# ── 2. Hiperparámetros ──────────────────────────────────────────────────────
configs = [
    {'max_depth': 3,    'min_samples_split': 2,  'min_samples_leaf': 1, 'criterion': 'gini'},
    {'max_depth': 5,    'min_samples_split': 5,  'min_samples_leaf': 2, 'criterion': 'gini'},
    {'max_depth': 7,    'min_samples_split': 10, 'min_samples_leaf': 4, 'criterion': 'entropy'},
    {'max_depth': None, 'min_samples_split': 2,  'min_samples_leaf': 1, 'criterion': 'gini'},
]
labels = ['Profundidad=3\nGini', 'Profundidad=5\nGini', 'Profundidad=7\nEntropy', 'Sin límite\nGini']

# Modelo principal con la mejor configuración
best_config = configs[0]
clf = DecisionTreeClassifier(**best_config, random_state=42)
clf.fit(X_train, y_train)

# ═══════════════════════════════════════════════════════════════════════════
# FIGURA 1 — Árbol de decisión
# ═══════════════════════════════════════════════════════════════════════════
fig1, ax1 = plt.subplots(figsize=(28, 12), facecolor=WINE_CREAM)
ax1.set_facecolor(WINE_CREAM)
plot_tree(clf, feature_names=feature_names, class_names=class_names,
          filled=True, rounded=True, fontsize=9, ax=ax1,
          impurity=True, proportion=False,
          node_ids=False, precision=2)

ax1.set_title('🍷  Árbol de Decisión — Dataset Wine\n'
              f'Profundidad={best_config["max_depth"]}  |  '
              f'min_samples_split={best_config["min_samples_split"]}  |  '
              f'Criterio={best_config["criterion"]}',
              fontsize=16, fontweight='bold', color=WINE_RED, pad=20)

plt.tight_layout()
plt.savefig('01_arbol_decision.png', dpi=150, bbox_inches='tight', facecolor=WINE_CREAM)
plt.show()
print("✅  Figura 1 guardada")

# ═══════════════════════════════════════════════════════════════════════════
# FIGURA 2 — Comparación de hiperparámetros (CV)
# ═══════════════════════════════════════════════════════════════════════════
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
cv_scores = []
for cfg in configs:
    m = DecisionTreeClassifier(**cfg, random_state=42)
    scores = cross_val_score(m, X, y, cv=cv, scoring='accuracy')
    cv_scores.append(scores)

fig2, axes = plt.subplots(1, 2, figsize=(16, 6), facecolor=WINE_CREAM)
colors = [WINE_RED, WINE_GOLD, ACCENT1, ACCENT2]

# Boxplot
ax = axes[0]
bp = ax.boxplot(cv_scores, patch_artist=True, notch=True,
                medianprops=dict(color='white', linewidth=2.5))
for patch, c in zip(bp['boxes'], colors):
    patch.set_facecolor(c); patch.set_alpha(0.85)
for whisker in bp['whiskers']: whisker.set_color(WINE_DARK)
for cap in bp['caps']: cap.set_color(WINE_DARK)
for flier in bp['fliers']: flier.set(marker='o', color=WINE_ROSE, alpha=0.6)
ax.set_xticklabels(labels, fontsize=9)
ax.set_ylabel('Accuracy (CV 10-fold)', fontsize=11)
ax.set_title('Distribución de Accuracy\npor configuración', fontsize=13, fontweight='bold', color=WINE_RED)
ax.set_ylim(0.85, 1.02)
ax.axhline(0.95, color=WINE_ROSE, ls='--', lw=1.5, alpha=0.7, label='Referencia 95%')
ax.legend(fontsize=9)

# Barras de media ± std
ax = axes[1]
means = [s.mean() for s in cv_scores]
stds  = [s.std()  for s in cv_scores]
x = np.arange(len(configs))
bars = ax.bar(x, means, color=colors, alpha=0.85, edgecolor='white', linewidth=1.5)
ax.errorbar(x, means, yerr=stds, fmt='none', color=WINE_DARK, capsize=6, linewidth=2)
for i, (m, s) in enumerate(zip(means, stds)):
    ax.text(i, m + s + 0.005, f'{m:.3f}', ha='center', fontsize=10, fontweight='bold', color=WINE_DARK)
ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
ax.set_ylabel('Accuracy Media', fontsize=11)
ax.set_title('Media ± Desv. Estándar\nValidación Cruzada (k=10)', fontsize=13, fontweight='bold', color=WINE_RED)
ax.set_ylim(0.85, 1.05)

fig2.suptitle('🍷  Comparación de Hiperparámetros — Wine Dataset',
              fontsize=15, fontweight='bold', color=WINE_RED, y=1.02)
plt.tight_layout()
plt.savefig('02_comparacion_hiperparametros.png', dpi=150, bbox_inches='tight', facecolor=WINE_CREAM)
plt.show()
print("✅  Figura 2 guardada")

# ═══════════════════════════════════════════════════════════════════════════
# FIGURA 3 — Curva de aprendizaje & profundidad vs accuracy
# ═══════════════════════════════════════════════════════════════════════════
fig3, axes = plt.subplots(1, 2, figsize=(16, 6), facecolor=WINE_CREAM)

# Curva de aprendizaje
train_sizes, train_scores, val_scores = learning_curve(
    clf, X, y, cv=cv, scoring='accuracy',
    train_sizes=np.linspace(0.1, 1.0, 10), random_state=42)

ax = axes[0]
ax.fill_between(train_sizes,
                train_scores.mean(1) - train_scores.std(1),
                train_scores.mean(1) + train_scores.std(1), alpha=0.15, color=WINE_RED)
ax.fill_between(train_sizes,
                val_scores.mean(1) - val_scores.std(1),
                val_scores.mean(1) + val_scores.std(1), alpha=0.15, color=ACCENT1)
ax.plot(train_sizes, train_scores.mean(1), 'o-', color=WINE_RED, lw=2.5, label='Entrenamiento')
ax.plot(train_sizes, val_scores.mean(1),   's-', color=ACCENT1, lw=2.5, label='Validación')
ax.set_xlabel('Muestras de Entrenamiento', fontsize=11)
ax.set_ylabel('Accuracy', fontsize=11)
ax.set_title('Curva de Aprendizaje', fontsize=13, fontweight='bold', color=WINE_RED)
ax.legend(fontsize=10); ax.set_ylim(0.7, 1.05)
ax.grid(True, alpha=0.3)

# Profundidad vs accuracy
depths = range(1, 15)
train_accs, val_accs = [], []
for d in depths:
    m = DecisionTreeClassifier(max_depth=d, random_state=42)
    m.fit(X_train, y_train)
    train_accs.append(m.score(X_train, y_train))
    val_accs.append(cross_val_score(m, X, y, cv=cv, scoring='accuracy').mean())

ax = axes[1]
ax.plot(depths, train_accs, 'o-', color=WINE_RED, lw=2.5, label='Train')
ax.plot(depths, val_accs,   's-', color=ACCENT1,  lw=2.5, label='CV Val')
ax.axvline(3, color=WINE_GOLD, ls='--', lw=2, label='Mejor profundidad (3)')
ax.fill_between(depths, train_accs, val_accs, alpha=0.1, color=WINE_ROSE, label='Overfitting zone')
ax.set_xlabel('Profundidad Máxima', fontsize=11)
ax.set_ylabel('Accuracy', fontsize=11)
ax.set_title('Profundidad vs Accuracy', fontsize=13, fontweight='bold', color=WINE_RED)
ax.legend(fontsize=10); ax.set_ylim(0.7, 1.05); ax.grid(True, alpha=0.3)

fig3.suptitle('🍷  Análisis de Rendimiento — Wine Dataset',
              fontsize=15, fontweight='bold', color=WINE_RED, y=1.02)
plt.tight_layout()
plt.savefig('03_curvas_aprendizaje.png', dpi=150, bbox_inches='tight', facecolor=WINE_CREAM)
plt.show()
print("✅  Figura 3 guardada")

# ═══════════════════════════════════════════════════════════════════════════
# FIGURA 4 — Matriz de confusión & Reporte de clasificación
# ═══════════════════════════════════════════════════════════════════════════
y_pred = clf.predict(X_test)
cm = confusion_matrix(y_test, y_pred)

fig4, axes = plt.subplots(1, 2, figsize=(16, 6), facecolor=WINE_CREAM)

# Matriz de confusión
ax = axes[0]
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(ax=ax, colorbar=False, cmap='Reds')
ax.set_title('Matriz de Confusión\n(Conjunto de Prueba)', fontsize=13, fontweight='bold', color=WINE_RED)
ax.set_xlabel('Predicción', fontsize=11)
ax.set_ylabel('Real', fontsize=11)
for text in ax.texts:
    text.set_fontsize(14); text.set_fontweight('bold')

# Reporte como heatmap
report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)
metrics = ['precision', 'recall', 'f1-score']
data_matrix = np.array([[report[c][m] for m in metrics] for c in class_names])

ax = axes[1]
im = ax.imshow(data_matrix, cmap='YlOrRd', vmin=0.7, vmax=1.0, aspect='auto')
ax.set_xticks(range(len(metrics)))
ax.set_xticklabels(['Precisión', 'Recall', 'F1-Score'], fontsize=11, fontweight='bold')
ax.set_yticks(range(len(class_names)))
ax.set_yticklabels(class_names, fontsize=11)
for i in range(len(class_names)):
    for j in range(len(metrics)):
        ax.text(j, i, f'{data_matrix[i,j]:.3f}', ha='center', va='center',
                fontsize=13, fontweight='bold',
                color='white' if data_matrix[i,j] > 0.9 else WINE_DARK)
plt.colorbar(im, ax=ax, shrink=0.8)
ax.set_title('Métricas por Clase\n(Clasificación)', fontsize=13, fontweight='bold', color=WINE_RED)
overall_acc = np.sum(np.diag(cm)) / np.sum(cm)
ax.text(0.5, -0.12, f'Accuracy General: {overall_acc:.3f}', transform=ax.transAxes,
        ha='center', fontsize=12, fontweight='bold', color=WINE_RED)

fig4.suptitle('🍷  Evaluación del Clasificador — Wine Dataset',
              fontsize=15, fontweight='bold', color=WINE_RED, y=1.02)
plt.tight_layout()
plt.savefig('04_confusion_y_metricas.png', dpi=150, bbox_inches='tight', facecolor=WINE_CREAM)
plt.show()
print("✅  Figura 4 guardada")

# ═══════════════════════════════════════════════════════════════════════════
# FIGURA 5 — Importancia de características & Curvas ROC
# ═══════════════════════════════════════════════════════════════════════════
fig5, axes = plt.subplots(1, 2, figsize=(18, 7), facecolor=WINE_CREAM)

# Importancia de características
importances = clf.feature_importances_
sorted_idx = np.argsort(importances)[::-1]
feat_sorted = [feature_names[i] for i in sorted_idx]
imp_sorted  = importances[sorted_idx]

ax = axes[0]
bar_colors = [WINE_RED if i < 3 else WINE_GOLD if i < 6 else ACCENT1
              for i in range(len(feat_sorted))]
bars = ax.barh(range(len(feat_sorted)), imp_sorted[::-1], color=bar_colors[::-1],
               edgecolor='white', height=0.7)
ax.set_yticks(range(len(feat_sorted)))
ax.set_yticklabels(feat_sorted[::-1], fontsize=9)
ax.set_xlabel('Importancia (Gini)', fontsize=11)
ax.set_title('Importancia de Características', fontsize=13, fontweight='bold', color=WINE_RED)
for i, (bar, imp) in enumerate(zip(bars, imp_sorted[::-1])):
    if imp > 0.01:
        ax.text(imp + 0.002, bar.get_y() + bar.get_height()/2,
                f'{imp:.3f}', va='center', fontsize=9, fontweight='bold', color=WINE_DARK)
ax.grid(True, axis='x', alpha=0.3)

# Curvas ROC (One-vs-Rest)
ax = axes[1]
y_bin = label_binarize(y_test, classes=[0, 1, 2])
y_score = clf.predict_proba(X_test)
roc_colors = [WINE_RED, WINE_GOLD, ACCENT1]

for i, (cls, col) in enumerate(zip(class_names, roc_colors)):
    fpr, tpr, _ = roc_curve(y_bin[:, i], y_score[:, i])
    roc_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, lw=2.5, color=col, label=f'{cls} (AUC = {roc_auc:.3f})')
    ax.fill_between(fpr, tpr, alpha=0.08, color=col)

ax.plot([0,1],[0,1], 'k--', lw=1.5, alpha=0.5, label='Azar (AUC = 0.500)')
ax.set_xlabel('Tasa de Falsos Positivos', fontsize=11)
ax.set_ylabel('Tasa de Verdaderos Positivos', fontsize=11)
ax.set_title('Curvas ROC (One-vs-Rest)', fontsize=13, fontweight='bold', color=WINE_RED)
ax.legend(fontsize=10, loc='lower right')
ax.grid(True, alpha=0.3)

fig5.suptitle('🍷  Características y Curvas ROC — Wine Dataset',
              fontsize=15, fontweight='bold', color=WINE_RED, y=1.02)
plt.tight_layout()
plt.savefig('05_importancia_y_roc.png', dpi=150, bbox_inches='tight', facecolor=WINE_CREAM)
plt.show()
print("✅  Figura 5 guardada")

# ─── Resumen final ──────────────────────────────────────────────────────────
print("\n" + "═"*55)
print("  RESUMEN DEL CLASIFICADOR — WINE DATASET")
print("═"*55)
print(f"  Muestras totales : {X.shape[0]}")
print(f"  Características  : {X.shape[1]}")
print(f"  Clases           : {list(class_names)}")
print(f"\n  Configuración seleccionada:")
for k, v in best_config.items():
    print(f"    {k}: {v}")
print(f"\n  Accuracy en test : {overall_acc:.4f}")
print(f"  CV 10-fold mean  : {cv_scores[0].mean():.4f} ± {cv_scores[0].std():.4f}")
print(f"\n  Feature más importante: {feat_sorted[0]} ({imp_sorted[0]:.3f})")
print("═"*55)
