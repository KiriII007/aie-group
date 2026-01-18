# HW06 – Report

> Файл: `homeworks/HW06/report.md`  
> Важно: не меняйте названия разделов (заголовков). Заполняйте текстом и/или вставляйте результаты.

## 1. Dataset

- Какой датасет выбран: `S06-hw-dataset-04.csv`
- Размер: (25000, 62)
- Целевая переменная: `target`
    `0`    23770 0.9508
    `1`    1230  0.0492
- Признаки: int64, float. Пропусков нет.

## 2. Protocol

- Разбиение: train/test
    - test_size = 0.25
    - random_state = 42
    - stratify = y
- Подбор: CV на train
    CV: StratifiedKFold, n_splits = 5, shuffle=True, random_state=42
    Оптимизировали: ROC-AUC на train-CV (scoring = "roc_auc").
    Test использовался один раз для финальной оценки выбранных моделей.
- Метрики: 
accuracy — базовая метрика “в целом”, но на дисбалансе может вводить в заблуждение.
F1 — баланс precision/recall для ненулевого (редкого) класса, важнее accuracy при дисбалансе.
ROC-AUC — измеряет качество ранжирования по вероятностям и менее чувствителен к дисбалансу, поэтому выбран как основной критерий выбора лучшей модели.
Average Precision — лучше отражает качество при сильном дисбалансе.

## 3. Models

DummyClassifier: strategy="most_frequent".
LogisticRegression (baseline): SimpleImputer(median) + StandardScaler + LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42).
DecisionTreeClassifier (GridSearchCV, roc_auc):
grid: max_depth=[2,3,4,5,7,10,None], min_samples_leaf=[1,5,10,20,50], ccp_alpha=[0,1e-4,1e-3,1e-2]
best params: max_depth=7, min_samples_leaf=50, ccp_alpha=0.001
best CV roc_auc = 0.8326
RandomForestClassifier (GridSearchCV, roc_auc):
фиксировано: n_estimators=100, class_weight="balanced_subsample"
grid: max_depth=[4,6], min_samples_leaf=[5,10], max_features=["sqrt",0.5]
best params: max_depth=6, min_samples_leaf=5, max_features=0.5
best CV roc_auc = 0.8873
HistGradientBoostingClassifier (GridSearchCV, roc_auc):
grid: learning_rate=[0.03,0.05], max_depth=[2,3,5], max_iter=[200,400], min_samples_leaf=[20,50]
best params: learning_rate=0.05, max_depth=5, max_iter=200, min_samples_leaf=50
best CV roc_auc = 0.8959

## 4. Results

	accuracy	f1	roc_auc	average_precision
hist_gb	0.97920	0.736842	0.902419	0.791208
random_forest	0.95248	0.601342	0.897605	0.717886
logreg_balanced	0.77920	0.257266	0.841861	0.457002
decision_tree	0.87200	0.356913	0.839639	0.376445
dummy_most_frequent	0.95088	0.000000	NaN	NaN

Победитель: HistGradientBoostingClassifier по критерию ROC-AUC на test = 0.9024 (и также лучший по F1 и AP).


## 5. Analysis

- Устойчивость: Если менять random_state то точности моделей меняются но не значитьельно и лучшей все равно остается "HistGradientBoostingClassifier"
- Ошибки: Матрица ошибкок -
- [5938,    5]
  [ 125,  182]
По ней хорошо видно что модель почти без ошибок определяет класс 0, но с классом 1 у нее проблемы, но все равно количество правильных определений больше чем ложно положительных
- Топ-15 признаков по permutation importance:

f54: 0.02449 ± 0.00364
f25: 0.01828 ± 0.00401
f47: 0.01081 ± 0.00561
f58: 0.01080 ± 0.00407
f33: 0.00954 ± 0.00188
f38: 0.00913 ± 0.00234
f04: 0.00699 ± 0.00190
f53: 0.00533 ± 0.00332
f41: 0.00426 ± 0.00205
f16: 0.00256 ± 0.00145
f52: 0.00229 ± 0.00125
f07: 0.00221 ± 0.00126
f29: 0.00221 ± 0.00047
f11: 0.00193 ± 0.00269
f43: 0.00188 ± 0.00177
Выводы:
Самый сильный вклад дают f54 и f25: перемешивание каждого из них заметно снижает ROC-AUC, значит модель активно использует информацию этих признаков для отделения редкого класса.
Набор важнейших признаков достаточно широкий (не 1–2 фичи), что похоже на ситуацию, когда сигнал “размазан” по нескольким измерениям — типично для синтетических fraud-like данных.
Важно: permutation importance показывает вклад признаков в качество модели, но не является причинно-следственным доказательством

## 6. Conclusion

Дерево решений легко переобучается; контроль сложности (max_depth, min_samples_leaf, ccp_alpha) критичен.
RandomForest обычно дают более устойчивое качество, чем одиночное дерево.
Boosting часто выигрывает на сложных нелинейных зависимостях, но требует аккуратной настройки.
На сильном дисбалансе accuracy недостаточна; важны ROC-AUC и метрики, чувствительные к классу 1.
Сохранение артефактов (метрики, параметры, модель, графики) делает эксперимент воспроизводимым.