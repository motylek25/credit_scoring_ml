# Кредитный скоринг (Australian Credit Approval)

Пет‑проект по скорингу заявок: на датасете Australian Credit из OpenML обучаем логистическую регрессию, подбираем порог по стоимости ошибок, переводим вероятность дефолта в кредитный скор (300–900) и сохраняем модель.

Что делает
- Загружает актуальную версию датасета Australian Credit (OpenML).
- Препроцессинг: числовые (импутация+скейл), категориальные (impute+One‑Hot).
- Модель Logistic Regression (class_weight="balanced").
- Подбор порога по стоимости ошибок (FP дороже FN).
- Оценка на тесте: ROC‑AUC, PR‑AUC, confusion matrix, отчёт.
- Важность признаков (permutation importance).
- Перевод p(bad) → score в диапазоне 300–900 (BASE=600, PDO=50, ODDS0=20).
- Сохранение бандла и функция score_new() для скоринга новых заявок.

Как запустить
1) Установить зависимости:
   ```
   pip install pandas numpy scikit-learn seaborn matplotlib joblib
   ```
2) Открыть ноутбук и выполнить ячейки CS‑0 → CS‑7.
3) В результате:
   - появится файл модели: credit_scoring_australian.joblib;
   - в ноутбуке — метрики, графики, таблица с примерными score.

