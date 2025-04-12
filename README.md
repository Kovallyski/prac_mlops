Учитывая задание и выбранный датасет **Rain in Australia** (https://www.kaggle.com/datasets/jsphyg/weather-dataset-rattle-package), построим MVP MLOps-системы на потоковых табличных данных. Ниже приведён план реализации, соответствующий структуре из задания:

---

### ✅ **Структура проекта**
```
project/
│
├── database/              # Хранилище сырых и обработанных данных
├── models_*/              # Сериализованные модели
├── prepo/                 # Сериализованные препроцессеры
├── reports_*/             # Мониторинг, summary
|
├── config.json                # Конфигурационный файл
├── run.py                 # Скрипт управления режимами (inference/update/summary)
├── pipeline.py            # Основной pipeline обработки
├── data_ingestion.py      # Потоковое получение данных
├── data_analysis.py       # Анализ и очистка данных
├── preprocessing.py       # Предобработка
├── train.py               # Обучение/дообучение модели
├── utils.py               # Общие функции, логирование
├── requirements.txt
├── Dockerfile
└── README.md
```

---

###**Функционал MVP**

#### 1. **Сбор данных** (`database.py`)
- Сохраняем загруженные CSV в "базу данных."
- Считаем метапараметры (кол-во строк, пропуски, диапазоны).
- Подгружаем нужные батчи вместе с их метапараметрами

#### 2. **Анализ данных** (`eda.py`)
- Качество данных: пропуски, уникальные значения, типы.

#### 3. **Подготовка данных** (`preprocessing.py`)
- Импутация пропусков (среднее/мода).
- One-Hot Encoding для категорий.
- Масштабирование числовых признаков.

#### 4. **Обучение и оценка моделей + их обслуживание** (`train.py`)
- Поддержка: `LogisticRegression`, `KNN`, `RandomForest`.
- Хранение версий модели.
- Дообучение
- CV или TimeSeriesSplit.
- Метрики: accuracy, f1.
- Подбор гиперпараметров вручную или через GridSearch.
- Сохранение лучшей модели (`pickle`).
- Поддержка прогнозов: `run.py -mode inference`.
- Интерпретация моделей (LIME и графики для конктретных моделей)

### 🚀 **Команды запуска (run.py)**

```bash
python run.py -mode "inference" -file test.csv
python run.py -mode "update"
python run.py -mode "summary"
python run.py -mode "add_file" -file ".input_data/new_batch.csv"
```
