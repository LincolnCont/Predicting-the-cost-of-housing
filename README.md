### Предсказание стоимости жилья

Инициализируем локальную Spark-сессию.

Прочитайтем содержимое файла /datasets/housing.csv.

Выведем типы данных колонок датасета. Используем методы pySpark.

Выполним предобработку данных: Исследуем данные на наличие пропусков и заполним их, выбрав значения по своему усмотрению. Преобразуем колонку с категориальными значениями техникой One hot encoding.

Построим две модели линейной регрессии на разных наборах данных: используя все данные из файла; используя только числовые переменные, исключив категориальные. Для построения модели используем оценщик LinearRegression из библиотеки MLlib.

Сравним результаты работы линейной регрессии на двух наборах данных по метрикам RMSE, MAE и R2.
