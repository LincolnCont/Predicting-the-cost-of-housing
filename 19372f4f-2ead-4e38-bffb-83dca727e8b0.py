#!/usr/bin/env python
# coding: utf-8

# ## Предсказание стоимости жилья
# 
# В проекте вам нужно обучить модель линейной регрессии на данных о жилье в Калифорнии в 1990 году. На основе данных нужно предсказать медианную стоимость дома в жилом массиве. Обучите модель и сделайте предсказания на тестовой выборке. Для оценки качества модели используйте метрики RMSE, MAE и R2.

# # План выполнения проекта
# Инициализируем локальную Spark-сессию.
# 
# Прочитайтем содержимое файла /datasets/housing.csv.
# 
# Выведем типы данных колонок датасета. Используем методы pySpark.
# 
# Выполним предобработку данных:
#     Исследуем данные на наличие пропусков и заполним их, выбрав значения по своему усмотрению.
#     Преобразуем колонку с категориальными значениями техникой One hot encoding.
# 
# Построим две модели линейной регрессии на разных наборах данных:
#     используя все данные из файла;
#     используя только числовые переменные, исключив категориальные.
#     Для построения модели используем оценщик LinearRegression из библиотеки MLlib.
# 
# Сравним результаты работы линейной регрессии на двух наборах данных по метрикам RMSE, MAE и R2.

# # Подготовка данных

# Импорт необходимых библиотек и запуск spark сессии

# In[1]:


import pandas as pd 
import numpy as np

import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.types import *
import pyspark.sql.functions as F

from pyspark.ml.feature import Imputer
from pyspark.ml.feature import StringIndexer, VectorAssembler, StandardScaler
from pyspark.ml.regression import LinearRegression

pyspark_version = pyspark.__version__
if int(pyspark_version[:1]) == 3:
    from pyspark.ml.feature import OneHotEncoder    
elif int(pyspark_version[:1]) == 2:
    from pyspark.ml.feature import OneHotEncodeEstimator
        
SEED = 31416

spark = SparkSession.builder                     .master("local")                     .appName("California Housing ML")                     .getOrCreate()


# Прочитайтем содержимое файла /datasets/housing.csv

# In[2]:


df = spark.read.load('/datasets/housing.csv', format='csv', sep=',', inferSchema=True, header='true')
df.printSchema()


# <div class="alert alert-success">
# <b>Комментарий ревьюера ✔️:</b> Огонь, данные на месте:)</div>

# In[3]:


pd.DataFrame(df.dtypes, columns=['column', 'type'])


# In[4]:


df.show(10)


# Изучим статистику по данным

# In[5]:


df.describe().toPandas()


# Исследуем данные на наличие пропусков

# In[6]:


df.select([F.count(F.when(F.isnan(c) | F.col(c).isNull(), c)).alias(c) for c in df.columns]).show()


# <div class="alert alert-success">
# <b>Комментарий ревьюера ✔️:</b> Да, хорошо
# 
# Можно еще вот так:
#     
# ```python
# columns = data.columns
# 
# for column in columns:
#     check_col = F.col(column).isNull()
#     print(column, data.filter(check_col).count())    
#     
# ```
# 
# 
# </div>

# Заполним пропуски медианными значениями

# In[7]:


imputer = Imputer(strategy='median', inputCols=['total_bedrooms'], outputCols=['total_bedrooms'])
df = imputer.fit(df).transform(df)
df.describe().toPandas()


# <div class="alert alert-success">
# <b>Комментарий ревьюера ✔️:</b> Корректно заполнил
# 
# 
# </div>

# Среднее и стандартное отклонение почти не изменились, но пропуском больше нет

# In[8]:


categorical_cols = ['ocean_proximity']
numerical_cols  = ['longitude', 'latitude', 'housing_median_age', 'total_rooms',
                   'total_bedrooms', 'population', 'households', 'median_income']
target = 'median_house_value' 


# Применим StringIndexer для перевода категориальных признаков в численные

# In[9]:


indexer = StringIndexer(inputCols=categorical_cols, 
                        outputCols=[c+'_idx' for c in categorical_cols]) 
df = indexer.fit(df).transform(df)

cols = [c for c in df.columns for i in categorical_cols if (c.startswith(i))]
df.select(cols).show(3)


# Теперь можем применить OHE

# In[10]:


encoder = OneHotEncoder(inputCols=[c+'_idx' for c in categorical_cols],
                        outputCols=[c+'_ohe' for c in categorical_cols])
df = encoder.fit(df).transform(df)

cols = [c for c in df.columns for i in categorical_cols if (c.startswith(i))]
df.select(cols).show(3)


# <div class="alert alert-success">
# <b>Комментарий ревьюера ✔️:</b> Отлично, категориальные данные готовы)
# 
# 
# </div>

# Для числовых признаков нужно шкалирование значений, чтобы сильные выбросы не смещали предсказания модели. Применим StandardScaler.

# Соберём вектор числовых признаков в отдельный столбец

# In[11]:


numerical_assembler = VectorAssembler(inputCols=numerical_cols, outputCol="numerical_features")
df = numerical_assembler.transform(df) 


# In[12]:


standardScaler = StandardScaler(inputCol='numerical_features', outputCol="numerical_features_scaled")
df = standardScaler.fit(df).transform(df)
df.printSchema()


# <div class="alert alert-success">
# <b>Комментарий ревьюера ✔️:</b> Хорошо, отмасштабировал корректно:)
# 
# 
# </div>

# Осталось собрать трансформированные категорийные и числовые признаки с помощью VectorAssembler

# In[13]:


all_features = ['ocean_proximity_ohe','numerical_features_scaled']

final_assembler = VectorAssembler(inputCols=all_features, outputCol='features')
df = final_assembler.transform(df)


# In[14]:


df.select('features').show(3)
df.select('numerical_features_scaled').show(3)


# In[ ]:





# # Обучение моделей

# Разделим наши наборы данных на тренировочные и тестовые

# In[15]:


train, test = df.randomSplit([.8,.2], seed=SEED)
print(train.count(), test.count())


# Создадим и обучим оценщик LinearRegression

# In[16]:


lr_all = LinearRegression(labelCol=target, featuresCol='features',
                          maxIter=10, regParam=0.3, elasticNetParam=0.8)
model_all = lr_all.fit(train)

lr_num = LinearRegression(labelCol=target, featuresCol='numerical_features_scaled',
                          maxIter=10, regParam=0.3, elasticNetParam=0.8)
model_num = lr_num.fit(train)


# <div class="alert alert-success">
# <b>Комментарий ревьюера ✔️:</b> Все верно обучил:)
# 
# 
# </div>

# Посмотрим на предсказания моделей, со всеми признаками:

# In[17]:


predictions_all = model_all.transform(test)

predicted_all = predictions_all.select('median_house_value', 'prediction')
predicted_all.show(10)


# И только с числовыми признаками:

# In[18]:


predictions_num = model_num.transform(test)

predicted_num = predictions_num.select('median_house_value', 'prediction')
predicted_num.show(10)


# # Анализ результатов

# Сравним результаты работы линейной регрессии на двух наборах данных по метрикам RMSE, MAE и R2

# In[19]:


training_summary_all = model_all.summary
print('RMSE: %f' % training_summary_all.rootMeanSquaredError)
print('MAE: %f' % training_summary_all.meanAbsoluteError)
print('R2: %f' % training_summary_all.r2)


# In[20]:


training_summary_num = model_num.summary
print('RMSE: %f' % training_summary_num.rootMeanSquaredError)
print('MAE: %f' % training_summary_num.meanAbsoluteError)
print('R2: %f' % training_summary_num.r2)


# RMSE и MAE измеряют разницу между предсказанным значением медианной цены дома и его реальной ценой. Реальная цена по набору лежит в диапазоне от 15000 до 500000. Ошибки в случае с использованием всего набора данных чуть меньше. Но не существенно.
# 
# Коэффициент детерминации, показывающий долю корректных предсказаний нашей модели выше у модели учитывающей все признаки.
# 
# Проверим наши модели на тестовых данных.

# In[21]:


test_summary_all = model_all.evaluate(test)
print('RMSE: %f' % test_summary_all.rootMeanSquaredError)
print('MAE: %f' % test_summary_all.meanAbsoluteError)
print('R2: %f' % test_summary_all.r2)


# In[22]:


test_summary_num = model_num.evaluate(test)
print('RMSE: %f' % test_summary_num.rootMeanSquaredError)
print('MAE: %f' % test_summary_num.meanAbsoluteError)
print('R2: %f' % test_summary_num.r2)


# <div class="alert alert-success">
# <b>Комментарий ревьюера ✔️:</b>  
# 
# Отлично, протестировал все верно:)
# 
# 
# </div>

# # Выводы
# 
# Мы проверили результаты работы линейной регрессии на обучающих и тестовых наборах данных.
# 
# Вычислили по три метрики для всех результатов: RMSE, MAE, R2.
# 
# На тестовом наборе данных лучший результат показала модель обученная как на количественных так и на категориальном признаках.
# 
# На тестовом наборе все метрики ухудшились, но незначительно.

# В данном проекте мы обучили модель линейной регрессии на данных о жилье в Калифорнии в 1990 году.
# 
# Научились инициализировать Spark сессии, загружать данные и обрабатывать их с помощью DataFrame API и работать с моделями в библиотеке MLlib (DataFrame-based).

# In[24]:


spark.stop()


# <div class="alert alert-success">
# <b>Комментарий ревьюера ✔️:</b>  
# 
# Немного полезной информации:
# +  https://www.tutorialspoint.com/pyspark/index.htm
# +  https://www.guru99.com/pyspark-tutorial.html
# +  https://databricks.com/spark/getting-started-with-apache-spark/machine-learning#load-sample-data
# 
# 
# </div>

# <font color='blue'><b>Итоговый комментарий ревьюера</b></font>
# <div class="alert alert-success">
# <b>Комментарий ревьюера ✔️:</b>Сергей, спасибо за хороший проект!!! Я готов принять работу, но хочу убедиться, что тебе все понятно.<br>
# Если есть какие либо вопросы я с удовольствием на них отвечу:)</div>
# 

# <div class="alert alert-success">
# <b>Комментарий ревьюера V2✔️:</b>  
# 
# Удачи в следующих проектах!!!
# </div>
