import pandas
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
trips_data = pandas.read_excel("Machine_Learning/trips_data.xlsx")
#trips_data_salary = plt.hist(trips_data['salary'])
#trips_data_age = plt.hist(trips_data['age']) # сохраняет в переменную выборку по возрасту
#plt.show()
#print(trips_data.query('age > 25 & salary > 230000')) # Находит все строчки возраст которых больше 25 и зп больше 230000

trips_data_correct = pandas.get_dummies(trips_data, columns = ['city', 'transport_preference', 'vacation_preference']) # заменяем все буквенные стобцы - циферными значениями
# trips_data_correct.to_excel("trips_data_output.xlsx") # записываем полученную таблицу в файл trips_data_output

y = trips_data_correct.target # Куда человек поедет в отпуск? то что мы пытаемся предсказать
x = trips_data_correct.drop('target', axis = 1) # выбираем все, кроме колонки (axis = 0) target. 

# Подключаем библиотеку прогноза
model = RandomForestClassifier()
model.fit(x, y) # обучаем модель, ищет закономерности между x и y

# куда поедем?
sample = {'salary': [80000], 
'age': [32], 
'family_members': [1], 
'city_Екатеринбург': [0], 
'city_Киев': [0], 
'city_Краснодар': [0], 
'city_Минск': [0], 
'city_Москва': [0], 
'city_Новосибирск': [0], 
'city_Омск': [0], 
'city_Петербург': [0], 
'city_Томск': [1], 
'city_Хабаровск': [0], 
'city_Ярославль': [0], 
'transport_preference_Автомобиль': [0], 
'transport_preference_Космический корабль': [0], 
'transport_preference_Морской транспорт': [0], 
'transport_preference_Поезд': [0], 
'transport_preference_Самолет': [1], 
'vacation_preference_Архитектура': [0], 
'vacation_preference_Ночные клубы': [0], 
'vacation_preference_Пляжный отдых': [1], 
'vacation_preference_Шоппинг': [0]
}

sample_df = pandas.DataFrame(sample, columns = x.columns)
sample_df.head()

predict_sample = model.predict(sample_df) # делаем предсказание на основе параметров человека в simple, куда ему отправится на отдых
predict_proba_sample = model.predict_proba(sample_df) # Описываем предсказанное решение (выдаст вероятности каждого класса (города))
print(predict_sample)
print(predict_proba_sample)

#Пользуйтесь документацией, гуглом
#2. С помощью pandas, найти:
#   — Самых взрослых людей в каждом городе
#   — У кого из любителей Самолетов самая высокая зарплата?
#   — Кто предпочитает Архитектуру, люди с высокой зарплатой или с низкой?
