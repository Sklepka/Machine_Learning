from statistics import LinearRegression
import matplotlib.pyplot
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.ensemble
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import max_error, mean_absolute_error,r2_score

# https://www.CryptoDataDownload.com/data/bitfinex ссылка на данные по торгам 

bitcoin = pd.read_csv("Bitcoin/Bitfinex_BTCUSD_d.csv", sep = ";", index_col = 0) # Читаем файл и убираем лишнюю колонку, разделитель указываем ";"
# преобразуем строку в структуру "DateTime"
bitcoin.date = pd.to_datetime(bitcoin.date,  format="%d.%m.%Y %H:%M")


# plt.plot(bitcoin.date, bitcoin.open, label = "open") # строим зависимость даты от цены открытия биткоина
# plt.show()

# roll_window = bitcoin.open.rolling(window = 60).mean().plot() # Rolling Window = скользящее окно. Среднее значение за 60 дней (строк)
# plt.show()

# Строим модель предсказания. Х = "добавить колонки на основе недавних значений", н = "Цена закрытия завтрашнего дня (какой завтра будет close"? задача - регрессия.
# Feature Engineering - придумывание новых колонок
# Важный момент: не включать текущий день = shift(1) (например в среднем значении за семь дней, седьмой день включается в расчет)

bitcoin["open_mean_7d"] = bitcoin.open.shift(1).rolling(window = 7).mean() # Среднее значение открытия за последнии 7 дней
bitcoin["volume_btc_max_30d"] = bitcoin["Volume BTC"].shift(1).rolling(window = 30).max() # Максимальное объем торгов за 30 дней

# Создаем 7 колонок с недавними значениями close, для проверки на сколько модель правильна
for day in range(1, 8):
    #print(f"Добавляем колонку close за {day} д. назад")
    bitcoin[f"close_{day}d"] = bitcoin["close"].shift(day)

# ToDO: Добавить колонки на основе дня месяца и года, взяв информацию из колонки date
# Например bitcoin.date.dt.weekday
# https://pandas.pydata.org/docs/reference/api/pandes.Series.dt.weekday.html

# Уберем колонки date и symbol
bitcoin.drop("symbol", axis=1, inplace=True)
bitcoin.drop("date", axis=1, inplace=True)

# Избавляемся от NAN
bitcoin.fillna(method = "backfill", inplace = True) # fillna - позволяет заполнить пустые значения

bitcoin["target"] = bitcoin["close"].shift(-1) # target - close следующего дня/ но мы создали еще один NAN

# [:-1] "все, кроме последнего элемента"
X = bitcoin[:-1].drop("target", axis=1)
y = bitcoin[:-1].target

# Разбиваем на тренировочную и тестовую выборку
# train - обучающая - учебник - X-train, y-train
# test - проверочная - экзамен - Даем модели X_test
# и просим сделать предсказание y_pred, и сравнием с  y _test
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.33) # разделяем и отдаем 33% на тестовую выборку


def try_model(model):
    model.fit(X_train,y_train) # Обучаем модель на тренировочной выборке
    y_pred = model.predict(X_test) # Проверяем модель на тестовой выборке
    print(model)
    print("max_eror = ", max_error(y_pred, y_test))
    print("mean_absolute_error = ", mean_absolute_error(y_pred, y_test))
    print("r2_score = ", r2_score(y_pred, y_test))


rfr = RandomForestRegressor()
rfr.feature_importances_ # важность колонок, показывает влияние колонок оказываемое на модель
pd.DataFrame(data = [rfr.feature_importances_], columns = X.columns).T.sort_values(by = 0, ascending = False) # Выведем колонки в порядке важности
try_model(RandomForestRegressor(n_estimators=1000000))
try_model(LinearRegression())

# ToDo: Пробовать разные модели
# ToDo: Пробовать разные настройки ?
# ToDo: Оценить модель разными метриками качества
# ToDo: Пробуем разные данные (Feature Engineering)
# Задание 1. Используйте модель MLPRegressor для предсказаний и попробуйте  поменять ее настройки чтобы получить ошибки(mae, max, r2) меньше чем было  в эфире
# Задание 2. Попробуйте использовать три других модели регрессии из sklearn(которые мы еще не пробовали), попробуйте указать какие-то настройки модели(см. документацию) так чтобы получить как можно меньшую ошибку
# Задание 3. Попробуйте делать предсказания не на 1 день вперед(мы делали предсказание на завтра) а на 5 дней вперед, т.е. Предсказывать курс, которые будет через 5 дней, что получится?
# Задание 4. Попробуйте также взять разные варианты создания новых колонок. Подумайте какие еще производные колонки можно добавить?
