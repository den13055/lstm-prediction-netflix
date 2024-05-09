import datetime
import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import matplotlib.pyplot as plt

from datetime import date
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# Загрузка обученной модели
model = load_model(r"./Model_50.h5")

st.header('Прогнозирование цен акций компании Netflix')

# Ввод параметров прогноза
st.write('Введите параметры для прогноза:')
period = st.slider("Количество дней для предсказания", 1, 10)

if st.button('Прогнозировать'):
	if period:
		symbol = "NFLX"
		START = '2010-01-01'
		TODAY = date.today().strftime("%Y-%m-%d")

		# Загрузка данных о цене акции Netflix
		data_load_state = st.text('Загрузка данных...')
		data = yf.download(symbol, START ,TODAY)
		data_load_state.text('Данные загружены, удачной торговли!')

		# Отображение данных о цене закрытия акции
		st.subheader('Данные цены закрытия акции')
		st.write(data.Close.tail())

		# График цены закрытия акции
		st.subheader('График цены закрытия акции')
		fig1 = plt.figure(figsize=(8,6))
		plt.plot(data.Close)
		plt.show()
		st.pyplot(fig1)

		# Подготовка данных для модели
		splitting_len = int(len(data)*0.8)
		scaler = MinMaxScaler(feature_range=(0, 1))
		x_test = pd.DataFrame(data.Close[splitting_len:])  # Создаем DataFrame с тестовыми данными из столбца 'Close'
		scaled_data = scaler.fit_transform(x_test[['Close']])  # Масштабируем значения столбца 'Close' методом MinMaxScaler

		x = []
		y = []

		window = 50

		for i in range(window,len(scaled_data)):
			x.append(scaled_data[i-window:i])
			y.append(scaled_data[i])

		x, y = np.array(x), np.array(y)

		# Прогнозирование цен акции
		forecast = model.predict(x)

		inv_forecast = scaler.inverse_transform(forecast)  # Обратное масштабирование прогнозируемых значений
		inv_y_test = scaler.inverse_transform(y)  # Обратное масштабирование исходных значений

		ploting_data = pd.DataFrame(
				{
						'original_test_data': inv_y_test.reshape(-1),  # Исходные значения после обратного масштабирования
						'predictions': inv_forecast.reshape(-1)  # Прогнозируемые значения после обратного масштабирования
				},
				index = data.index[splitting_len+window:]  # Индексация данных для построения графика
		)

		st.subheader("Исходные значения & Прогнозируемые значения")
		st.write(ploting_data)

		st.subheader('Оригинальная цена закрытия & Прогнозируемая цена закрытия')
		fig = plt.figure(figsize=(15,6))
		plt.plot(pd.concat([data.Close[2013 :splitting_len+window],ploting_data], axis=0))
		plt.legend(["Data- not used", "Original Test data", "Predicted Test data"])
		st.pyplot(fig)

		# Расчет значения точности модели
		rms=np.sqrt(np.mean(np.power((inv_y_test.reshape(-1)-inv_forecast.reshape(-1)),2)))
		st.write('Значение RMSE на валидационном множестве: ',rms)

		# Прогнозирование цен на следующие дни
		last_n_days = scaled_data[-window:]  # Выбираем последние `window` дней из отмасштабированных данных
		x_test = np.array([last_n_days])  # Преобразуем последние дни в массив
		x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))  # Изменяем форму массива

		last_sequence = x_test[-1]  # Выбираем последовательность последних дней

		last_sequence = last_sequence.reshape(1, window, 1)  # Изменяем форму последовательности

		predictions_next_days = []  # Создаем список для прогнозируемых значений на следующие дни

		for _ in range(period):
				next_day_prediction = model.predict(last_sequence)  # Прогнозируем значение на следующий день
				predictions_next_days.append(next_day_prediction[0][0])  # Добавляем прогнозируемое значение в список
				last_sequence = np.roll(last_sequence, -1, axis=1)  # Сдвигаем последовательность на один день
				last_sequence[0, -1, 0] = next_day_prediction  # Заменяем последний день последовательности на прогнозируемое значение

		predictions_next_days = scaler.inverse_transform(np.array(predictions_next_days).reshape(-1,1))  # Обратное масштабирование прогнозируемых значений
		tomorrow = datetime.date.today() + datetime.timedelta(1)  # Вычисляем дату завтрашнего дня

		st.subheader(f'Прогнозируемая цена закрытия на следующие {period} дней')
		fig2 = plt.figure(figsize=(8,6))
		plt.plot(predictions_next_days, marker="*", color="green")
		plt.xlabel('Days')
		plt.ylabel("Price")
		plt.xticks(range(period), [f"{tomorrow + datetime.timedelta(days=x)}" for x in range(period)])
		plt.grid(True)
		st.pyplot(fig2)

	else:
		st.write('Пожалуйста, заполните все поля')