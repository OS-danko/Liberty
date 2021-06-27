#Загрузка айдио файлов
import librosa


#Отображение графикрв
import matplotlib.pyplot as plt
import librosa.display


#
import librosa
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import os
from PIL import Image
import pathlib
import csv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import keras
from keras import layers
from keras import layers
import keras
from keras.models import Sequential
import warnings
warnings.filterwarnings('ignore')
#
#Теперь конвертируем файлы аудиоданных в PNG или извлекаем спектрограмму для каждого аудио.
cmap = plt.get_cmap('inferno')
plt.figure(figsize=(8,8))
genres = 'blues classical country disco hiphop jazz metal pop reggae rock'.split()
for g in genres:
	pathlib.Path(f'img_data/{g}').mkdir(parents=True, exist_ok=True)
	for filename in os.listdir(f'./drive/My Drive/genres/{g}'):
		songname = f'./drive/My Drive/genres/{g}/{filename}'
		y, sr = librosa.load(songname, mono=True, duration=5)
		plt.specgram(y, NFFT=2048, Fs=2, Fc=0, noverlap=128, cmap=cmap, sides='default', mode='default', scale='dB');
		plt.axis('off');
		plt.savefig(f'img_data/{g}/{filename[:-3].replace(".", "")}.png')
		plt.clf()


#Создание заголовка для файла CSV
header = 'filename chroma_stft rmse spectral_centroid spectral_bandwidth rolloff zero_crossing_rate'
for i in range(1, 21):
	header += f' mfcc{i}'
header += ' label'
header = header.split()

#Извлекаем признаки из спектрограммы: MFCC, спектральный центроид, частоту пересечения нуля, частоты цветности и спад спектра
file = open('dataset.csv', 'w', newline='')
with file:
	writer = csv.writer(file)
	writer.writerow(header)
genres = 'blues classical country disco hiphop jazz metal pop reggae rock'.split()
for g in genres:
	for filename in os.listdir(f'./drive/My Drive/genres/{g}'):
		songname = f'./drive/My Drive/genres/{g}/{filename}'
		y, sr = librosa.load(songname, mono=True, duration=30)
		rmse = librosa.feature.rmse(y=y)
		chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
		spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
		spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
		rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
		zcr = librosa.feature.zero_crossing_rate(y)
		mfcc = librosa.feature.mfcc(y=y, sr=sr)
		to_append = f'{filename} {np.mean(chroma_stft)} {np.mean(rmse)} {np.mean(spec_cent)} {np.mean(spec_bw)} {np.mean(rolloff)} {np.mean(zcr)}'    
		for e in mfcc:
			to_append += f' {np.mean(e)}'
		to_append += f' {g}'
		file = open('dataset.csv', 'a', newline='')
		with file:
			writer = csv.writer(file)
			writer.writerow(to_append.split())


#Выполняем предварительную обработку данных, которая включает загрузку данных CSV, создание меток, масштабирование признаков и разбивку данных на наборы для обучения и тестирования.
data = pd.read_csv('dataset.csv')
data.head()
# Удаление ненужных столбцов
data = data.drop(['filename'],axis=1)
# Создание меток
genre_list = data.iloc[:, -1]
encoder = LabelEncoder()
y = encoder.fit_transform(genre_list)
# Масштабирование столбцов признаков
scaler = StandardScaler()
X = scaler.fit_transform(np.array(data.iloc[:, :-1], dtype = float))
# Разделение данных на обучающий и тестовый набор
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


#Создаем модель ANN
model = Sequential()
model.add(layers.Dense(256, activation='relu', input_shape=(X_train.shape[1],)))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


#Подгоняем модель
classifier = model.fit(X_train,
                    y_train,
                    epochs=100,
                    batch_size=128)






#Загрузка айдио файлов
audio_data = 'TF/wav/1.wav'
x , sr = librosa.load(audio_data)
print(type(x), type(sr))
# x , sr = librosa.load(audio_data, sr=44100)
# print(type(x), type(sr))

#Отображение графикрв



X = librosa.stft(x)
Xdb = librosa.amplitude_to_db(abs(X))

fs=sr
mfccs = librosa.feature.mfcc(x, sr=fs)
print(mfccs.shape)
# Отображение MFCC:
plt.figure(figsize=(15, 7))
librosa.display.specshow(mfccs, sr=sr, x_axis='time')
plt.show()



