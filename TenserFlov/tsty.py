import pandas
import time
import random as rnd
import os
from threading import Thread
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import matplotlib.ticker as ticker



#ГРАФИК
class Graphic():
	def __init__(self):
		
		pass
	
	def Drow(self,x,y,do,hm,titles,xlabels,ylabels,colors,markers):
		#self.fig, ax = plt.subplots()
		self.fig = plt.figure()
		self.fig.set(facecolor = 'white')
		self.fig.set_figwidth(8)
		self.fig.set_figheight(6)
		
		if do=="plot":
			self.fig.suptitle(titles)
			ax=self.fig.add_subplot(111)
			ax.set(xlabel=xlabels,ylabel=ylabels,facecolor = 'black')
			if hm==1:
				ax.plot(x, y, color = colors, linewidth = 1, marker=markers)
				X_MAX=max(x)
				X_MIN=min(x)
				Y_MAX=max(y)
				Y_MIN=min(y)
			else:
				for i in range(len(x)):
					ax.plot(x[i], y[i], color = colors[i], linewidth = 1, marker=markers[i])
				X_MAX=max([max(x[i]) for i in range(len(x))])
				X_MIN=min([min(x[i]) for i in range(len(x))])
				Y_MAX=max([max(y[i]) for i in range(len(y))])
				Y_MIN=min([min(y[i]) for i in range(len(y))])
			self.setting_ax(ax,abs(X_MAX)+abs(X_MIN),abs(Y_MAX)+abs(Y_MIN))
		elif do=="scatter":
			self.fig.suptitle(titles)
			ax=self.fig.add_subplot(111)
			ax.set(xlabel=xlabels,ylabel=ylabels,facecolor = 'black')
			if hm==1:
				ax.scatter(x, y, color = colors, linewidth = 1, marker=markers)
				X_MAX=max(x)
				X_MIN=min(x)
				Y_MAX=max(y)
				Y_MIN=min(y)
			else:
				for i in range(len(x)):
					ax.scatter(x[i], y[i], color = colors[i], linewidth = 1, marker=markers[i])
				X_MAX=max([max(x[i]) for i in range(len(x))])
				X_MIN=min([min(x[i]) for i in range(len(x))])
				Y_MAX=max([max(y[i]) for i in range(len(y))])
				Y_MIN=min([min(y[i]) for i in range(len(y))])
			self.setting_ax(ax,abs(X_MAX)+abs(X_MIN),abs(Y_MAX)+abs(Y_MIN))
		elif do=="subplot":
			ax=self.fig.add_subplot(111)
			self.fig.suptitle(titles[0])
			for i in range(len(titles[1])):
				x[i]
				y[i]
				titles[1][i]
				xlabels[i]
				ylabels[i]
				ax.scatter(x, y, color = 'r', linewidth = 1, marker='o')
				ax.set(title="lllll",xlabel="llll",ylabel="llll",facecolor = 'black')
		
		
		#plt.ion()
		plt.show()
		#self.fig.canvas.draw()
		#plt.show(block=False)
		
		



	def setting_ax(self,ax,X_MAX,Y_MAX):
		#  Устанавливаем интервал основных и
		#  вспомогательных делений:
		ax.xaxis.set_major_locator(ticker.MultipleLocator(X_MAX/10))
		ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
		ax.yaxis.set_major_locator(ticker.MultipleLocator(Y_MAX/10))
		ax.yaxis.set_minor_locator(ticker.MultipleLocator(1))


		#  Настраиваем вид основных тиков:
		ax.tick_params(axis = 'both',    #  Применяем параметры к обеим осям
					   which = 'major',    #  Применяем параметры к основным делениям
					   direction = 'out',    #  Рисуем деления внутри и снаружи графика
					   length = 8,    #  Длинна делений
					   width = 2,     #  Ширина делений
					   color = 'm',    #  Цвет делений
					   pad = 3,    #  Расстояние между черточкой и ее подписью
					   labelsize = 8,    #  Размер подписи
					   labelcolor = 'b',    #  Цвет подписи
					   bottom = True,    #  Рисуем метки снизу
					   top = True,    #   сверху
					   left = True,    #  слева
					   right = True,    #  и справа
					   labelbottom = True,    #  Рисуем подписи снизу
					   labeltop = True,    #  сверху
					   labelleft = True,    #  слева
					   labelright = True,    #  и справа
					   labelrotation = 45)    #  Поворот подписей


		#  Настраиваем вид вспомогательных тиков:
		ax.tick_params(axis = 'both',    #  Применяем параметры к обеим осям
					   which = 'minor',    #  Применяем параметры к вспомогательным делениям
					   direction = 'out',    #  Рисуем деления внутри и снаружи графика
					   length = 5,    #  Длинна делений
					   width = 1,     #  Ширина делений
					   color = 'm',    #  Цвет делений
					   pad = 3,    #  Расстояние между черточкой и ее подписью
					   labelsize = 8,    #  Размер подписи
					   labelcolor = 'b',    #  Цвет подписи
					   bottom = True,    #  Рисуем метки снизу
					   top = True,    #   сверху
					   left = True,    #  слева
					   right = True)    #  и справа
					   
		#  Добавляем линии основной сетки:
		ax.grid(which='major',
				color = 'g',
				linestyle = '-')

		#  Включаем видимость вспомогательных делений:
		ax.minorticks_on()

		#  Теперь можем отдельно задавать внешний вид
		#  вспомогательной сетки:
		ax.grid(which='minor',
				color = 'g',
				linestyle = ':')

#ЗВУК


#---------
#ЗАГРУЗКА ДАННЫХ
def excel_load(name):
	excel=pandas.read_excel(name)
	data=[]
	keys=excel.keys()
	return excel, keys
#---------
#СОХРАНЕНИЕ И ЗАГРУЗКА МОДЕЛИ
#Сохранение модели
def model_dump(MODEL,name):
	import pickle
	pickle.dump(MODEL,open("MODELS\\"+name+".model","wb"))
#Загрузка модели
def models_load():
	import pickle
	MODELS=[]
	for file in os.listdir("MODELS\\"):
		if file.endswith(".model"):
			MODELS.append(pickle.load(open("MODELS\\"+file,"rb")))
	return MODELS
#--------
#МОДЕЛИ
def init_models():
	MODELS=[]
	
	#add model:
	#В MODELS добавляется: имя модели, параметры в виде строки, сама модель
	
	from sklearn.ensemble import RandomForestRegressor as SKL
	MODEL=SKL(n_estimators=200)
	MODELS.append(["RandomForestRegressor","n_estimators=200",MODEL])
	
	return MODELS
	
	

#Предсказание средней квадратичной ошибки модели по данным	
def test_models(MODELS,x_train,y_train,x_test,y_test):
	#Проверка средней абсолютной ошибки
	from sklearn.metrics import mean_absolute_error as mae
	
	DF_models=pandas.DataFrame(columns=['Name_model','Params','Time_learn','Mean_absolute_error_train','Mean_absolute_error_test'])
	
	for name_model, params, model in MODELS:
		print("Start testing [",name_model,"]")
		time_start=time.time()
		model.fit(x_train,y_train)
		time_end=time.time()
		
		c1=rnd.sample(range(0,len(x_train)), 10)
		c2=rnd.sample(range(0,len(x_test)), 10)
		p_train=model.predict(x_train.iloc[c1])
		p_test=model.predict(x_test.iloc[c2])
		r_train=y_train.iloc[c1].values
		r_test=y_test.iloc[c2].values
		
		DF_models.loc[len(DF_models)]=[name_model+".model",params,time_end-time_start,mae(p_train,r_train),mae(p_test,r_test)]
		
		model_dump(model,name_model+".model")
		print("save: ",name_model+".model")
	
	return DF_models

#df['namecolumn'].median()
#df['namecolumn'].mean()
#df['namecolumn'].value_counts() - количество по подгруппам
#Отбор датафрейма по фильтру
#filt=df['country']=='tomsk'
#df.loc[filt]

def Test1():
#ЗАГРУЗКА ДАННЫХ
	data, keys=excel_load("data.xlsx")
	data=data[keys[1]].values
	
#ПРЕОБРАЗОАНИЕ ДАННЫХ
	future=7 #прогноз на количество дней
	past=future*4 #период для прогноза (обучение)
	#Составление обучающей таблицы из примеров
	start=past
	end=len(data)-past
	training_past=[]
	training_future=[]
	for day in range(start,end):
		past_data=data[day-past:day]
		futu_data=data[day:day+future]
		training_past.append(past_data)
		training_future.append(futu_data)
	
	columns_past=[]
	columns_future=[]
	for i in range(past):
		columns_past.append(f"p_{i}")
	for i in range(future):
		columns_future.append(f"f_{i}")
	#print(columns_past)
	#print(columns_future)
	
	DF_past=pandas.DataFrame(training_past,columns=columns_past)
	DF_future=pandas.DataFrame(training_future,columns=columns_future)
	print(DF_past)
	print(DF_future)
	
#РАЗДЕЛЕНИЕ НА ОБУЧАЮЩУЮ И ТЕСТОВУЮ ВЫБОРКИ
	#Разделение на обучающую (train), проверочную (test) выборки
	x_train=DF_past[:-10]
	y_train=DF_future[:-10]
	
	x_test=DF_past[-10:]
	y_test=DF_future[-10:]
	
	from sklearn.model_selection import train_test_split
	x_train, x_test, y_train, y_test = train_test_split(DF_past, DF_future, test_size=0.3) # 70% training and 30% tes
	
	c1=rnd.sample(range(0,len(x_train)), 10)
	c2=rnd.sample(range(0,len(x_test)), 10)
#ОБУЧЕНИЕ МОДЕЛИ
	models=init_models()
	DF_models=test_models(models,x_train,y_train,x_test,y_test)
	
def Test2():
	
	
	models=models_load()
	print(len(models))



import numpy as np
from scipy import *
import soundfile as sf
#READ AUDIO FILE
def read_bin_file(filename):
	f=open(filename, 'rb')
	values = np.fromfile(f,dtype="int8")
	return values
def read_wav_file(filename):
	data, samplerate = sf.read(filename)
	return samplerate, data

#SPECTROGRAM
from scipy.fftpack import fft, rfft, fftfreq
from scipy import signal
def plotSpectrogram(Fs1,x):
	NFFT = 1024  # the length of the windowing segments
	Fs = Fs1	# the sampling frequency

	fig, ax = plt.subplots(nrows=4, ncols=1, facecolor='#EDFAEA', figsize=(10, 8))
	
	#Сигнал
	ax[0].plot(range(len(x)), x, linewidth = 1)
	ax[0].set_ylabel([0,8000])
	#Спектр сигнала
	#window
	#p_window = np.hamming(1024)
	#nperseg - длина сегмента окна (256 - по умолчанию)
	#noverlap - число точек в перекрытии
	#nfft - длина БПФ (256 - по умолчанию)
	Pxx, bins, freqs, im = ax[1].specgram(x, NFFT=NFFT, Fs=Fs, noverlap=500,aspect='auto')
#,nperseg=1000)
	# The `specgram` method returns 4 objects. They are:
	# - Pxx: the periodogram
	# - freqs: the frequency vector
	# - bins: the centers of the time bins
	# - im: the matplotlib.image.AxesImage instance representing the data in the plot

	#plt.savefig('спектрограмма.png', facecolor=facecolor)

	plt.xlabel('Freq (Hz)')
	plt.ylabel('|Y(freq)|')
	
	#ax.set_xticks()
	#ax.set_xticklabels()
	#plt.sca(axes[1, 1])
	#plt.xticks(range(3), ['A', 'Big', 'Cat'])
	#Взятые элементы от спектра
	#ax3
	
	#,origin='lower'
	#upper
	N=4000
	Fs=2048
	data=x[int(N-Fs/2):int(N+Fs/2)]
	
	
	data=x
	w = blackman(len(data))
	Y1 = 20*log10(abs(fft(data,Fs))) # fft computing and normalization
	Y2 = 20*log10(abs(fft(w*data,Fs))) # fft computing and normalization
	fft1 = Y1[range(int(len(Y1)/2))]
	fft2 = Y2[range(int(len(Y2)/2))]
	ax[2].plot(range(len(fft1)),abs(fft1),'r',linewidth=1) # plotting the spectrum
	ax[3].plot(range(len(fft2)),abs(fft2),'r',linewidth=1) # plotting the spectrum
	ax[3].set_xlabel([1024,1024+len(fft2)])
	plt.xlabel('Freq (Hz)')
	plt.ylabel('|Y(freq)|')
	plt.show()


SAMPLE_RATE = 16000 # Hz
WINDOW_SIZE = 2048 # размер окна, в котором делается fft
WINDOW_STEP = 512 # шаг окна
WINDOW_OVERLAP = WINDOW_SIZE - WINDOW_STEP	#перекрытие окна

def show_specgram(wave_data):
	SAMPLE_RATE = 16000 # Hz
	WINDOW_SIZE = 2048 # размер окна, в котором делается fft
	WINDOW_STEP = 512 # шаг окна
	
	fig = plt.figure()
	ax = fig.add_axes((0.1, 0.1, 0.8, 0.8))
	ax.specgram(wave_data,
		NFFT=WINDOW_SIZE, noverlap=WINDOW_OVERLAP, Fs=SAMPLE_RATE)
	plt.show()

#Задачу поиска фрагмента в эфире можно разбить на две части: сначала найти среди большого числа эталонных 
#фрагментов кандидаты, а затем проверить, действительно ли кандидат звучит в данном фрагменте эфира, и если да,
# то в какой момент начинается и заканчивается звучание. Обе операции используют для своей работы «отпечаток» 
#фрагмента звучания. Он должен быть устойчивым к шумам и быть достаточно компактным. Этот отпечаток строится так: 
#мы разбиваем спектрограмму на короткие отрезки по времени, и в каждом таком отрезке ищем частоту с максимальной 
#амплитудой (на самом деле лучше искать несколько максимумов в различных диапазонах, но для простоты возьмем один 
#максимум в наиболее содержательном диапазоне). Набор таких частот (или индексов частот) и представляет собой отпечаток. 
#Очень грубо можно сказать, что это «ноты», звучащие в каждый момент времени.
def get_fingerprint(wave_data):
	# pxx[freq_idx][t] - мощность сигнала
	pxx, _, _ = mlab.specgram(wave_data,
		NFFT=WINDOW_SIZE, noverlap=WINDOW_OVERLAP, Fs=SAMPLE_RATE)
	band = pxx[15:250]  # наиболее интересные частоты от 60 до 1000 Hz
	return np.argmax(band.transpose(), 1)  # max в каждый момент времени




import queue
from matplotlib.animation import FuncAnimation

q = queue.Queue()
def ddt():
	import sounddevice as sd

	duration = 3  # seconds
	#with sd.Stream(callback=callback_f):
	#	sd.sleep(duration * 1000)
	
	device=1
	device_info = sd.query_devices(device, 'input')
	print(device_info)
	
	window=1024
	samplerate=16000
	global downsample
	downsample=10
	channels=[1]
	interval=100
	length = int(window * samplerate / (1000 * downsample))
	global plotdata
	plotdata = np.zeros((length, len(channels)))
	
	
	fig, ax = plt.subplots()
	#plt.fill_between(x, y1, y2)#Заливка между линиями
	global lines
	lines = ax.plot(plotdata)
	ax.axis((0, len(plotdata), -1, 1))
	ax.set_yticks([0])
	ax.yaxis.grid(True)
	ax.tick_params(bottom='off', top='off', labelbottom='off',
				   right='off', left='off', labelleft='off')
	fig.tight_layout(pad=0)

	stream = sd.InputStream(device=device, channels=max(channels),
		samplerate=samplerate, callback=audio_callback)
	ani = FuncAnimation(fig, update_plot, interval=interval, blit=True)
	with stream:
		plt.show()


def audio_callback(indata, outdata, frames, time):
	global downsample
	#volume_norm = np.linalg.norm(indata)*10
	#axx.plot(volume_norm)
	#fig1.canvas.draw()
	#plt.show()
	#outdata[:] = indata
	q.put(indata[::downsample])


def update_plot(frame):
	global plotdata
	while True:
		try:
			data = q.get_nowait()
		except queue.Empty:
			break
		shift = len(data)
		plotdata = np.roll(plotdata, -shift, axis=0)
		plotdata[-shift:, :] = data
	for column, line in enumerate(lines):
		line.set_ydata(plotdata[:, column])
	return lines	

if __name__ == "__main__":
	x=[i for i in range(20)]
	y=[rnd.randint(-40,40) for i in range(20)]
	g=Graphic()
	#g.Drow(x,y,"plot",1,"T1","X","Y",'b','.')
	#g.Drow(x,y,"scatter",1,"T1","X","Y",'b','.')
	
	X=[]
	Y=[]
	X.append([i for i in range(20)])
	Y.append([rnd.randint(-40,40) for i in range(20)])
	X.append([i for i in range(20)])
	Y.append([rnd.randint(-40,40) for i in range(20)])
	X.append([i for i in range(20)])
	Y.append([rnd.randint(-40,40) for i in range(20)])
	X.append([i for i in range(20)])
	Y.append([rnd.randint(-40,40) for i in range(20)])
	x=[i for i in range(20)]
	y=[rnd.randint(-40,40) for i in range(20)]
	#g.Drow(X,Y,"plot",2,"T1","X","Y",['b','r'],['.','.'])
	#g.Drow(X,Y,"scatter",2,"T1","X","Y",['b','r'],['.','.'])
	#g.Drow(X,Y,"subplot",4,[["T1"],["t1","t2","t3","t4"]],["X","X","X","X"],["Y","Y","Y","Y"],['#012','#123','#234','#345'],['.','.','.','.'])
	
	
	
	import soundfile
	data, samplerate = soundfile.read('./wav/3.wav')
	soundfile.write('./wav/3-16.wav', data, samplerate, subtype='PCM_16')
	
	from scipy.io import wavfile
	samplingFrequency, signalData = wavfile.read('./wav/3-16.wav')
	print(samplingFrequency)
	
	
	#show_specgram(signalData)
	#print(get_fingerprint(signalData))
	
	ddt()
	
	time.sleep(3)
	#plotSpectrogram(samplingFrequency,signalData)
	