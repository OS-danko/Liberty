import sounddevice as sd
import numpy as np
import matplotlib.pyplot as plt

duration = 1 #sec
fs = 44100

def record():
	"""records from the mic"""
	recording = sd.rec(int(duration * fs), samplerate = fs, channels =1, 
	dtype='float64')
	#waits till ur finished recording
	sd.wait(duration)
	return recording

def play(recording):
	"""plays recording"""
	sd.play(recording, fs)
	sd.wait(duration)

from scipy import signal
def plot_signal_freq(ys):
	N = ys.shape[0]
	#print(N,'=',ys.shape)
	L = N/fs
	tuckey_window=signal.tukey(N,0.01,True) #generate the Tuckey window, widely open, alpha=0.01
	ysc=ys[:,0]*tuckey_window               #applying the Tuckey window
	yk = np.fft.rfft(ysc)                   #real to complex DFT
	k = np.arange(yk.shape[0])
	freqs = k/L
	fig, ax = plt.subplots()
	#print(len(freqs))
	#print(len(np.abs(yk)))
	ax.plot(freqs, np.abs(yk))
	plt.show()


#while True:
	#recording = record()
	#print(type(recording.dtype))
	#print(recording)
	#play(recording)
	#plot_signal_freq(recording)

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
dat = np.random.random(200).reshape(20,10) # создаём матрицу значений
def colormap_choice():
	# Создаём список цветовых палитр из словаря
	maps = [m for m in plt.cm.datad]
	from math import sqrt

	fig, axes= plt.subplots(nrows=int(sqrt(len(maps))), ncols=int(sqrt(len(maps))), sharex=True, sharey=True)
	iter=0
	for ax in fig.axes:
		#random_cmap = np.choice(maps[iter])
		ccmp=maps[iter]
		cf = ax.contourf(dat, cmap=plt.get_cmap(ccmp))
		ax.set_title('%s colormap' % ccmp)
		fig.colorbar(cf, ax=ax)
		iter+=1
	   
	plt.suptitle(u'Различные цветовые палитры')   # единый заголовок рисунка
	plt.show()
def subplot_imshow():
	#fig = plt.figure()
	fig, axes= plt.subplots(nrows=1, ncols=3, sharex=True, sharey=True)
	me = axes[0].imshow(dat) 
	#plt.colorbar(me)
	plt.title('Simple imshow plot')

	cr = axes[1].contour(dat) 
	#plt.colorbar(cr)
	plt.title('Simple contour plot')

	cf = axes[2].contourf(dat) 
	plt.colorbar(cf)
	plt.title('Simple contourf plot')
	plt.show()

	s = ['one','two','three ','four' ,'five']
	x = [1, 2, 3, 4, 5]
	# pie()
	fig = plt.figure()
	plt.pie(x, labels=s)
	plt.title('Simple pie chart')
	plt.show()


subplot_imshow()