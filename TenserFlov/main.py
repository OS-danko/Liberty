#Грфический интерфейс
import tkinter as tk
from tkinter import Tk, ttk
#Загрузка изображений
from PIL import Image, ImageTk

#Доп информация при наведении
from ListboxToolTip	import CreateToolTip

from threading import Thread

from queue import Queue
queue=Queue()
task=444
queue.put(task)
task=queue.get()

import time
def GetTime():
	return str(time.strftime("%Y-%m-%d-%H.%M.%S", time.localtime()))



class Main():#Frame
	def __init__(self, root):
		super(Main, self).__init__()
		self.MyWidth=700
		self.MyHeight=600
		self.MyLoc_x=100
		self.MyLoc_y=100
		
		
		
		self.root=root
		
		self.bg_image=None
		
		self.main()
	
	def main(self):
		self.root.geometry('{}x{}+{}+{}'.format(self.MyWidth, 
												self.MyHeight, 
												self.MyLoc_x, 
												self.MyLoc_y))
		self.root.title('SiziF')
		self.root.resizable(True, True)
		self.root.iconbitmap("main.ico")
		
		#BACKGROUND IMAGE
		bg_image=Image.open('bg.png')
		img=ImageTk.PhotoImage(bg_image.resize((self.winfo_width(), self.winfo_height())))
		L_backgroun = Label(self,image=img)
		L_backgroun.image=img
		L_backgroun.pack(side="bottom", fill = "both", expand = "yes")
		
		#TOOLBAR
		self.toolbar1 = tk.Frame(self.root,bg='#000')
		self.toolbar1.pack(side="top", fill=tk.X,pady=1)
		
		self.toolbar2 = tk.Frame(self.root,bg='#f00')
		self.toolbar2.pack(side="top", fill="both",expand="yes")
		
		
		
		#toolbar1
		
		#Buttons 
		img_o=Image.open('main.ico')
		img=ImageTk.PhotoImage(img_o.resize((15,15)))
		self.B_jarviz=tk.Button(self.toolbar1, command=self.Cclose, image=img,background="#bbb",activebackground="#555")
		self.B_jarviz.image=img
		self.B_jarviz.pack(side="left")
		CreateToolTip(self.B_jarviz, "Закрыть")
		
		self.B_Close=tk.Button(self.toolbar1, command=self.Cclose, text="X",font='Times 10', fg="#000", background="#bbb",activebackground="#555", padx=2, pady=0)
		self.B_Close.pack(side="right")
		CreateToolTip(self.B_Close, "Закрыть")
		
		#LABELS
		L1=Label(self.toolbar1,text="Phone (без +7):", font='Times 12', fg='#FFF', bg="#111")
		L1.pack(side="left")#, padx=2)
		
		#EDITS
		self.E1_text = StringVar()
		E1= Entry(self.toolbar5,width=15, textvariable=self.E1_text)
		E1.pack(side="left")
		
		t1=self.E1_text.get()
		
	def Cclose(self):
		self.root.quit()


	def test(self):
		import numpy as np
		from sklearn import linear_model
		from sklearn import svm

		classifiers = [
			svm.SVR(),
			linear_model.SGDRegressor(),
			linear_model.BayesianRidge(),
			linear_model.LassoLars(),
			linear_model.ARDRegression(),
			linear_model.PassiveAggressiveRegressor(),
			linear_model.TheilSenRegressor(),
			linear_model.LinearRegression()]

		trainingData    = np.array([ [2.3, 4.3, 2.5],  [1.3, 5.2, 5.2],  [3.3, 2.9, 0.8],  [3.1, 4.3, 4.0]  ])
		trainingScores  = np.array( [3.4, 7.5, 4.5, 1.6] )
		predictionData  = np.array([ [2.5, 2.4, 2.7],  [2.7, 3.2, 1.2] ])

		for item in classifiers:
			print(item)
			clf = item
			clf.fit(trainingData, trainingScores)
			print(clf.predict(predictionData),'\n')

	def code(self):
		import pandas
		
		#LOAD
		
		#excel
		data_excel=pandas.read_excel("data.xlsx")
		print(data_excel.describe())
		
		data_excel_change=pandas.get_dummies(data_excel,columns=[data_excel.keys()[2]])
		print(data_excel_change)
		
		print("+"*9)
		#csv
		# fixed_df = pandas.read_csv('data_anime_0.csv')
		# print(fixed_df.describe())
		# print(fixed_df.keys())
		# #print(fixed_df)
		# print(fixed_df[:3])
		# print(fixed_df[fixed_df.keys()[2]][:3])
		# print(fixed_df.head())
		
		print("-"*9+"YX")
		#то что мы хотим предсказывать
		#Y=data_excel.y#['y']
		Y = data_excel.iloc[:, 2].values
		print("Y:")
		print(Y)
		#То на основе чего предсказываем
		#X=data_excel.drop('y',axis=1)	#Берем все кроме одной колонки
		X = data_excel.iloc[:, 0:2].values
		print("X:")
		print(X)
		#выбор модели
		from sklearn.linear_model import LinearRegression	#подключение модели - линейной регрессии
		from sklearn.ensemble import RandomForestClassifier	#подключение модели - лес случайных деревьев
		model=RandomForestClassifier()	#Можно указать параметры
		
		clf=RandomForestClassifier(n_estimators=100)
		from sklearn.model_selection import train_test_split
		X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3) # 70% training and 30% tes
		# print("X"+"-"*9)
		# print(X_train)
		# print("Y"+"-"*9)
		# print(y_train)
		clf.fit(X_train.astype('int'),y_train.astype('int'))
		y_pred=clf.predict(X_test)
		print(X_test)
		print(y_pred)
		
		
		
		model.fit(X,Y)	#подогнать на основе данных модель
		mydata=[[1.79, 1.9]]
		ans=model.predict(mydata)
		print(ans)
		

import os
import matplotlib.pyplot as plt
import numpy as np
def save_plot(name, fmt='png'):
	pwd = os.getcwd()
	iPath = './pictures/'
	if not os.path.exists(iPath):
		os.mkdir(iPath)
	os.chdir(iPath)
	plt.savefig('{}.{}'.format(name, fmt), fmt='png')
	os.chdir(pwd)

def plot_colormap1():
	N = 100
	x = np.arange(N)
	# Задаём выборку из Гамма-распредления с параметрами формы=1. и масштаба=0.
	z = np.random.gamma(2., 1., N)
	y = z.reshape(10,10)

	cc = plt.contourf(y) 
	cbar = plt.colorbar(cc)

def plot_ddd1():
	N = 100
	x = np.arange(N)
	# Задаём выборку из Гамма-распредления с параметрами формы=1. и масштаба=0.
	z = np.random.gamma(2., 1., N)
	my_dict = {'color' : 'grey', 'linewidth' : 1.5, 'linestyle' : '--'}
	xz = [x, z]
	cc = plt.plot(*xz, **my_dict)

def plot_ddd2():
	N = 100
	x = np.arange(N)
	# Задаём выборку из Гамма-распредления с параметрами формы=1. и масштаба=0.
	z = np.random.gamma(2., 1., N)
	z1 = np.cos(x/10.)
	z2 = np.cos(x/20.)
	# создание словаря
	my_dict = {'color' : 'green', 'linewidth' : 4.0, 'alpha' : 0.5} 

	plt.fill_between(x, z2, z1, color='green', alpha=0.25) 
	plt.scatter(x, z1, color='green', linewidth=0.5)
	plt.plot(x, z2, **my_dict)

# Диаграммы:
# plt.bar(), plt.barh(), plt.barbs(), broken_barh() - столбчатая диаграмма;
# plt.hist(), plt.hist2d(), plt.hlines - гистограмма;
# plt.pie() - круговая диаграмма;
# plt.boxplot() - "ящик с усами" (boxwhisker);
# plt.errorbar() - оценка погрешности, "усы".

# Изображения в изолиниях:
# plt.contour() - изолинии;
# plt.contourf() - изолинии с послойной окраской;

# IV. Отображения:
# plt.pcolor(), plt.pcolormesh() - псевдоцветное изображение матрицы (2D массива);
# plt.imshow() - вставка графики (пиксели + сглаживание);
# plt.matshow() - отображение данных в виде квадратов.

# V. Заливка:
# plt.fill() - заливка многоугольника;
# plt.fill_between(), plt.fill_betweenx() - заливка между двумя линиями;

# Векторные диаграммы:
# plt.streamplot() - линии тока;
# plt.quiver() - векторное поле.
def plot_D(data):
	x = np.arange(len(data))
	my_dict = {'color' : 'grey', 'linewidth' : 1.5, 'linestyle' : '--'}
	xz = [x, data]
	cc = plt.plot(*xz, **my_dict)

def plot_D2(data1,data2):
	x1 = np.arange(len(data1))
	x2 = np.arange(len(data2))
	my_dict1 = {'color' : 'grey', 'linewidth' : 1.5, 'linestyle' : '--'}
	my_dict2 = {'color' : 'red', 'linewidth' : 1.5, 'linestyle' : '--'}
	xz1 = [x1, data1]
	xz2 = [x2, data2]
	plt.plot(*xz1, **my_dict1)
	plt.plot(*xz2, **my_dict2)
	plt.show()
#https://habr.com/ru/post/468295/
#https://tproger.ru/translations/python-data-vizualization/
#https://matplotlib.org/gallery.html




if __name__ == "__main__":
	
	root = Tk()
	app = Main(root)
	
	

	
	
	data=data_excel.iloc[:, 1].values
	#plot_D(data)
	#print(data)
	save_plot('sex1')
	
	#plt.show()
	
	
		
	elif do==1:
		#Модель линейной регрессии
		from sklearn.linear_model import LinearRegression
		LR=LinearRegression(normalize=True)
		 do==2:
		from sklearn.neural_network import MLPRegressor #регрессор на нейронныйх сетях
		mlp=MLPRegressor(max_iter=1000,hidden_layer_sizes=(100,100,),random_state=True)
		
	#средняя абсолютная ошибка
	#from sklearn.metrics import mean_absolute_error
	#cou=3
	#prediction=MODEL.predict([x_test.iloc[cou]])[0]
	#real=y_test.iloc[cou]
	#print(mean_absolute_error(prediction,real))
	#plot_D2(prediction,real)
	
	
	
	#GridSearchCV - поиск по сетке параметров
	#Например хотим применить комбинацию параметров к модели и понять при какой комбинации она работает лучше
	#CV - cross validation (кросс-валидация)
	#воляет обучить модель несколько раз на разных разбиениях выборки
	#* - обучающая выборка
	#X - тестовая выборка
	#[*****************XXX] - взяли с конца значения для теста
	#[**************XXXXXX] - взяли 30%
	#по хорошему перед этим лучше еще и перемешать - тоесть взять из разных мест эти 30%
	#Кросс-валидация - обучаем модель на данных (при разбиении данных случайным образом(исключая повторения)) несколько раз
	#(она предназначен чтобы исключить переобучение)
	#[XXXXX***************]
	#[*****XXXXX**********]
	#[**********XXXXX*****]
	#[***************XXXXX]
	#scrolling = neg_mean_absolute_error - средняя абсолютная ошибка
	from sklearn.model_selection import GridSearchCV
	from sklearn.neighbors import KNeighborsRegressor
	KNR=KNeighborsRegressor(p=2)
	param_grid={
		"n_neighbors": [1,2,3,4,8],				#количество соседей
		"weights": ['uniform', 'distance']	#веса
	}
	gs=GridSearchCV(KNR, param_grid,"neg_mean_absolute_error", cv=4)
	gs.fit(x_train,y_train)
	print(gs.best_params_)#показать лучшие параметры
	print(gs.best_score_)#показать какая получилась лучшая средняя ошибка
	best_model=gs.best_estimator_ #самя лучшая обученная модель
	#средняя абсолютная ошибка
	from sklearn.metrics import mean_absolute_error
	cou=3
	prediction=best_model.predict([x_test.iloc[cou]])[0]
	real=y_test.iloc[cou]
	print(mean_absolute_error(prediction,real))
	plot_D2(prediction,real)
	
	
	
	
	
	
