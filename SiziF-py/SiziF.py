from tkinter import *						#Грфический интерфейс
import tkinter as tk
import time									#Время
from PIL import Image, ImageTk				#Загрузка изображений

from ListboxToolTip	import CreateToolTip	#Доп информация при наведении
from Bomber.main import Bomber as C_Bomber
from Info.main import Info as C_Info
from URL_DLoader.main import URL_DLoader as C_URL_DLoader
from LightShot.main import LightShot as C_LightShot

#Класс меню и главного цикла программы
class Main():#Frame
	def __init__(self, root):
		#super().__init__(root)
		self.MyWidth=300
		self.MyHeight=120
		self.MyLoc_x=100
		self.MyLoc_y=100
		
		self.root=root
		
		self.bg_image=None
		
		self.main()

	def main(self):
		#self.root.delete(0, 'end')
		list = self.root.pack_slaves()
		for l in list:
			l.destroy()

		self.root.geometry('{}x{}+{}+{}'.format(self.MyWidth, 
												self.MyHeight, 
												self.MyLoc_x, 
												self.MyLoc_y))
		self.root.title('SiziF')
		self.root.resizable(True, True)
		self.root.iconbitmap("main.ico")
		self.bg_image=Image.open('bg.png')
		
		#TOOLBAR
		self.toolbar1 = Frame(self.root,bg='#000')
		self.toolbar1.pack(side=TOP, fill=X,pady=1)
		
		self.toolbar2 = Frame(self.root,bg='#000')
		self.toolbar2.pack(side=TOP, fill="both",expand="yes")

		# self.menu1 = Menu(self.root)
		# self.root.config(menu=self.menu1)
		
		
		# self.new_item = Menu(self.menu1 , tearoff=0)
		# self.menu1.add_cascade(label='Главная', menu=self.new_item)
		# self.new_item.add_command(label='Файл', command=self.main)
		# self.new_item.add_separator()
		# self.new_item.add_command(label='СМС-Бомбер', command=self.Bomber)
		# self.new_item.add_command(label='DoS', command=self.DoS)
		
		#BACKGROUND IMAGE toolbar2
		img=ImageTk.PhotoImage(self.bg_image.resize((self.root.winfo_width(), self.root.winfo_height())))
		self.L_backgroun = Label(self.toolbar2,image=img)
		#self.L_backgroun.image=None
		self.L_backgroun.image=img
		self.L_backgroun.pack(side="bottom", fill = "both", expand = "yes")
		
		#Buttons toolbar1
		img_o=Image.open('main.ico')
		img=ImageTk.PhotoImage(img_o.resize((30,30)))
		self.B_jarviz=Button(self.toolbar1, command=self.Cclose, image=img,background="#bbb",activebackground="#555")
		self.B_jarviz.image=img
		self.B_jarviz.pack(side="left")
		CreateToolTip(self.B_jarviz, "Закрыть")
		
		
		img_o=Image.open('reload.png')
		img=ImageTk.PhotoImage(img_o.resize((30,30)))
		self.B_reload=Button(self.toolbar1, command=self.bg_resizer, image=img,background="#bbb",activebackground="#555")
		self.B_reload.image=img
		self.B_reload.pack(side="right")
		CreateToolTip(self.B_reload, "Обновить форму")
		
		#Buttons toolbar2
		self.CreateButtons(60)
		
		
		self.root.mainloop()
		
	def Cclose(self):
		self.root.quit()

	def CreateButtons(self,size_icons):
		functions=['Bomber',		#0
					'Info',			#1
					'URL_DLoader',	#2
					'LightShot']	#3
		
		self.ButtonsMenu=[]
		B_iter=0
		self.Fields=[]
		F_iter=0
		otstup=5
		lenFrame=0
		#Поле
		self.Fields.append(Frame(self.L_backgroun))
		self.Fields[F_iter].pack(side='top')
		
		counter=1
		for name in functions:
			lenFrame=(counter)*(size_icons+2*otstup)
			if lenFrame>self.root.winfo_width():
				F_iter+=1
				counter=1
				#Поле
				self.Fields.append(Frame(self.L_backgroun))
				self.Fields[F_iter].pack(side='top')
			
			#Кнопка на поле
			img_o=Image.open(name+'\\icon.png')
			img=ImageTk.PhotoImage(img_o.resize((size_icons,size_icons)))
			self.ButtonsMenu.append(Button(self.Fields[F_iter], image=img,background="#555",activebackground="#bbb"))
			self.ButtonsMenu[B_iter].image=img
			self.ButtonsMenu[B_iter].pack(side='left')#,ipadx=otstup, ipady=otstup)
			#Контекст кнопки
			CreateToolTip(self.ButtonsMenu[B_iter], functions[B_iter])
			B_iter+=1
			counter+=1
		
		self.ButtonsMenu[0].bind('<ButtonRelease-1>', lambda event: C_Bomber())
		self.ButtonsMenu[1].bind('<ButtonRelease-1>', lambda event: C_Info())
		self.ButtonsMenu[2].bind('<ButtonRelease-1>', lambda event: C_URL_DLoader())
		self.ButtonsMenu[3].bind('<ButtonRelease-1>', lambda event: C_LightShot())



	def bg_resizer(self):
		list = self.L_backgroun.pack_slaves()
		for l in list:
			l.destroy()
		#Изменить размер background
		new_width = self.root.winfo_width()
		new_height = self.root.winfo_height()
		img=ImageTk.PhotoImage(self.bg_image.resize((new_width, new_height)))
		self.L_backgroun.configure(image=img)
		self.L_backgroun.image=img
		
		#Изменить положение кнопок
		self.CreateButtons(60)
		
		
		self.root.update()



if __name__ == "__main__":
	root = Tk()
	app = Main(root)
	#app.pack()
	#root.mainloop()
