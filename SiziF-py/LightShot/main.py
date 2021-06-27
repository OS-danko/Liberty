from tkinter import *			#Грфический интерфейс
import tkinter as tk
from PIL import Image, ImageTk				#Загрузка изображений

from ListboxToolTip	import CreateToolTip	#Доп информация при наведении

from requests import get
from random import choice
import threading
from bs4 import BeautifulSoup
from strgen import StringGenerator
import os
import time

class LightShot(tk.Toplevel):
	def __init__(self):
		super().__init__()
		#self.root=Tk()
		self.name_folder='LightShot\\'
		self.MyWidth=600
		self.MyHeight=300
		self.MyLoc_x=100
		self.MyLoc_y=100
		self.geometry('{}x{}+{}+{}'.format(self.MyWidth, 
												self.MyHeight, 
												self.MyLoc_x, 
												self.MyLoc_y))
		self.title('LightShot')
		self.resizable(True, True)
		self.iconbitmap(self.name_folder+'icon.ico')
		
		#BACKGROUND IMAGE
		bg_image=Image.open(self.name_folder+'bg.png')
		img=ImageTk.PhotoImage(bg_image.resize((self.winfo_width(), self.winfo_height())))
		L_backgroun = Label(self,image=img)
		#self.L_backgroun.image=None
		L_backgroun.image=img
		L_backgroun.pack(side="bottom", fill = "both", expand = "yes")
		
		#TOOLBAR
		self.toolbar1 = Frame(L_backgroun,bg='#000')
		self.toolbar1.pack(side=TOP, fill=X,pady=1)
		self.toolbar2 = Frame(L_backgroun,bg='#000')
		self.toolbar2.pack(side=TOP, fill=X,pady=1)
		
		#LABELS
		L1=Label(self.toolbar1, font='Times 12', fg='#555', bg="#bbb")
		L1.pack(side="left", pady=2)
		L1.config(text='https://prnt.sc/')
		
		#EDITS
		self.E1= Entry(self.toolbar1,font='Times 12', fg='#555', bg="#bbb", width=7)
		self.E1.pack(side=LEFT, pady=2, padx=2)
		self.E1.delete(0, END)
		self.E1.insert(0, StringGenerator('[\h]{6}').render().lower())
		
		#BUTTONS
		self.B_jarviz=Button(self.toolbar1, command=self.do, text="Начать загрузку",background="#bbb",activebackground="#555")
		self.B_jarviz.pack(side="left", pady=2, padx=10)
		CreateToolTip(self.B_jarviz, "Начало многопоточной загрузки\nизображений с сервиса ligthshot (10 потоков)")
		
		#TEXTS
		#Scroll
		self.S1 = Scrollbar(self.toolbar2)#, orient=HORIZONTAL)
		self.S1.pack(side=LEFT, fill=Y)
		# Text Widget
		self.T1 = Text(self.toolbar2, wrap=NONE, yscrollcommand=self.S1.set)#, xscrollcommand=self.S1.set)
		self.T1.pack(side=LEFT)
		# Configure the scrollbars
		self.S1.config(command=self.T1.yview)#self.T1.xview
		
		if not os.path.exists('images'):
			os.mkdir('images')
		
		self.user_agents   = {'User-agent': 'Mozilla/5.0 (X11; Linux x86_64; rv:70.0) Gecko/20100101 Firefox/70.0'}
		self.do=0

	def do(self):
		threads_count = 10
		
		if self.do==0:
			self.do=1
			for i in range(1, threads_count):
				name='thread'+str(i)
				thread = threading.Thread(target = self.scan, args=(name,))
				thread.start()
		elif self.do==1:
			self.do=0
		


	def scan(self,name):
		while self.do==1:
			time.sleep(3)
			# Random url
			random = StringGenerator('[\h]{6}').render().lower()
			url    = 'https://prnt.sc/' + random
			# Make a GET request to get HTML
			content = get(url, timeout = 3, headers = self.user_agents).text
			# Parse HTML to get page title
			soup  = BeautifulSoup(content, 'html.parser')
			# Check if Cloudflare block request
			if 'Cloudflare' in soup.title.get_text().split():
				self.addT1('[-]<'+name+'> Cloudflare blocked request!')
				self.do=0
				break
			# Try to download image
			else:
				try:
					image = soup.img['src']
				except TypeError:
					continue
				else:
					if image.startswith('http'):
						self.save(image,name)
			if self.do==0:
				break

	def addT1(self,data):
		text=self.T1.get("1.0",END)
		self.T1.delete("1.0", END)
		self.T1.insert("1.0", data+text)

	def save(self,url,name):
		file = url.split('/')[-1]
		try:
			data = get(url, allow_redirects = True, headers = self.user_agents)
		except:
			pass
		else:
			path = 'images/' + file
			open(path, 'wb').write(data.content)
			self.addT1('[+]<'+name+'>'+file)


