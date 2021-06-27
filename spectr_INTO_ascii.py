import numpy
import pyaudio
import re
import sys

from colorama import init
init()

WIDTH = 90
BOOST = 2.0


# Create a nice output gradient using ANSI escape sequences.
cols = [30, 34, 35, 91, 93, 97]
chars = [(' ', False), (':', False), ('%', False), ('#', False),
         ('#', True), ('%', True), (':', True)]
gradient = []
for bg, fg in zip(cols, cols[1:]):
	for char, invert in chars:
		if invert:
			bg, fg = fg, bg
		gradient.append('\x1b[{};{}m{}'.format(fg, bg + 10, char))


class Spectrogram(object):
	def __init__(self):
		self.audio = pyaudio.PyAudio()
		self.min=0
		self.max=0
	
	def __enter__(self):
		"""Open the microphone stream."""
		device_index = 1
		rate = 16000

		self.buffer_size = int(rate * 0.05)
		self.stream = self.audio.open(format=pyaudio.paInt16,
									  channels=1, rate=rate, input=True,
									  input_device_index=1,
									  frames_per_buffer=self.buffer_size)
		return self

	def __exit__(self, *ignored):
		"""Close the microphone stream."""
		self.stream.close()


	def color(self, x):
		"""
		Given 0 <= x <= 1 (input is clamped), return a string of ANSI
		escape sequences representing a gradient color.
		"""
		if self.min>x:
			self.min=x
		if self.max<x:
			self.max=x
		x = max(0.0, min(0.07, x))
		return gradient[int(x/(self.max-self.min) * (len(gradient) - 1))]

	def listen(self):
		"""Listen for one buffer of audio and print a gradient."""
		block_string = self.stream.read(self.buffer_size)
		block = numpy.fromstring(block_string, dtype='h') / 32768.0
		nbands = 30 * WIDTH
		fft = abs(numpy.fft.fft(block, n=nbands))

		pos, neg = numpy.split(fft, 2)
		bands = (pos + neg[::-1]) / float(nbands) * BOOST
		line = (self.color(x) for x in bands[:WIDTH])
		print(''.join(line) + '\x1b[0m')
		sys.stdout.flush()

if __name__ == '__main__':
	with Spectrogram() as s:
		while True:
			s.listen()