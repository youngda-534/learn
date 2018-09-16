import math
import random
import numpy as np
import pandas as pd

random.seed(0)

def sigmoid(x):
	return 1.0 / (1.0 + math.exp(-x))

def sigmoid_derivative(x):
	return x * (1 - x)

def rand(a,b):
	return (b - a) * random.random() + a

def make_matrix(m, n, fill=0.0):
	mat = []
	for i in range(m):
		mat.append([fill] * n)
	return mat

class BPNeuralNetwork:
	def __init__(self):
		self.input_n = 0
		self.hidden_n = 0
		self.output_n = 0
		self.input_cells = []
		self.hidden_cells = []
		self.output_cells = []
		self.input_weights = []
		self.output_weights = []
			
	def setup(self, ni, nh, no):
		self.input_n = ni + 1
		self.hidden_n = nh
		self.output_n = no
		self.input_cells = [1.0] * self.input_n
		self.hidden_cells = [1.0] * self.hidden_n
		self.output_cells = [1.0] * self.output_n
		self.input_weights = make_matrix(self.input_n, self.hidden_n)
		self.output_weights = make_matrix(self.hidden_n, self.output_n)
		# init input_weights
		for i in range(self.input_n):
			for h in range(self.hidden_n):
				self.input_weights[i][h] = rand(-0.2, 0.2)
		# init output_weights
		for h in range(self.hidden_n):
			for j in range(self.output_n):
				self.output_weights[h][j] = rand(-2.0, 2.0)
	
	def predict(self, inputs):
		for i in range(self.input_n - 1):
			self.input_cells[i] = inputs[i]
		# 计算隐藏层输出
		for h in range(self.hidden_n):
			total = 0.0
			for i in range(self.input_n):
				total = total + self.input_cells[i] * self.input_weights[i][h]
			self.hidden_cells[h] = sigmoid(total)
		# 计算输出层输出
		for j in range(self.output_n):
			total = 0.0
			for h in range(self.hidden_n):
				total = total + self.hidden_cells[h] * self.output_weights[h][j]
			self.output_cells[j] = sigmoid(total)
		return self.output_cells[:]

	def back_propogate(self, case, label, learn):
		self.predict(case)
		output_deltas = [0.0] * self.output_n
		# 计算gj
		for j in range(self.output_n):
			error = label[j] - self.output_cells[j]
			output_deltas[j] = sigmoid_derivative(self.output_cells[j]) * error
		hidden_deltas = [0.0] * self.hidden_n
		# 计算eh
		for h in range(self.hidden_n):
			error = 0.0
			for j in range(self.output_n):
				error = error + output_deltas[j] * self.output_weights[h][j]
			hidden_deltas[h] = sigmoid_derivative(self.hidden_cells[h]) * error 
		# 更新隐藏层到输出层权值
		for h in range(self.hidden_n):
			for j in range(self.output_n):
				self.output_weights[h][j] += learn * output_deltas[j] * self.hidden_cells[h]
		# 更新输入层到隐藏层权值
		for i in range(self.input_n):
			for h in range(self.hidden_n):
				self.input_weights[i][h] += learn * hidden_deltas[h] * self.input_cells[i]		
		error = 0.0
		# 计算输出误差
		for j in range(len(label)):
			error += 0.5 * (label[j] - self.output_cells[j]) ** 2
		return error
		
	def train(self, cases, labels, limit=10000, learn=0.05):	
		for i in range(limit):
			error = 0.0
			for j in range(len(cases)):
				label = labels[j]
				case = cases[j]
				error += self.back_propogate(case, label, learn)

	def test(self):
		cases = [
				[0, 0],
				[0, 1],
				[1, 0],
				[1, 1],
				]
		labels = [[0], [1], [1], [0]]
		self.setup(2, 5, 1)
		self.train(cases, labels, 10000, 0.05)
		for case in cases:
			print(self.predict(case))

if __name__ == '__main__':
	nn = BPNeuralNetwork()
	nn.test()
	print(np.array(nn.input_weights[:]))
	print(np.array(nn.output_weights[:]))
