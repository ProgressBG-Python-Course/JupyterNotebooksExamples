import numpy as np
import pandas as pd

from sklearn import datasets
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import preprocessing

class DataGenerator:
	def randUniform(self,n,m):
		data = np.random.rand(n, m)
		return data
	def randInt(self, n, m, range):
		data = np.random.choice(a=range, size=(n, m))
		return data
	def randBeta(self, n, m):
		data = np.random.beta(a=3, b=3, size=(n, m))
		return data
	def randExp(self, samples):
		# generate 1000 data points randomly drawn from an exponential distribution
		return np.random.exponential(samples = 1000)


class Scaler:
	def __init__(self):
		pass

	def Standardize(self, df):
		scl = preprocessing.StandardScaler().fit(df)
		print(scl)
		return scl.transform(df)


def plotDistribution(data1, data2):
	# plot both in 1 row, 2 columns
	fig, ax=plt.subplots(1,2)

	# print(data1['data'])

	sns.distplot(data1['data'], ax=ax[0], color='y')
	ax[0].set_title(data1['title'])

	sns.distplot(data2['data'], ax=ax[1])
	ax[1].set_title(data2['title'])
	plt.show()

def plotScater():
	# plot scatered data
	# plt.figure(figsize=(8,6))
	plt.scatter(x=data[:, 0], y=data[:, 1], color='red',
				label=title1, alpha=0.3)

	plt.scatter(x=data_std[:, 0], y=data_std[:, 1], color='blue',
				label=title2, alpha=0.3)

	plt.legend( loc='upper left')
	plt.grid()

	plt.tight_layout()
	# plt.title('random beta distribution')
	plt.show()

# generate some random data
dg = DataGenerator()
data_initial = dg.randInt(n=1000,m=2, range=100)
title1 = 'Initial Data'
print(f'\n{title1}\n{data_initial}')


# standardize:
title2 = 'Standardized'
data_std = Scaler().Standardize(data_initial)
print(f'\n{title2}\n{data_std}')

plotDistribution(
	data1={
	'title': title1,
	'data': data_initial
	},
	data2={
		'title': title2,
		'data': data_std,
	}
)

