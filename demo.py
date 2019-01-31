import numpy as np # matrix operations
import matplotlib.pyplot as plt # visualization
from data import * #generate data
from activation import * # activation functions
from loss import * # loss function
from utils import * # postprocessing data

np.random.seed(0) 

# hyperparameters
hidden_unit = 8 # number of hidden units
output_unit = 1 # number of output units
binary_bit = 4 # number of bits 
display_step = 10000 # when model displays results
largest_num = 2**binary_bit-1 #the largest number with given number of bits
num_ex=50 # number of training examples we want to generate
alpha = 0.001 # learning rate

#parameters
wax = 2*np.random.random((hidden_unit,2)) - 1
waa = 2*np.random.random((hidden_unit,hidden_unit)) - 1
way = 2*np.random.random((output_unit,hidden_unit)) - 1
ba = 2*np.random.random((hidden_unit,1)) - 1
by = 2*np.random.random((output_unit,1)) - 1

#generate dataset
m,n,p = datagen(num_ex,binary_bit)

m_seq = np.zeros((m.shape[0],binary_bit))
n_seq = np.zeros((n.shape[0],binary_bit))
p_seq = np.zeros((p.shape[0],binary_bit))

for ex_index in range(m.shape[0]):
	m_seq[ex_index] = [float(bits) for bits in int2binary(m[ex_index][0],binary_bit)]
	n_seq[ex_index] = [float(bits) for bits in int2binary(n[ex_index][0],binary_bit)]
	p_seq[ex_index] = [float(bits) for bits in int2binary(p[ex_index][0],binary_bit)]

# the hidden layer matrices
a = {0: np.zeros((hidden_unit,m.shape[0]))}

# predictions
pred = {}

# used to store loss (used for visualization)
err = []

# initial step
j=0
try:
	# the training loops
	while True:

		overall = 0.

		dwax = np.zeros_like(wax)
		dway = np.zeros_like(way)
		dwaa = np.zeros_like(waa)
		dba = np.zeros_like(ba)
		dby = np.zeros_like(by)

		for time in range(1,binary_bit+1):
			x = np.array([m_seq[:,binary_bit-time],n_seq[:,binary_bit-time]])
			y = np.expand_dims(p_seq[:,binary_bit-time],axis=0)

			# forward prop
			a[time] = tanh(wax.dot(x) + waa.dot(a[time-1]) + ba)
			pred[binary_bit-time] = sigmoid(way.dot(a[time]) + by)

			overall += crossentropy(pred[binary_bit-time],y)

			# backpropagation
			error = pred[binary_bit-time] - y 

			dway_update = error.dot(a[time].T)
			dby_update = np.sum(error,axis=1,keepdims=True)

			dza = way.T.dot(error)*tanh(wax.dot(x) + waa.dot(a[time-1]) + ba,deriv=True)
			dwax_update = dza.dot(x.T)
			dwaa_update = dza.dot(a[time-1].T)
			dba_update = np.sum(dza,axis=1,keepdims=True)

			dwax += dwax_update
			dwaa += dwaa_update
			dway += dway_update
			dba += dba_update
			dby += dby_update

		# store the loss at the current time step
		err.append(overall)

		# update parameters
		wax -= alpha*dwax
		waa -= alpha*dwaa
		way -= alpha*dway 
		ba -= alpha*dba 
		by -= alpha*dby

		# display the results
		if j%display_step==0:
			print('--------------------------')
			print('Iteration %d'%j)
			print('Loss %s'%overall)
			test = np.random.randint(m.shape[0])
			prediction = [int(pred[i][0,test] >= 0.5) for i in range(binary_bit)]
			print('%d + %d = %d'%(int(m[test]),int(n[test]),binary2int(prediction)))
			print('--------------------------')

		j+=1
except KeyboardInterrupt:
	#visualize the results
	plt.plot(np.arange(1,1+len(err)),err)
	plt.show()