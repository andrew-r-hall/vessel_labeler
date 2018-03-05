'''A learning algorithm to label blood vessels in T1-weighted images, using MRA as 
training data'''
import gc
import numpy as np
import nibabel as nib
from sklearn.neural_network import MLPClassifier

### Loads a Nifti image, returns a numpy array
def load_image(filename):
	img	 = nib.load(filename)
	data = np.asarray(img.get_data())
	return(data)

### Takes a T1-weighted image to be processed into pad_depth^2 sized inputs to determine
### whether the voxel at the center is a vessel or not	
def preprocess_X(X_data , pad_depth):
	side_lengths = [3 , 5 , 7 , 9 , 11]
	side_length  = side_lengths[pad_depth-1]
	size = np.size(X_data)
	X_data = np.pad(X_data , pad_depth , 'constant' , constant_values=(0,0))
	X = []
	n=0
	for i in range(np.shape(X_data)[0] - pad_depth*2):
		for j in range(np.shape(X_data)[1] - pad_depth*2):
			for k in range(np.shape(X_data)[2] - pad_depth*2):
				kernel = np.asarray(X_data[i:i+side_length , j:j+side_length , k:k+side_length]).flatten()
				entry = np.concatenate((np.array([i+pad_depth , j+pad_depth , k+pad_depth]),kernel.flatten()))
				X.append(entry)
				n+=1
				pct = (n/float(size)) * 100
				if(pct % 10 == 0):
					print(pct)
	gc.collect()
	return(X)

	
def preprocess_Y(Y_data , X):
	Y = []
	n=0
	size = np.size(Y_data)
	for i in range(np.shape(Y_data)[0]):
		for j in range(np.shape(Y_data)[1]):
			for k in range(np.shape(Y_data)[2]):
				if(i,j,k == X[n][0],X[n][1],X[n][2]):
					Y.append(Y_data[i,j,k])
					pct = (n/float(size)) * 100
					if(pct % 10 ==0):
						print(pct)
				else:
					print('Alert! Alert! Dis bad bawss')
					break
				n+=1
	gc.collect()
	return(Y)				

def split_data(X,fraction):
	sep = int(len(X)*fraction)
	train = X[:sep]
	test  = X[sep:]	
	return(train,test)

def train_net(X_train,Y_train):
	hidden_layer = int(len(X_train[0])/2)
	net = MLPClassifier(solver='lbfgs' , alpha=1e-5 , hidden_layer_sizes=(hidden_layer) , random_state=1)
	net.fit(X_train,Y_train)
	return(net)

def test_net(X_test , Y_test , net):
	score = net.score(X_test , Y_test)
	return(score)


def run_program():	
	X_data = load_image('Normal001-T1-Flash.nii')
	Y_data = load_image('Normal001-MRA-masked.nii')

	X  = preprocess_X(X_data , 2)
	Y  = preprocess_Y(Y_data , X)

	net = train_net(X,Y)
	
	X_data = load_image('Normal002-MRA-masked.nii')
	Y_data = load_image('Normal002-T1-Flash.nii')
	
	X = preprocess_X(X_data , 2)
	Y = preprocess_Y(Y_data , X)
	
	score = test_net(X , Y , net)
	
	return(score , net)
