from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt


boston = datasets.load_boston()
# create a data set for input and one data set for target variables
data = boston.data
target = boston.target

response = raw_input("Do you want to see the description for the data set : press y/Y to see details ")
if response.lower() == 'y':
	print boston.DESCR

#define list of training vectors and test vector for input data and target data
training_input = list()
training_target = list() #Normalising target is not mandatory
testing_input = list()
testing_target = list()

#fill the training and test  vector with data in 9:1 ratio
for i in range(0,data.shape[0]):
	if target[i] == 0:
		print 'skipping the observation :', data[i]
	else:
		new_row = data[i][:]
		if (i % 10 == 0):
			testing_input.append(new_row)
			testing_target.append(target[i])
		else:
			training_input.append(new_row)
			training_target.append(target[i])

#convert lists to numpy array for easy implementation later
training_input = np.array(training_input)
training_target = np.array(training_target)
testing_input = np.array(testing_input)
testing_target = np.array(testing_target)

#store a list for learning rate for which analysis has to be done
learning_rates = [0.00001,0.0001,0.001,0.01,0.1,1]
#total number of epochs = 10
total_epochs = 10 							

#normalise vectors function which internally calls functions to normalise input and targets depending on the number of columns
def normalise_vectors():
	normalise_input(training_input)
	normalise_input(testing_input)
	normalise_target(training_target)
	normalise_target(testing_target)

def normalise_input(dataset): 				#dataset is either the training/testing data numpy array 	
	for j in range(0,dataset.shape[1]): 	#run for each column ie 13 times as we have 13 columns
		minValue = min(dataset[:,j]) 		#find minimum element in jth column
		maxValue = max(dataset[:,j])		#find maximum element in jth column

		for i in range(dataset.shape[0]): 	#for each row element in jth column ie loops runs for number of rows in data array
			dataset[i,j] = (dataset[i,j] - minValue)/float(maxValue - minValue)
	

def normalise_target(dataset):				#dataset is either the training/testing target numpy array 
	minValue = min(dataset) 				#find minimum element in target.since target has only one column
	maxValue = max(dataset)					#find maximum element in target.since target has only one column

	for i in range(dataset.shape[0]): 		#for each row element in jth column ie loops runs for number of rows in data array
		dataset[i] = (dataset[i] - minValue)/float(maxValue - minValue)
	

def model(beta_list,input_array):
	predicted_target = 0
	predicted_target += beta_list[0]
	for i in range(0, data.shape[1]): 		  # data.shape[1] = 13 ie for each column x[1],x[2]....x[13]
		predicted_target += beta_list[i+1] * input_array[i]
 
	return predicted_target

#calculate rms for each epoch for a particular learning rate
def calculate_rms(beta):
	x = 0
	for i in range(0,testing_input.shape[0]): #iterate till number of rows in testing input dataset
			error = model(beta,testing_input[i]) - testing_target[i]
			x = x + error ** 2

	return ((float(x)/testing_input.shape[0]) ** 0.5)

#function for building the model and calculation the rms associated with each learning rate for a particular epoch
def calculate_model_prediction_rms():
	normalise_vectors()						 #normalise vectors before building the model.
	error = 0	
	
	for learning_index in range(0,len(learning_rates)):		
		rms = list() 						 #create a list to store rms value for each epoch for a particular learning rate
		beta = [0]*((data.shape[1])+1)		 #beta are one number more than the number of inputs and initialised to zero
		number_epochs = 0					 #initiate epoch to zero initially

		while (number_epochs < total_epochs):	#loop through the code for total_epochs (10) number of times
	 		for i in range(0,training_input.shape[0]): 					   #iterate till number of rows in training input dataset
				error = model(beta,training_input[i]) - training_target[i] #calculate error for each row in the input dataset
				for j in range(0,len(beta)):		 #loop runs from 0 to 13 ie B0 to B13 ie 14 beta values for 13 columns in input
					if (j == 0):
						beta[j] = beta[j] - learning_rates[learning_index] * error 	#beta 0 is modified with the learning rate and error
					else:
						#train other beta values ie beta 1 to beta 13 with error,learning rate and input x value
						beta[j] = beta[j] - learning_rates[learning_index] * error * training_input[i][j-1]
					
			rms.append(calculate_rms(beta))								   #calculate rms for each epoch and append to the list
			number_epochs += 1											   #increment count of epochs	

		#plotting rms values
		plt.plot(range(total_epochs),rms,color='red')
		plt.xticks(range(10) + [1] * 10)
		plt.xlabel("Epoch Number")
		plt.ylabel("RMS Value")
		plt.suptitle("RMS Curves for a Learning rate",color = 'blue')
		plt.title("Learning Rate : " "%.6f" % float(learning_rates[learning_index]))
		plt.show()
		
		

user_input = raw_input("Do you want to see the division of the data sets? Press y/Y to view : ")
if user_input.lower() == 'y':
	print "training_input",training_input.shape
	print "training_target",training_target.shape
	print "testing_input",testing_input.shape
	print "testing_target",testing_target.shape

#Function call for the main function which internally calls all the function.
calculate_model_prediction_rms()





