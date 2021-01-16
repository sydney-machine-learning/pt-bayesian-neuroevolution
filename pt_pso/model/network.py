"""
:------------------------------------------------------------------------------:
`Bayesian neuroevolution using distributed swarm optimisation and tempered MCMC`
 Authors:   Arpit Kapoor (kapoor.arpit97@gmail.com)
            Eshwar Nukala (eshwar@iitg.ac.in)
            Dr Rohitash Chandra (rohitash.chandra@unsw.edu.au)
:------------------------------------------------------------------------------:
"""

import numpy as np
import time

class Network:

    def __init__(self, topology, train_data, test_data, learn_rate = 0.5, alpha = 0.1):
        self.topology = topology  # NN topology [input, hidden, output]
        np.random.seed(int(time.time()))
        self.train_data = train_data
        self.test_data = test_data
        self.W1 = np.random.randn(self.topology[0], self.topology[1]) / np.sqrt(self.topology[0]),
        self.B1 = np.random.randn(1, self.topology[1]) / np.sqrt(self.topology[1])  # bias first layer
        self.W2 = np.random.randn(self.topology[1], self.topology[2]) / np.sqrt(self.topology[1])
        self.B2 = np.random.randn(1, self.topology[2]) / np.sqrt(self.topology[1])  # bias second layer
        self.hidout = np.zeros((1, self.topology[1]))  # output of first hidden layer
        self.out = np.zeros((1, self.topology[2]))  # output last layer
        self.pred_class=0

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sampleEr(self, actualout):
        error = np.subtract(self.out, actualout)
        sqerror = np.sum(np.square(error)) / self.topology[2]
        return sqerror

    def ForwardPass(self, X):
        z1 = X.dot(self.W1) - self.B1
        self.hidout = self.sigmoid(z1)  # output of first hidden layer
        z2 = self.hidout.dot(self.W2) - self.B2
        self.out = self.sigmoid(z2)  # output second hidden layer

        self.pred_class = np.argmax(self.out)


        #print(self.pred_class, self.out, '  ---------------- out ')

    '''def BackwardPass(self, Input, desired):
        out_delta = (desired - self.out).dot(self.out.dot(1 - self.out))
        hid_delta = out_delta.dot(self.W2.T) * (self.hidout * (1 - self.hidout))
        print(self.B2.shape)
        self.W2 += (self.hidout.T.reshape(self.Top[1],1).dot(out_delta) * self.lrate)
        self.B2 += (-1 * self.lrate * out_delta)
        self.W1 += (Input.T.reshape(self.Top[0],1).dot(hid_delta) * self.lrate)
        self.B1 += (-1 * self.lrate * hid_delta)'''


    def calculate_rmse(self,predict, targets):
        #targets=np.argmax(targets,axis=1)
        #print(predict,targets)
        return np.sqrt((np.square(np.subtract(np.absolute(predict), np.absolute(targets)))).mean())
    
    def calculate_mse(self,predict, targets):
        #targets=np.argmax(targets,axis=1)
        #print(predict,targets)
        return ((np.square(np.subtract(np.absolute(predict), np.absolute(targets)))).mean())
    
    def forward_pass(self, X):
        z1 = X.dot(self.W1) - self.B1
        self.hidout = self.sigmoid(z1)  # output of first hidden layer
        z2 = self.hidout.dot(self.W2) - self.B2
        self.out = self.sigmoid(z2)  # output second hidden layer

    def decode(self, w):
        w_layer1size = self.topology[0] * self.topology[1] 
        w_layer2size = self.topology[1] * self.topology[2]

        w_layer1 = w[0:w_layer1size]
        self.W1 = np.reshape(w_layer1, (self.topology[0], self.topology[1]))

        w_layer2 = w[w_layer1size:w_layer1size + w_layer2size]
        self.W2 = np.reshape(w_layer2, (self.topology[1], self.topology[2]))
        self.B1 = w[w_layer1size + w_layer2size:w_layer1size + w_layer2size + self.topology[1]].reshape(1,self.topology[1])
        self.B2 = w[w_layer1size + w_layer2size + self.topology[1]:w_layer1size + w_layer2size + self.topology[1] + self.topology[2]].reshape(1,self.topology[2])

 

    def encode(self):
        w1 = self.W1.ravel()
        w1 = w1.reshape(1,w1.shape[0])
        w2 = self.W2.ravel()
        w2 = w2.reshape(1,w2.shape[0])
        w = np.concatenate([w1.T, w2.T, self.B1.T, self.B2.T])
        w = w.reshape(-1)
        return w

    @staticmethod
    def softmax(fx):
        ex = np.exp(fx)
        sum_ex = np.sum(ex,axis=1)
        sum_ex = np.multiply(np.ones(ex.shape), sum_ex[:, np.newaxis])
        probability = np.divide(ex, sum_ex)
        return probability
 


    '''def langevin_gradient(self, data, w, depth):  # BP with SGD (Stocastic BP)

        self.decode(w)  # method to decode w into W1, W2, B1, B2.
        size = data.shape[0]

        Input = np.zeros((1, self.Top[0]))  # temp hold input
        Desired = np.zeros((1, self.Top[2]))
        fx = np.zeros(size)

        for i in range(0, depth):
            for i in range(0, size):
                pat = i
                Input = data[pat, 0:self.Top[0]]
                Desired = data[pat, self.Top[0]:]
                self.ForwardPass(Input)
                self.BackwardPass(Input, Desired)
        w_updated = self.encode()

        return  w_updated'''

    def generate_output_classification(self, data, w ):  # BP with SGD (Stocastic BP)

        self.decode(w)  # method to decode w into W1, W2, B1, B2.
        size = data.shape[0]

        Input = np.zeros((1, self.topology[0]))  # temp hold input
        Desired = np.zeros((1, self.topology[2]))
        fx = np.zeros((size,self.topology[2]))
        #print(np.shape(fx),np.shape(self.topology[0]))
        prob = np.zeros((size,self.topology[2]))

        for i in range(0, size):  # to see what fx is produced by your current weight update
            Input = data[i, 0:self.topology[0]]
            self.ForwardPass(Input)
            fx[i] = self.out
            
        prob=self.softmax(fx)
        #print(fx, 'fx')
        #print(prob, 'prob' )

        return fx, prob

    def generate_output(self, data, w):  # BP with SGD (Stocastic BP)
        self.decode(w)  # method to decode w into W1, W2, B1, B2.
        size = data.shape[0]
        Input = np.zeros((1, self.topology[0]))  # temp hold input
        fx = np.zeros((size,self.topology[2]))
        for i in range(0, size):  # to see what fx is produced by your current weight update
            Input = data[i, 0:self.topology[0]]
            self.forward_pass(Input)
            fx[i] = self.out
        return fx

    def accuracy(self , pred, actual):
        count=0 
        actual=np.argmax(actual,axis=1)
        prob=self.softmax(pred)
        prob=np.argmax(prob,axis=1)
        for i in range(prob.shape[0]):
            if prob[i] == actual[i]:
                count+=1
        return (float(count)/pred.shape[0])*100

        

    def classification_perf(self, x, type_data,problem_type):

        if type_data == 'train':
            data = self.train_data
        else:
            data = self.test_data

        y = (data[:, self.topology[0]:self.topology[0] + self.topology[2]])
        #print(np.shape(y))
        if problem_type=='regression':
            fx = self.generate_output(data,x)
            fit= self.calculate_rmse(fx,y) 
            return fit,None
        else:
            fx,prob = self.generate_output_classification(data,x)
            fit= self.calculate_rmse(fx,y) 
            acc = self.accuracy(fx,y) 
            return fit,acc
        

       

        
    def evaluate_fitness(self, x,problem_type):    #  function  (can be any other function, model or diff neural network models (FNN or RNN ))
          
        fit ,_=self.classification_perf(x,'train',problem_type)



        return fit  #fit # note we will maximize fitness, hence minimize error