"""
:------------------------------------------------------------------------------:
`Bayesian neuroevolution using distributed swarm optimisation and tempered MCMC`
 Authors:   Arpit Kapoor (kapoor.arpit97@gmail.com)
            Eshwar Nukala (eshwar@iitg.ac.in)
            Dr Rohitash Chandra (rohitash.chandra@unsw.edu.au)
:------------------------------------------------------------------------------:
"""
import numpy as np
import random
import time
import math
import pandas as pd
import os
from pso.pso import PSO, Particle
from config.config import opt
import copy

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


class MCMC(PSO, Particle):
    def __init__(self, opt, path, geometric=True):
        self.opt = opt
        self.train_data = opt.train_data
        self.test_data = opt.test_data
        self.topology = list(map(int,opt.topology.split(',')))
        self.num_param = (self.topology[0] * self.topology[1]) + (self.topology[1] * self.topology[2]) + self.topology[1] + self.topology[2]   
        self.problem_type = opt.problem_type


        self.num_samples = opt.num_samples
        self.pop_size=opt.population_size
        self.directory = path
        self.path=path
        self.w_size = (self.topology[0] * self.topology[1]) + (self.topology[1] * self.topology[2]) + self.topology[1] + self.topology[2]
        self.neural_network = Network(self.topology, self.train_data, self.test_data)
        self.initialize_sampling_parameters() 
        max_limit=10
        min_limit=-10

        self.min_limits = np.repeat(min_limit, self.w_size)
        self.max_limits = np.repeat(max_limit, self.w_size)
        max_limit_vel=(self.weights_stepsize)*(self.weights_stepsize)*20
        min_limit_vel=self.weights_stepsize*self.weights_stepsize*-20
        self.min_limits_vel = np.repeat(min_limit_vel, self.w_size)
        self.max_limits_vel = np.repeat(max_limit_vel, self.w_size)
        #self.initialize_sampling_parameters()
        make_directory(path)
        PSO.__init__(self, self.pop_size, self.w_size, self.max_limits, self.min_limits,self.neural_network.evaluate_fitness,opt.problem_type,self.max_limits_vel,self.min_limits_vel)
        


    def fitness_function(self, x):
        fitness = self.neural_network.evaluate_fitness(x)
        return fitness

    def initialize_sampling_parameters(self):
        self.eta_stepsize = 0.005
        self.weights_stepsize=0.05
        self.sigma_squared = 50
        self.alpha=.1
        self.nu_1 = 0
        self.nu_2 = 0
        self.start = time.time()

    def decayed_learning_rate(self,step):
        decay_steps=self.num_samples/self.pop_size
        
        step = min(step, decay_steps)
        cosine_decay = 0.5 * (1 + np.cos(np.pi * step / decay_steps))
        decayed = (1 - self.alpha) * cosine_decay + self.alpha
        return self.weights_stepsize * decayed

    @staticmethod
    def convert_time(secs):
        if secs >= 60:
            mins = str(int(secs/60))
            secs = str(int(secs%60))
        else:
            secs = str(int(secs))
            mins = str(00)
        if len(mins) == 1:
            mins = '0'+mins
        if len(secs) == 1:
            secs = '0'+secs
        return [mins, secs]

    @staticmethod
    def create_directory(directory):
        if not os.path.isdir(directory):
            os.mkdir(directory)

    @staticmethod
    def calculate_rmse(actual, targets):
        return np.sqrt((np.square(np.subtract(np.absolute(actual), np.absolute(targets)))).mean())

    @staticmethod
    def calculate_mse(actual, targets):
        return ((np.square(np.subtract(np.absolute(actual), np.absolute(targets)))).mean())

    @staticmethod
    def multinomial_likelihood(neural_network, data, weights):
        y = (data[:, neural_network.topology[0]:neural_network.topology[0]+ neural_network.topology[2]])
        #print((y[0]))
        fx,probability = neural_network.generate_output_classification(data, weights)
        rmse = neural_network.calculate_rmse(fx, y) # Can be replaced by calculate_nmse function for reporting NMSE
        #probability = neural_network.softmax(fx)
        #print(probability[6])
        loss = 0
        for index_1 in range(y.shape[0]):
            for index_2 in range(y.shape[1]):
                if y[index_1, index_2] == 1:
                    loss += np.log(probability[index_1, index_2])
        out = np.argmax(probability, axis=1)
        y_out = np.argmax(y, axis=1)
        count = 0
        for index in range(y_out.shape[0]):
            if out[index] == y_out[index]:
                count += 1
        accuracy = (count)/y_out.shape[0] * 100
        #print(rmse)
        return [loss, rmse, accuracy]

    @staticmethod
    def classification_prior(sigma_squared, weights):
        part_1 = -1 * ((weights.shape[0]) / 2) * np.log(2*np.pi*sigma_squared)
        part_2 = 1 / (2 * sigma_squared) * (sum(np.square(weights)))
        log_loss = part_1 - part_2
        return log_loss

    @staticmethod
    def gaussian_likelihood(neural_network, data, weights, tausq):
        desired = data[:, neural_network.topology[0]: neural_network.topology[0] + neural_network.topology[2]]
        prediction = neural_network.generate_output(data, weights)
        rmse = MCMC.calculate_rmse(prediction, desired)
        loss = -0.5 * np.log(2 * np.pi * tausq) - 0.5 * np.square(desired - prediction) / tausq
        return [np.sum(loss), rmse]

    @staticmethod
    def gaussian_prior(sigma_squared, nu_1, nu_2, weights, tausq):
        part1 = -1 * (weights.shape[0] / 2) * np.log(sigma_squared)
        part2 = 1 / (2 * sigma_squared) * (sum(np.square(weights)))
        log_loss = part1 - part2 - (1 + nu_1) * np.log(tausq) - (nu_2 / tausq)
        return log_loss

    def likelihood_function(self, neural_network, data, weights, tau):
        if self.problem_type == 'regression':
            likelihood, rmse = self.gaussian_likelihood(neural_network, data, weights, tau*tau)
            return likelihood, rmse, None
        elif self.problem_type == 'classification':
            likelihood, rmse, accuracy = self.multinomial_likelihood(neural_network, data, weights)
            return likelihood, rmse, accuracy

    def prior_function(self, weights, tau):  
        if self.problem_type == 'regression':
            loss = self.gaussian_prior(self.sigma_squared, self.nu_1, self.nu_2, weights, tau*tau)
        elif self.problem_type == 'classification':
            loss = self.classification_prior(self.sigma_squared, weights)
        return loss

    def evaluate_proposal(self, neural_network, train_data, test_data, weights_proposal, tau_proposal, likelihood_current, prior_current,diff_prop):
        accept = False
        likelihood_ignore, rmse_test_proposal, acc_test = self.likelihood_function(neural_network, test_data, weights_proposal, tau_proposal)
        likelihood_proposal, rmse_train_proposal, acc_train = self.likelihood_function(neural_network, train_data, weights_proposal, tau_proposal)
        prior_proposal = self.prior_function(weights_proposal, tau_proposal)
        difference_likelihood = likelihood_proposal - likelihood_current 
        difference_prior = prior_proposal - prior_current
        #print(difference_likelihood,difference_prior)
        mh_ratio = difference_prior+difference_likelihood+diff_prop
        
        u = np.log(np.random.uniform(0,1))
        print(difference_prior,difference_likelihood,diff_prop,u)
        if u < mh_ratio:
            accept = True
            likelihood_current = likelihood_proposal
            prior_current = prior_proposal
        if acc_train == None:
            return accept, rmse_train_proposal, rmse_test_proposal,None,None, likelihood_current, prior_current
        else:
            return accept, rmse_train_proposal, rmse_test_proposal, acc_train, acc_test, likelihood_current, prior_current

    def mcmc_sampler(self, save_knowledge=True):    
        #print(self.directory)       
        save_knowledge = True
        train_rmse_file = open(os.path.join(self.path, 'train_rmse.csv'), 'w')
        test_rmse_file = open(os.path.join(self.path, 'test_rmse.csv'), 'w')
        if opt.problem_type == 'classification':
            train_acc_file = open(os.path.join(self.path, 'train_acc.csv'), 'w')
            test_acc_file = open(os.path.join(self.path, 'test_acc.csv'), 'w')
        
        

        # Initialize MCMC
        
        self.start_time = time.time()

       

        train_size = self.train_data.shape[0]
        test_size = self.test_data.shape[0]
        y_test = self.test_data[:, self.topology[0]: self.topology[0] + self.topology[2]]
        y_train = self.train_data[:, self.topology[0]: self.topology[0] + self.topology[2]]
        swarm_initial, best_swarm_pos, best_swarm_err =  self.swarm, self.best_swarm_pos, self.best_swarm_err
        swarm_current = copy.copy(swarm_initial)
        swarm_proposal = copy.copy(swarm_initial)
        weights_current = np.random.uniform(-2,2,size=(self.pop_size,self.w_size))
        weights_proposal = np.zeros(shape=(self.pop_size,self.w_size))
        weights_proposal_evolve = np.zeros(shape=(self.pop_size,self.w_size))
        eta = [None for n in range(self.pop_size)]
        eta_proposal = [None for n in range(self.pop_size)]
        tau_proposal = [None for n in range(self.pop_size)]
        prior = [None for n in range(self.pop_size)]
        likelihood = [None for n in range(self.pop_size)]
        rmse_train_current = [None for n in range(self.pop_size)]
        rmse_test_current = [None for n in range(self.pop_size)]
        accuracy_test_current = [None for n in range(self.pop_size)]
        accuracy_train_current = [None for n in range(self.pop_size)]

        for i in range(self.pop_size):
            #weights_current[i]=swarm_current[i].position
            prediction_train = self.neural_network.generate_output(self.train_data, weights_current[i])
            prediction_test = self.neural_network.generate_output(self.test_data, weights_current[i])
            eta[i] = np.log(np.var(prediction_train - y_train))
            
            eta_proposal[i] = copy.copy(eta[i])
            tau_proposal[i] = np.exp(eta[i])

            prior[i] = self.prior_function(weights_current[i], tau_proposal[i])
            likelihood[i], rmse_train_current[i], accuracy_train_current[i] = self.likelihood_function(self.neural_network, self.train_data, weights_current[i].copy(), tau_proposal[i].copy())

            rmse_test_current[i],accuracy_test_current[i]=self.neural_network.classification_perf(weights_current[i], 'test',opt.problem_type)

        num_accept = 0
        sample=0

        while(sample<1000):
            step=sample/self.pop_size
            w_stepsize=self.decayed_learning_rate(step)
            e_stepsize=w_stepsize
            for i in range(self.pop_size):
                weights_proposal[i]=weights_current[i] + np.random.normal(0, w_stepsize, self.w_size)
                eta_proposal[i] = eta[i] + np.random.normal(0, e_stepsize, 1)
                tau_proposal[i] = np.exp(eta_proposal[i])
                
                diff_prop=0

                accept, rmse_train_current[i], rmse_test_current[i], accuracy_train_current[i], accuracy_test_current[i], likelihood[i], prior[i] = self.evaluate_proposal(self.neural_network, self.train_data, self.test_data, weights_proposal[i], tau_proposal[i], likelihood[i], prior[i],diff_prop)

                if accept:
                    num_accept += 1
                    weights_current[i] = weights_proposal[i]
                    self.swarm[i].position=weights_current[i]
                    self.swarm[i].error=rmse_train_current[i]
                    eta[i] = eta_proposal[i]

                if save_knowledge:
                    np.savetxt(train_rmse_file, [rmse_train_current[i]])
                    np.savetxt(test_rmse_file, [rmse_test_current[i]])
                    if(opt.problem_type== 'classification'):
                        np.savetxt(train_acc_file, [accuracy_train_current[i]])
                        np.savetxt(test_acc_file, [accuracy_test_current[i]])
                
                #accuracy_test_check,_=self.neural_network.classification_perf(weights_current,'test')
                if(opt.problem_type=='regression'):
                    print(f'Sample: {sample}  RMSE Train: {rmse_train_current[i]:.4f}  RMSE test: {rmse_test_current[i]:.4f} accept r: {num_accept/(sample+1):.4f}')
                else:
                    print(f'Sample: {sample}  accuracy train: {accuracy_train_current[i]:.2f}  accuracy test: {accuracy_test_current[i]:.2f} RMSE Train: {rmse_train_current[i]:.4f} accept r: {num_accept/(sample+1):.4f}')
                

                elapsed_time = ":".join(MCMC.convert_time(time.time() - self.start))
                
                sample+=1
            
        for i in range(self.pop_size):
            if self.swarm[i].error < self.swarm[i].best_part_err:
                # print('hello')
                self.swarm[i].best_part_err = self.swarm[i].error.copy()
                self.swarm[i].best_part_pos = copy.copy(self.swarm[i].position)

            if self.swarm[i].error < self.best_swarm_err:
                # print('hello again')
                self.best_swarm_err = self.swarm[i].error.copy()
                self.best_swarm_pos = copy.copy(self.swarm[i].position)
            else:
                pass
                # print(swarm[i].position, best_swarm_pos)
        while(sample<self.num_samples):
            step=sample/self.pop_size
            w_stepsize=self.decayed_learning_rate(step)
            e_stepsize=w_stepsize
            swarm_new,new_best_pos,new_best_err=self.evolve(copy.copy(self.swarm), self.best_swarm_pos.copy(), self.best_swarm_err.copy())

            #self.swarm, self.best_swarm_pos, self.best_swarm_err = self.evolve(copy.copy(self.swarm), self.best_swarm_pos.copy(), self.best_swarm_err.copy())
            for i in range(self.pop_size):

                weights_proposal[i]=swarm_new[i].position + np.random.normal(0, w_stepsize, self.w_size)
                weights_proposal_evolve[i]=self.evolve_single_particle(weights_proposal[i].copy(),copy.copy(swarm_new),new_best_pos.copy(), new_best_err.copy(),i)
                #print(weights_proposal_evolve)


            for i in range(self.pop_size):
                eta_proposal[i] = eta[i] + np.random.normal(0, e_stepsize, 1)
                tau_proposal[i] = np.exp(eta_proposal[i])
                wc_delta = (weights_current[i]- weights_proposal_evolve[i]) 
                wp_delta = (weights_proposal[i] - swarm_new[i].position)
                
                sigma_sq = w_stepsize*w_stepsize

                first = -0.5 * np.sum(wc_delta  *  wc_delta  ) / sigma_sq  # this is wc_delta.T  *  wc_delta /sigma_sq
                second = -0.5 * np.sum(wp_delta * wp_delta ) / sigma_sq
                #print(first,second,wc_delta,wp_delta)
                
                #print(np.sum(wc_delta  *  wc_delta  ),np.sum(wp_delta * wp_delta ))
                diff_prop =  (first - second)
                #diff_prop=0
                

                accept, rmse_train_current[i], rmse_test_current[i], accuracy_train_current[i], accuracy_test_current[i], likelihood[i], prior[i] = self.evaluate_proposal(self.neural_network, self.train_data, self.test_data, weights_proposal[i], tau_proposal[i], likelihood[i], prior[i],diff_prop)
                #print(weights_current[i])
                if accept:
                    num_accept += 1
                    weights_current[i] = weights_proposal[i]
                    self.swarm[i].position=weights_current[i]
                    self.swarm[i].error=rmse_train_current[i]
                    #swarm_new[i].position=weights_current[i]
                    # swarm_new[i].error=rmse_train_current[i]
                    eta[i] = eta_proposal[i]


                if save_knowledge:
                    np.savetxt(train_rmse_file, [rmse_train_current[i]])
                    np.savetxt(test_rmse_file, [rmse_test_current[i]])
                    if(opt.problem_type== 'classification'):
                        np.savetxt(train_acc_file, [accuracy_train_current[i]])
                        np.savetxt(test_acc_file, [accuracy_test_current[i]])

                print(likelihood[i])
                #print(weights_current[i])
                #accuracy_test_check,_=self.neural_network.classification_perf(weights_current,'test')
                if(opt.problem_type=='regression'):
                    print(f'Sample: {sample}  RMSE Train: {rmse_train_current[i]:.4f}  RMSE test: {rmse_test_current[i]:.4f} accept r: {num_accept/(sample+1):.4f}')
                else:
                    print(f'Sample: {sample}  accuracy train: {accuracy_train_current[i]:.2f}  accuracy test: {accuracy_test_current[i]:.2f} RMSE Train: {rmse_train_current[i]:.4f} accept r: {num_accept/(sample+1):.4f}')
                

                elapsed_time = ":".join(MCMC.convert_time(time.time() - self.start))
                
                sample+=1
            
            for i in range(self.pop_size):
                if self.swarm[i].error < self.swarm[i].best_part_err:
                    # print('hello')
                    self.swarm[i].best_part_err = self.swarm[i].error.copy()
                    self.swarm[i].best_part_pos = copy.copy(self.swarm[i].position)

                if self.swarm[i].error < self.best_swarm_err:
                    # print('hello again')
                    self.best_swarm_err = self.swarm[i].error.copy()
                    self.best_swarm_pos = copy.copy(self.swarm[i].position)
                else:
                    pass
                    # print(swarm[i].position, best_swarm_pos)

        train_rmse_file.close()
        test_rmse_file.close()
        if opt.problem_type == 'classification':
            train_acc_file.close()
            test_acc_file.close()
        burn_in = int(sample*opt.burn_in)

        rmse_train = np.zeros((1,sample - burn_in))
        rmse_test = np.zeros((1,sample - burn_in))
        if self.problem_type == 'classification':
            acc_train = np.zeros((1,sample - burn_in))
            acc_test = np.zeros((1,sample - burn_in))
        accept_ratio = np.zeros((1,1))

        #print(opt.root)
        file_name = os.path.join(self.path, 'test_rmse.csv')
        #print(file_name)
        dat = np.genfromtxt(file_name, delimiter=',')
        #print(np.shape(dat))
        rmse_test[0,:] = dat[burn_in:]

        file_name = os.path.join(self.path, 'train_rmse.csv')
        dat = np.genfromtxt(file_name, delimiter=',')
        #print(np.shape(dat))
        rmse_train[0,:] = dat[burn_in:]

        if self.problem_type == 'classification':
            file_name = os.path.join(self.path, 'test_acc.csv')
            dat = np.genfromtxt(file_name, delimiter=',')
            acc_test[0,:] = dat[burn_in:]

            file_name = os.path.join(self.path, 'train_acc.csv')
            dat = np.genfromtxt(file_name, delimiter=',')
            acc_train[0,:] = dat[burn_in:]

        self.rmse_train = rmse_train.reshape((sample - burn_in), 1)
        self.rmse_test = rmse_test.reshape((sample - burn_in), 1)
        if self.problem_type == 'classification':
            self.acc_train = acc_train.reshape((sample - burn_in), 1)
            self.acc_test = acc_test.reshape((sample - burn_in), 1)

        # PLOT THE RESULTS
        #self.plot_figures()    

        with open(os.path.join(self.path, 'results.txt'), 'w') as results_file:
            results_file.write(f'NUMBER OF SAMPLES PER CHAIN: {sample}\n')
            #results_file.write(f'NUMBER OF CHAINS: {self.num_chains}\n')
            #results_file.write(f'NUMBER OF SWAPS MAIN: {total_swaps_main}\n')
            results_file.write(f'PERCENT SAMPLES ACCEPTED: {num_accept/(sample)*100:.2f}\n')
            #results_file.write(f'SWAP ACCEPTANCE: {self.num_swap*100/self.total_swap_proposals:.4f} %\n')
            #results_file.write(f'SWAP ACCEPTANCE MAIN: {swaps_appected_main*100/total_swaps_main:.2f} %\n')
            if self.problem_type == 'classification':
                results_file.write(f'ACCURACY: \n\tTRAIN -> MEAN: {self.acc_train.mean():.2f} STD: {self.acc_train.std():.2f} BEST: {self.acc_train.max():.2f} \n\tTEST -> MEAN: {self.acc_test.mean():.2f} STD: {self.acc_test.std():.2f} BEST: {self.acc_test.max():.2f}\n')
            results_file.write(f'RMSE: \n\tTRAIN -> MEAN: {self.rmse_train.mean():.5f} STD: {self.rmse_train.std():.5f}\n\tTEST -> MEAN: {self.rmse_test.mean():.5f} STD: {self.rmse_test.std():.5f}\n')
            results_file.write(f'TIME TAKEN: {elapsed_time} secs\n')

            
        
        #print("NUMBER OF SWAPS MAIN =", total_swaps_main)
        #print("SWAP ACCEPTANCE = ", self.num_swap*100/self.total_swap_proposals," %")
        #print("SWAP ACCEPTANCE MAIN = ", swaps_appected_main*100/total_swaps_main," %")

        if self.problem_type == 'classification':
            return (self.rmse_train, self.rmse_test, self.acc_train, self.acc_test,num_accept)
        return (self.rmse_train, self.rmse_test,num_accept)

    

def make_directory(path):
    if not os.path.isdir(path):
        os.makedirs(path)

    

if __name__ == '__main__':
    make_directory(os.path.join('Results1'))
    results_dir = os.path.join( 'Results1', '{}_{}'.format(opt.problem, opt.run_id))
    make_directory(results_dir)
    #print(results_dir)
    

    # READ DATA
    data_path = os.path.join(opt.root, 'Datasets')
    #print(opt.root)

    opt.train_data = np.genfromtxt(os.path.join(data_path, opt.train_data), delimiter=' ',dtype=None)
    opt.test_data = np.genfromtxt(os.path.join(data_path, opt.test_data),  delimiter=' ',dtype=None)

    # CREATE EVOLUTIONARY PT CLASS
    mcmc = MCMC(opt, results_dir)
    #print(results_dir)
    # INITIALIZE PT CHAINS
    np.random.seed(int(time.time()))
    #evo_pt.initialize_chains()
    # quit()

    # RUN EVO PT
    if opt.problem_type == 'classification':
        rmse_train, rmse_test, acc_train, acc_test,num_accept = mcmc.mcmc_sampler()
        print(np.mean(acc_train),np.std(acc_train),np.max(acc_train), np.mean(acc_test), np.std(acc_test),np.max(acc_test),num_accept)
    elif opt.problem_type == 'regression':
        rmse_train, rmse_test ,num_accept= mcmc.mcmc_sampler()
        print(np.mean(rmse_train),np.std(rmse_train),np.min(rmse_train), np.mean(rmse_test), np.std(rmse_test),np.min(rmse_test),num_accept)

    