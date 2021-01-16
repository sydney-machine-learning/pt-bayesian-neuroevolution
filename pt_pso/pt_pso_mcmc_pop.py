"""
:------------------------------------------------------------------------------:
`Bayesian neuroevolution using distributed swarm optimisation and tempered MCMC`
 Authors:   Arpit Kapoor (kapoor.arpit97@gmail.com)
            Eshwar Nukala (eshwar@iitg.ac.in)
            Dr Rohitash Chandra (rohitash.chandra@unsw.edu.au)
:------------------------------------------------------------------------------:
"""
 
from __future__ import division
import multiprocessing
from multiprocessing import Process, Value
import numpy as np
import random
import time
import math
import random
import os, sys
from pprint import pprint
import json
import copy
import matplotlib.pyplot as plt

from config.config import opt
from model.network import Network
from pso.pso import PSO, Particle


class Replica(PSO, Process, Particle):
    def __init__(self, num_samples, burn_in, population_size, topology, train_data, test_data, directory, temperature, swap_sample, parameter_queue, problem_type,  main_process, event, active_chains, num_accepted, swap_interval, max_limit=(5), min_limit=-5):
        # Multiprocessing attributes
        multiprocessing.Process.__init__(self)
        self.process_id = temperature
        self.parameter_queue = parameter_queue
        self.signal_main = main_process
        self.event =  event
        self.active_chains = active_chains
        self.num_accepted = num_accepted
        self.event.clear()
        self.signal_main.clear()
        # Parallel Tempering attributes
        self.temperature = temperature
        self.swap_sample = swap_sample
        self.swap_interval = swap_interval
        self.burn_in = burn_in
        # MCMC attributes
        self.num_samples = num_samples
        
        self.topology = topology
        self.pop_size=population_size
        self.train_data = train_data
        self.test_data = test_data
        self.problem_type = problem_type
        self.directory = directory
        self.w_size = (topology[0] * topology[1]) + (topology[1] * topology[2]) + topology[1] + topology[2]
        self.neural_network = Network(topology, train_data, test_data)
        self.min_limits = np.repeat(min_limit, self.w_size)
        self.max_limits = np.repeat(max_limit, self.w_size)
        self.initialize_sampling_parameters()
        max_limit_vel=(self.weights_stepsize)*(self.weights_stepsize)*10
        min_limit_vel=self.weights_stepsize*self.weights_stepsize*-10
        self.min_limits_vel = np.repeat(min_limit_vel, self.w_size)
        self.max_limits_vel = np.repeat(max_limit_vel, self.w_size)
        self.create_directory(directory)
        PSO.__init__(self, self.pop_size, self.w_size, self.max_limits, self.min_limits,self.neural_network.evaluate_fitness,opt.problem_type,self.max_limits_vel,self.min_limits_vel)

    def initialize_sampling_parameters(self):
        self.weights_stepsize = 0.05
        self.eta_stepsize = 0.05
        self.alpha=.1
        self.sigma_squared = 50
        self.nu_1 = 0
        self.nu_2 = 0
        self.start_time = time.time()
    
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
    def multinomial_likelihood(neural_network, data, weights, temperature):
        y = (data[:, neural_network.topology[0]:neural_network.topology[0]+ neural_network.topology[2]])

        fx, probability = neural_network.generate_output_classification(data, weights)
        rmse = neural_network.calculate_rmse(fx, y) # Can be replaced by calculate_nmse function for reporting NMSE
        #probability = neural_network.softmax(fx)

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
        return [loss/temperature, rmse, accuracy]


    @staticmethod
    def classification_prior(sigma_squared, weights,temperature):
        part_1 = -1 * ((weights.shape[0]) / 2) * np.log(2*np.pi*sigma_squared)
        part_2 = 1 / (2 * sigma_squared) * (sum(np.square(weights)))
        log_loss = part_1 - part_2
        return log_loss/temperature

    @staticmethod
    def gaussian_likelihood(neural_network, data, weights, tausq, temperature):
        desired = data[:, neural_network.topology[0]: neural_network.topology[0] + neural_network.topology[2]]
        prediction = neural_network.generate_output(data, weights)
        rmse = neural_network.calculate_rmse(prediction, desired)
        loss = -0.5 * np.log(2 * np.pi * tausq) - 0.5 * np.square(desired - prediction) / tausq
        return [np.sum(loss)/temperature, rmse]

    @staticmethod
    def gaussian_prior(sigma_squared, nu_1, nu_2, weights, tausq,temperature):
        part1 = -1 * (weights.shape[0] / 2) * np.log(sigma_squared)
        part2 = 1 / (2 * sigma_squared) * (sum(np.square(weights)))
        log_loss = part1 - part2 - (1 + nu_1) * np.log(tausq) - (nu_2 / tausq)
        return log_loss/temperature


    def likelihood_function(self, neural_network, data, weights, tau, temperature):
        if self.problem_type == 'regression':
            likelihood, rmse = self.gaussian_likelihood(neural_network, data, weights, tau*tau, temperature)
            return likelihood, rmse, None
        elif self.problem_type == 'classification':
            likelihood, rmse, accuracy = self.multinomial_likelihood(neural_network, data, weights, temperature)
            return likelihood, rmse, accuracy

    def prior_function(self, weights, tau,temperature):
        if self.problem_type == 'regression':
            loss = self.gaussian_prior(self.sigma_squared, self.nu_1, self.nu_2, weights, tau*tau,temperature)
        elif self.problem_type == 'classification':
            loss = self.classification_prior(self.sigma_squared, weights,temperature)
        # quit()
        return loss

    def evaluate_proposal(self, neural_network, train_data, test_data, weights_proposal, tau_proposal, likelihood_current, prior_current,diff_prop):
        accept = False
        likelihood_ignore, rmse_test_proposal, acc_test = self.likelihood_function(neural_network, test_data, weights_proposal, tau_proposal, self.temperature)
        likelihood_proposal, rmse_train_proposal, acc_train = self.likelihood_function(neural_network, train_data, weights_proposal, tau_proposal, self.temperature)
        prior_proposal = self.prior_function(weights_proposal, tau_proposal,self.temperature)
        difference_likelihood = likelihood_proposal - likelihood_current
        difference_prior = prior_proposal - prior_current
        #print(difference_likelihood,difference_prior)
        mh_ratio = difference_prior+difference_likelihood+diff_prop
        u = np.log(np.random.uniform(0,1))
        if u < mh_ratio:
            accept = True
            likelihood_current = likelihood_proposal
            prior_current = prior_proposal
        if acc_train == None:
            return accept, rmse_train_proposal, rmse_test_proposal,None,None, likelihood_current, prior_current
        else:
            return accept, rmse_train_proposal, rmse_test_proposal, acc_train, acc_test, likelihood_current, prior_current

    def run(self):    
        print(f'Entered Run, chain: {self.temperature:.2f}')
        np.random.seed(int(self.temperature*1000))
        curr_temp=self.temperature
        save_knowledge = True
        train_rmse_file = open(os.path.join(self.directory, 'train_rmse_{:.4f}.csv'.format(self.temperature)), 'w')
        test_rmse_file = open(os.path.join(self.directory, 'test_rmse_{:.4f}.csv'.format(self.temperature)), 'w')
        if opt.problem_type == 'classification':
            train_acc_file = open(os.path.join(self.directory, 'train_acc_{:.4f}.csv'.format(self.temperature)), 'w')
            test_acc_file = open(os.path.join(self.directory, 'test_acc_{:.4f}.csv'.format(self.temperature)), 'w')
        
        weights_file = open(os.path.join(self.directory, 'weights_{:.4f}.csv'.format(self.temperature)), 'w')

        # Initialize MCMC
        print(f'Initialize MCMC, chain: {self.temperature:.2f}')
        self.start_time = time.time()

        print(f'Initialize MCMC, chain: {self.temperature:.2f}')
        self.start_time = time.time()

        train_size = self.train_data.shape[0]
        test_size = self.test_data.shape[0]
        y_test = self.test_data[:, self.topology[0]: self.topology[0] + self.topology[2]]
        y_train = self.train_data[:, self.topology[0]: self.topology[0] + self.topology[2]]
        swarm_initial, best_swarm_pos, best_swarm_err =  self.swarm, self.best_swarm_pos, self.best_swarm_err
        swarm_current = copy.copy(swarm_initial)
        swarm_proposal = copy.copy(swarm_initial)
        weights=[]
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
        rmse_train=[]
        rmse_test=[]

        for i in range(self.pop_size):
            #weights_current[i]=swarm_current[i].position
            prediction_train = self.neural_network.generate_output(self.train_data, weights_current[i])
            prediction_test = self.neural_network.generate_output(self.test_data, weights_current[i])
            eta[i] = np.log(np.var(prediction_train - y_train))
            
            eta_proposal[i] = copy.copy(eta[i])
            tau_proposal[i] = np.exp(eta[i])

            prior[i] = self.prior_function(weights_current[i], tau_proposal[i],self.temperature)
            likelihood[i], rmse_train_current[i], accuracy_train_current[i] = self.likelihood_function(self.neural_network, self.train_data, weights_current[i].copy(), tau_proposal[i].copy(),self.temperature)

            rmse_test_current[i],accuracy_test_current[i]=self.neural_network.classification_perf(weights_current[i], 'test',opt.problem_type)

        num_accept = 0
        sample=0

        while(sample<100):
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
                    rmse_train.append(rmse_train_current[i])
                    np.savetxt(test_rmse_file, [rmse_test_current[i]])
                    rmse_test.append(rmse_test_current[i])
                    
                    np.savetxt(weights_file, [weights_current[i]])
                    if(opt.problem_type== 'classification'):
                        np.savetxt(train_acc_file, [accuracy_train_current[i]])
                        np.savetxt(test_acc_file, [accuracy_test_current[i]])
                    
                
                #accuracy_test_check,_=self.neural_network.classification_perf(weights_current,'test')
                if(opt.problem_type=='regression'):
                    print(f'Sample: {sample}  RMSE Train: {rmse_train_current[i]:.4f}  RMSE test: {rmse_test_current[i]:.4f}')
                else:
                    print(f'Sample: {sample}  accuracy train: {accuracy_train_current[i]:.2f}  accuracy test: {accuracy_test_current[i]:.2f} RMSE Train: {rmse_train_current[i]:.4f} ')
                

                elapsed_time = ":".join(Replica.convert_time(time.time() - self.start_time))
                
                sample+=1
            
            
        for i in range(self.pop_size):
            if self.swarm[i].error < self.swarm[i].best_part_err:
                self.swarm[i].best_part_err = self.swarm[i].error.copy()
                self.swarm[i].best_part_pos = copy.copy(self.swarm[i].position)

            if self.swarm[i].error < self.best_swarm_err:
                self.best_swarm_err = self.swarm[i].error.copy()
                self.best_swarm_pos = copy.copy(self.swarm[i].position)
            else:
                pass

        
        while(sample<self.num_samples):
            
            step=sample/self.pop_size
            w_stepsize=self.decayed_learning_rate(step)
            e_stepsize=w_stepsize
            swarm_new,new_best_pos,new_best_err=self.evolve(copy.copy(self.swarm), self.best_swarm_pos.copy(), self.best_swarm_err.copy())

            #self.swarm, self.best_swarm_pos, self.best_swarm_err = self.evolve(copy.copy(self.swarm), self.best_swarm_pos.copy(), self.best_swarm_err.copy())
            for i in range(self.pop_size):

                weights_proposal[i]=swarm_new[i].position + np.random.normal(0, w_stepsize, self.w_size)
                weights_proposal_evolve[i]=self.evolve_single_particle(weights_proposal[i].copy(),copy.copy(swarm_new),new_best_pos.copy(), new_best_err.copy(),i)



            for i in range(self.pop_size):
                eta_proposal[i] = eta[i] + np.random.normal(0, e_stepsize, 1)
                tau_proposal[i] = np.exp(eta_proposal[i])
                wc_delta = (weights_current[i]- weights_proposal_evolve[i]) 
                wp_delta = (weights_proposal[i] - swarm_new[i].position)
                
                sigma_sq = w_stepsize*w_stepsize

                first = -0.5 * np.sum(wc_delta  *  wc_delta  ) / sigma_sq  # this is wc_delta.T  *  wc_delta /sigma_sq
                second = -0.5 * np.sum(wp_delta * wp_delta ) / sigma_sq

                diff_prop =  (first - second)

                

                accept, rmse_train_current[i], rmse_test_current[i], accuracy_train_current[i], accuracy_test_current[i], likelihood[i], prior[i] = self.evaluate_proposal(self.neural_network, self.train_data, self.test_data, weights_proposal[i], tau_proposal[i], likelihood[i], prior[i],diff_prop)

                if accept:
                    num_accept += 1
                    weights_current[i] = weights_proposal[i]
                    self.swarm[i].position=weights_current[i]
                    self.swarm[i].error=rmse_train_current[i]
                    eta[i] = eta_proposal[i]

                if save_knowledge:
                    np.savetxt(train_rmse_file, [rmse_train_current[i]])
                    rmse_train.append(rmse_train_current[i])
                    np.savetxt(test_rmse_file, [rmse_test_current[i]])
                    rmse_test.append(rmse_test_current[i])
                    np.savetxt(weights_file, [weights_current[i]])
                    if(opt.problem_type== 'classification'):
                        np.savetxt(train_acc_file, [accuracy_train_current[i]])
                        np.savetxt(test_acc_file, [accuracy_test_current[i]])

                #accuracy_test_check,_=self.neural_network.classification_perf(weights_current,'test')
                '''if(opt.problem_type=='regression'):
                    print(f'Sample: {sample}  RMSE Train: {rmse_train_current[i]:.4f}  RMSE test: {rmse_test_current[i]:.4f} accept r: {num_accept/(sample+1):.4f}')
                else:
                    print(f'Sample: {sample}  accuracy train: {accuracy_train_current[i]:.2f}  accuracy test: {accuracy_test_current[i]:.2f} RMSE Train: {rmse_train_current[i]:.4f} accept r: {num_accept/(sample+1):.4f}')'''
                
                if(opt.problem_type=='regression'):
                    print(f'Temperature: {self.temperature:.2f} Sample: {sample} RMSE Train: {rmse_train_current[i]:.4f}  RMSE test: {rmse_test_current[i]:.4f} ')
                else:
                    print(f'Temperature: {self.temperature:.2f} Sample: {sample}  accuracy train: {accuracy_train_current[i]:.2f}  accuracy test: {accuracy_test_current[i]:.2f} ')
                
                sample+=1
                elapsed_time = ":".join(Replica.convert_time(time.time() - self.start_time))
            for i in range(self.pop_size):
                if self.swarm[i].error < self.swarm[i].best_part_err:
                    self.swarm[i].best_part_err = self.swarm[i].error.copy()
                    self.swarm[i].best_part_pos = copy.copy(self.swarm[i].position)

                if self.swarm[i].error < self.best_swarm_err:
                    self.best_swarm_err = self.swarm[i].error.copy()
                    self.best_swarm_pos = copy.copy(self.swarm[i].position)
                else:
                    pass

            if(sample>self.num_samples/2):
                self.temperature=1
            index_best=0
            index_worst=0
            rmse_worst=0
            rmse_best=1000
            for i in range(self.pop_size):
                if(self.swarm[i].error<rmse_best):
                    index_best=i
                    rmse_best=self.swarm[i].error
                if(self.swarm[i].error>rmse_worst):
                    index_worst=i
                    rmse_worst=self.swarm[i].error

            #if(sample>self.num_samples/2):
                #continue
            swapped=False        
            param = np.concatenate([weights_current[index_best], np.asarray([eta[index_best]]).reshape(1), np.asarray([likelihood[index_best]*self.temperature]),np.asarray([likelihood[index_worst]*self.temperature]),np.asarray([self.temperature]),np.asarray([prior[index_best]*self.temperature]).reshape(1),np.asarray([swapped]).reshape(1),self.swarm[index_best].best_part_pos])
            #swapped=False
            self.parameter_queue.put(param)
            self.event.clear()
            self.signal_main.set()
        #     print(f'Temperature: {self.temperature:.2f} Current sample: {sample} out of {self.num_samples} is num with {self.swap_sample.value} as next swap')
            
        #     # Wait for signal from Master
            self.event.wait()

            result = self.parameter_queue.get()
            swapped=result[self.w_size+5]
            if(swapped==True):
                self.swarm[index_worst].position=result[:self.w_size]
                self.swarm[index_worst].error=self.neural_network.evaluate_fitness(self.swarm[index_worst].position,opt.problem_type)# curr error
                self.swarm[index_worst].best_part_pos=result[self.w_size+6:]
                weights_current[index_worst]=result[:self.w_size]
                eta[index_worst]=result[self.w_size]
                likelihood[index_worst]=result[self.w_size+1]/self.temperature
                prior[index_worst]=result[self.w_size+4]/self.temperature
        
            for i in range(self.pop_size):
                if self.swarm[i].error < self.swarm[i].best_part_err:
                    self.swarm[i].best_part_err = self.swarm[i].error.copy()
                    self.swarm[i].best_part_pos = copy.copy(self.swarm[i].position)

                if self.swarm[i].error < self.best_swarm_err:
                    self.best_swarm_err = self.swarm[i].error.copy()
                    self.best_swarm_pos = copy.copy(self.swarm[i].position)
                else:
                    pass
            # print("Temperature: {} Sample: {:d}, Best Fitness: {:.4f}, Proposal: {:.4f}, Time Elapsed: {:s}".format(self.temperature, sample, rmse_train_current, rmse_train, elapsed_time))

        elapsed_time = time.time() - self.start_time
        accept_ratio = num_accept/self.num_samples
        #print("Written {} values for Accuracies".format(writ))
        # Close the files
        train_rmse_file.close()
        test_rmse_file.close()
        weights_file.close()
        if opt.problem_type == 'classification':
            train_acc_file.close()
            test_acc_file.close()
        with self.active_chains.get_lock():
            self.active_chains.value -= 1
        with self.num_accepted.get_lock():
            self.num_accepted.value += num_accept
        print(f"Temperature: {self.temperature} done, {sample+1} samples sampled out of {self.num_samples}. Number of active chains: {self.active_chains.value}")
        best=np.ones((self.pop_size,))
        for i in range(0,self.pop_size):
            best[i]=self.swarm[i].best_part_err
        best=best.argsort()[-6:][::-1]
        color=['red', 'green', 'blue', 'yellow','brown','black']
        
        for i in range(6):
            temp_train=rmse_train[best[i]::self.pop_size]
            x=np.linspace(0,self.num_samples,len(temp_train))
            plt.plot(x,temp_train,color=color[i],label=str(i+1))
            
        plt.title(label='Train RMSE for top 6 particles for temperature '+str(curr_temp))
        plt.legend(loc='best')
        plt.savefig(self.directory+'/'+str(curr_temp)+'.png')

class EvoPT(object):

    def __init__(self, opt, path, geometric=True):
        #FNN Chain variables
        self.opt = opt
        self.train_data = opt.train_data
        self.test_data = opt.test_data
        self.topology = list(map(int,opt.topology.split(',')))
        self.num_param = (self.topology[0] * self.topology[1]) + (self.topology[1] * self.topology[2]) + self.topology[1] + self.topology[2]   
        self.problem_type = opt.problem_type
        #Parallel Tempering variables
        self.burn_in = opt.burn_in
        self.swap_interval = opt.swap_interval
        self.path = path
        self.max_temp = opt.max_temp
        self.num_swap = 0
        self.total_swap_proposals = 0
        self.num_chains = opt.num_chains
        self.chains = []
        self.temperatures = []
        self.num_samples = int(opt.num_samples/self.num_chains)
        self.geometric = geometric
        self.population_size = opt.population_size
        # create queues for transfer of parameters between process chain
        self.active_chains = Value('d', lock=True)
        self.swap_sample = Value('d', lock=True)
        self.num_accepted = Value('d', lock=True)
        self.parameter_queue = [multiprocessing.Queue() for i in range(self.num_chains)]
        self.chain_queue = multiprocessing.JoinableQueue()
        self.wait_chain = [multiprocessing.Event() for i in range (self.num_chains)]
        self.event = [multiprocessing.Event() for i in range (self.num_chains)]
        make_directory(path)
        self.directory=path

    def default_beta_ladder(self, ndim, ntemps, Tmax): #https://github.com/konqr/ptemcee/blob/master/ptemcee/sampler.py
        """
        Returns a ladder of :math:`\beta \equiv 1/T` under a geometric spacing that is determined by the
        arguments ``ntemps`` and ``Tmax``.  The temperature selection algorithm works as follows:
        Ideally, ``Tmax`` should be specified such that the tempered posterior looks like the prior at
        this temperature.  If using adaptive parallel tempering, per `arXiv:1501.05823
        <http://arxiv.org/abs/1501.05823>`_, choosing ``Tmax = inf`` is a safe bet, so long as
        ``ntemps`` is also specified.
        :param ndim:
            The number of dimensions in the parameter space.
        :param ntemps: (optional)
            If set, the number of temperatures to generate.
        :param Tmax: (optional)
            If set, the maximum temperature for the ladder.
        Temperatures are chosen according to the following algorithm:
        * If neither ``ntemps`` nor ``Tmax`` is specified, raise an exception (insufficient
          information).
        * If ``ntemps`` is specified but not ``Tmax``, return a ladder spaced so that a Gaussian
          posterior would have a 25% temperature swap acceptance ratio.
        * If ``Tmax`` is specified but not ``ntemps``:
          * If ``Tmax = inf``, raise an exception (insufficient information).
          * Else, space chains geometrically as above (for 25% acceptance) until ``Tmax`` is reached.
        * If ``Tmax`` and ``ntemps`` are specified:
          * If ``Tmax = inf``, place one chain at ``inf`` and ``ntemps-1`` in a 25% geometric spacing.
          * Else, use the unique geometric spacing defined by ``ntemps`` and ``Tmax``.
        """

        if type(ndim) != int or ndim < 1:
            raise ValueError('Invalid number of dimensions specified.')
        if ntemps is None and Tmax is None:
            raise ValueError('Must specify one of ``ntemps`` and ``Tmax``.')
        if Tmax is not None and Tmax <= 1:
            raise ValueError('``Tmax`` must be greater than 1.')
        if ntemps is not None and (type(ntemps) != int or ntemps < 1):
            raise ValueError('Invalid number of temperatures specified.')

        tstep = np.array([25.2741, 7., 4.47502, 3.5236, 3.0232,
                          2.71225, 2.49879, 2.34226, 2.22198, 2.12628,
                          2.04807, 1.98276, 1.92728, 1.87946, 1.83774,
                          1.80096, 1.76826, 1.73895, 1.7125, 1.68849,
                          1.66657, 1.64647, 1.62795, 1.61083, 1.59494,
                          1.58014, 1.56632, 1.55338, 1.54123, 1.5298,
                          1.51901, 1.50881, 1.49916, 1.49, 1.4813,
                          1.47302, 1.46512, 1.45759, 1.45039, 1.4435,
                          1.4369, 1.43056, 1.42448, 1.41864, 1.41302,
                          1.40761, 1.40239, 1.39736, 1.3925, 1.38781,
                          1.38327, 1.37888, 1.37463, 1.37051, 1.36652,
                          1.36265, 1.35889, 1.35524, 1.3517, 1.34825,
                          1.3449, 1.34164, 1.33847, 1.33538, 1.33236,
                          1.32943, 1.32656, 1.32377, 1.32104, 1.31838,
                          1.31578, 1.31325, 1.31076, 1.30834, 1.30596,
                          1.30364, 1.30137, 1.29915, 1.29697, 1.29484,
                          1.29275, 1.29071, 1.2887, 1.28673, 1.2848,
                          1.28291, 1.28106, 1.27923, 1.27745, 1.27569,
                          1.27397, 1.27227, 1.27061, 1.26898, 1.26737,
                          1.26579, 1.26424, 1.26271, 1.26121,
                          1.25973])

        if ndim > tstep.shape[0]:
            # An approximation to the temperature step at large
            # dimension
            tstep = 1.0 + 2.0*np.sqrt(np.log(4.0))/np.sqrt(ndim)
        else:
            tstep = tstep[ndim-1]

        appendInf = False
        if Tmax == np.inf:
            appendInf = True
            Tmax = None
            ntemps = ntemps - 1

        if ntemps is not None:
            if Tmax is None:
                # Determine Tmax from ntemps.
                Tmax = tstep ** (ntemps - 1)
        else:
            if Tmax is None:
                raise ValueError('Must specify at least one of ``ntemps'' and '
                                 'finite ``Tmax``.')

            # Determine ntemps from Tmax.
            ntemps = int(np.log(Tmax) / np.log(tstep) + 2)

        betas = np.logspace(0, -np.log10(Tmax), ntemps)
        if appendInf:
            # Use a geometric spacing, but replace the top-most temperature with
            # infinity.
            betas = np.concatenate((betas, [0]))

        return betas

    def assign_temperatures(self):
        #Geometric Spacing
        if self.geometric is True:
            betas = self.default_beta_ladder(2, ntemps=self.num_chains, Tmax=self.max_temp)
            self.temperatures = [np.inf if beta == 0 else 1.0/beta for beta in betas]
        #Linear Spacing
        else:
            temp = 2
            for i in range(0,self.num_chains):
                self.temperatures.append(temp)
                temp += (self.max_temp/self.num_chains) #2.5
                print (self.temperatures[i])

    def initialize_chains(self):
        self.assign_temperatures()
        with self.swap_sample.get_lock():
            self.swap_sample.value = self.swap_interval
        for chain in range(0, self.num_chains):
            weights = np.random.randn(self.num_param)
            self.chains.append(Replica(self.num_samples, self.burn_in, self.population_size, self.topology, self.train_data, self.test_data, self.path, self.temperatures[chain], self.swap_sample, self.parameter_queue[chain], self.problem_type, main_process=self.wait_chain[chain], event=self.event[chain], active_chains=self.active_chains,  num_accepted=self.num_accepted, swap_interval=self.swap_interval))

    def swap_procedure(self, parameter_queue_1, parameter_queue_2):
        swapped=False
        if not parameter_queue_2.empty() and not parameter_queue_1.empty():
            
            param_1 = parameter_queue_1.get()
            param_2 = parameter_queue_2.get()
            w_1 = param_1[0:self.num_param]
            eta_1 = param_1[self.num_param]
            likelihood_1 = param_1[self.num_param+1]
            likelihood_swap_1=param_1[self.num_param+2]
            T_1 = param_1[self.num_param+3]
            prior_1=param_1[self.num_param+4]
            best_part_1=param_1[self.num_param+6:]
            w_2 = param_2[0:self.num_param]
            eta_2 = param_2[self.num_param]
            likelihood_2 = param_2[self.num_param+1]
            likelihood_swap_2=param_2[self.num_param+2]
            T_2 = param_2[self.num_param+3]
            prior_2 = param_2[self.num_param+4]
            best_part_2=param_2[self.num_param+6:]
            print(likelihood_1,likelihood_2)
            print(prior_1,prior_2)

            #SWAPPING PROBABILITIES
            try:
                swap_proposal =  min(1,0.5*np.exp(likelihood_2 - likelihood_swap_1))
            except OverflowError:
                swap_proposal = 1
            u = np.random.uniform(0,1)
            if u < swap_proposal:
                
                #swapped +=1
                self.num_swap += 1
                param_1[0:self.num_param]=w_2
                param_1[self.num_param]=eta_2
                param_1[self.num_param+1]=likelihood_2
                param_1[self.num_param+4]=prior_2
                param_1[self.num_param+6:]=best_part_2
                swapped=True
                param_1[self.num_param+5]=True
                print("Swapped {}, {}".format(T_1, T_2))


            else:
                print(swap_proposal)
                print("No swapping!!")
                #swapped = False
            self.total_swap_proposals += 1

            try:
                swap_proposal =  min(1,0.5*np.exp(likelihood_1 - likelihood_swap_2))
            except OverflowError:
                swap_proposal = 1
            u = np.random.uniform(0,1)
            if u < swap_proposal: 
                #swapped +=1
                self.num_swap += 1
                param_2[0:self.num_param]=w_1
                param_2[self.num_param]=eta_1
                param_2[self.num_param+1]=likelihood_1
                param_2[self.num_param+4]=prior_1
                param_2[self.num_param+6:]=best_part_1
                print("Swapped {}, {}".format(T_1, T_2))
                swapped=True
                param_2[self.num_param+5]=True


            else:
                print(swap_proposal)
                print("No swapping!!")
                #swapped = False
            self.total_swap_proposals += 1
            return param_1, param_2, swapped
        else:
            print("No Swapping occured")
            self.total_swap_proposals += 1
            raise Exception('empty queue')
            return

    def run_chains(self):
        start_time = time.time()
        x_test = np.linspace(0,1,num=self.test_data.shape[0])
        x_train = np.linspace(0,1,num=self.train_data.shape[0])
        
        # only adjacent chains can be swapped therefore, the number of proposals is ONE less num_chains
        swap_proposal = np.ones(self.num_chains-1)

        # create parameter holders for paramaters that will be swapped
        replica_param = np.zeros((self.num_chains, self.num_param))
        likelihood = np.zeros(self.num_chains)
        eta = np.zeros(self.num_chains)

        for index in range(self.num_chains):
            self.chains[index].start()
            with self.active_chains.get_lock():
                self.active_chains.value += 1

        swaps_appected_main = 0
        total_swaps_main = 0

        #SWAP PROCEDURE
        while self.active_chains.value > 0:
            print("Waiting for Swap Signals...")
            signal_count = 0
            for index in range(0,self.num_chains):
                print("Waiting for chain: {}".format(index+1))
                while True:
                    flag = self.wait_chain[index].wait(timeout=2)
                    if flag:
                        print("Signal from chain: {}".format(index+1))
                        signal_count += 1
                        break
                    elif self.active_chains.value <= 0:
                        break

            with self.swap_sample.get_lock():
                self.swap_sample.value += self.swap_interval

            if signal_count != self.num_chains:
                print("Skipping the swap!")
                continue

            for index in range(0,self.num_chains-1):
                try:
                    param_1, param_2, swapped = self.swap_procedure(self.parameter_queue[index], self.parameter_queue[index+1])
                    self.parameter_queue[index].put(param_1)
                    self.parameter_queue[index+1].put(param_2)
                    if index == 0:
                        if swapped:
                            swaps_appected_main += 1
                        total_swaps_main += 1
                except Exception as e:
                    print(e)
                    print("Nothing Returned by swap method!")
            for index in range (self.num_chains):
                    self.event[index].set()
                    self.wait_chain[index].clear()

        print("Joining processes")

        #JOIN THEM TO MAIN PROCESS
        for index in range(0,self.num_chains):
            self.chains[index].join()
            print("Chain {} joined".format(index))
        self.chain_queue.join()
        print("Done")

        #GETTING DATA
        burn_in = int(self.num_samples*self.burn_in)
        rmse_train = np.zeros((self.num_chains,self.num_samples - burn_in))
        weights=[]
        rmse_test = np.zeros((self.num_chains,self.num_samples - burn_in))
        if self.problem_type == 'classification':
            acc_train = np.zeros((self.num_chains,self.num_samples - burn_in))
            acc_test = np.zeros((self.num_chains,self.num_samples - burn_in))
        accept_ratio = np.zeros((self.num_chains,1))
        weights_file = os.path.join(self.path, 'weights_{:.4f}.csv'.format(self.temperatures[0]))
        weights1=np.genfromtxt(weights_file, delimiter=' ')
        weights=weights1[burn_in:,:]
        for i in range(1,self.num_chains):
            weights_file = os.path.join(self.path, 'weights_{:.4f}.csv'.format(self.temperatures[i]))
            weights1=np.genfromtxt(weights_file, delimiter=' ')
            weights2=weights1[burn_in:,:]
            weights=np.concatenate((weights,weights2),axis=0)

        for i in range(self.num_chains):

            file_name = os.path.join(self.path, 'test_rmse_{:.4f}.csv'.format(self.temperatures[i]))
            #weights_file = os.path.join(self.path, 'weights_{:.4f}.csv'.format(self.temperatures[i]))
            dat = np.genfromtxt(file_name, delimiter=',')
            #weights1=np.genfromtxt(weights_file, delimiter=',')
            rmse_test[i,:] = dat[burn_in:]
            #weights2=weights1[burn_in:]
            #weights.append(weights2)
            file_name = os.path.join(self.path, 'train_rmse_{:.4f}.csv'.format(self.temperatures[i]))
            dat = np.genfromtxt(file_name, delimiter=',')
            rmse_train[i,:] = dat[burn_in:]

            if self.problem_type == 'classification':
                file_name = os.path.join(self.path, 'test_acc_{:.4f}.csv'.format(self.temperatures[i]))
                dat = np.genfromtxt(file_name, delimiter=',')
                acc_test[i,:] = dat[burn_in:]

                file_name = os.path.join(self.path, 'train_acc_{:.4f}.csv'.format(self.temperatures[i]))
                dat = np.genfromtxt(file_name, delimiter=',')
                acc_train[i,:] = dat[burn_in:]
        
        self.rmse_train = rmse_train.reshape(self.num_chains*(self.num_samples - burn_in), 1)
        self.rmse_test = rmse_test.reshape(self.num_chains*(self.num_samples - burn_in), 1)
        if self.problem_type == 'classification':
            self.acc_train = acc_train.reshape(self.num_chains*(self.num_samples - burn_in), 1)
            self.acc_test = acc_test.reshape(self.num_chains*(self.num_samples - burn_in), 1)

        # PLOT THE RESULTS
        #self.plot_figures()

        with open(os.path.join(self.path, 'results.txt'), 'w') as results_file:
            results_file.write(f'NUMBER OF SAMPLES PER CHAIN: {self.num_samples}\n')
            results_file.write(f'NUMBER OF CHAINS: {self.num_chains}\n')
            results_file.write(f'NUMBER OF SWAPS MAIN: {total_swaps_main}\n')
            results_file.write(f'PERCENT SAMPLES ACCEPTED: {self.num_accepted.value/(self.num_samples*self.num_chains)*100:.2f}\n')
            results_file.write(f'SWAP ACCEPTANCE: {self.num_swap*100/self.total_swap_proposals:.4f} %\n')
            results_file.write(f'SWAP ACCEPTANCE MAIN: {swaps_appected_main*100/total_swaps_main:.2f} %\n')
            if self.problem_type == 'classification':
                results_file.write(f'ACCURACY: \n\tTRAIN -> MEAN: {self.acc_train.mean():.2f} STD: {self.acc_train.std():.2f} BEST: {self.acc_train.max():.2f} \n\tTEST -> MEAN: {self.acc_test.mean():.2f} STD: {self.acc_test.std():.2f} BEST: {self.acc_test.max():.2f}\n')
            results_file.write(f'RMSE: \n\tTRAIN -> MEAN: {self.rmse_train.mean():.5f} STD: {self.rmse_train.std():.5f} BEST: {self.rmse_train.min():.5f} \n\tTEST -> MEAN: {self.rmse_test.mean():.5f} STD: {self.rmse_test.std():.5f} BEST: {self.rmse_test.min():.5f}\n')
            results_file.write(f'TIME TAKEN: {time.time()-start_time:.2f} secs\n')

            
        
        print("NUMBER OF SWAPS MAIN =", total_swaps_main)
        print("SWAP ACCEPTANCE = ", self.num_swap*100/self.total_swap_proposals," %")
        print("SWAP ACCEPTANCE MAIN = ", swaps_appected_main*100/total_swaps_main," %")
        print(np.shape(weights))
        filename=self.path+'/weights.csv'
        np.savetxt(fname=filename,X=weights)

        if self.problem_type == 'classification':
            return (self.rmse_train, self.rmse_test, self.acc_train, self.acc_test,self.num_swap)
        return (self.rmse_train, self.rmse_test,self.num_swap)

    def plot_figures(self):
        
        # X-AXIS 
        x = np.linspace(0, 1, len(self.rmse_train))

        # PLOT TRAIN RMSE
        plt.plot(x, self.rmse_train, label='rmse_train')
        plt.xlabel('samples')
        plt.ylabel('RMSE')
        plt.title('{} RMSE Train'.format(self.opt.problem.upper()))
        plt.legend()
        plt.savefig(os.path.join(self.path, 'rmse_train.png'))
        plt.clf()

        # PLOT TEST RMSE
        plt.plot(x, self.rmse_test, label='rmse_test')
        plt.xlabel('samples')
        plt.ylabel('RMSE')
        plt.title('{} RMSE Test'.format(self.opt.problem.upper()))
        plt.legend()
        plt.savefig(os.path.join(self.path, 'rmse_test.png'))
        plt.clf()

        if self.problem_type == 'classification':
            # PLOT TRAIN ACCURACY
            plt.plot(x, self.acc_train, label='acc_train')
            plt.xlabel('samples')
            plt.ylabel('Accuracy %')
            plt.title('{} Accurace Train'.format(self.opt.problem.upper()))
            plt.legend()
            plt.savefig(os.path.join(self.path, 'acc_train.png'))
            plt.clf()

            # PLOT TEST ACCURACY
            plt.plot(x, self.acc_test, label='acc_test')
            plt.xlabel('samples')
            plt.ylabel('Accuracy %')
            plt.title('{} Accurace Test'.format(self.opt.problem.upper()))
            plt.legend()
            plt.savefig(os.path.join(self.path, 'acc_test.png'))
            plt.clf()

def make_directory(path):
    if not os.path.isdir(path):
        os.makedirs(path)

if __name__ == '__main__': 

    # CREATE RESULTS DIRECTORY
    results_dir = os.path.join(opt.root, 'results', '{}_{}'.format(opt.problem, opt.run_id))
    make_directory(results_dir)
    logfile = os.path.join(results_dir, 'params.json')
    with open(logfile, 'w') as stream:
        json.dump(vars(opt), stream)

    # READ DATA
    data_path = os.path.join(opt.root, 'datasets')
    print(os.path.join(data_path, opt.train_data), os.path.exists(os.path.join(data_path, opt.train_data)))

    opt.train_data = np.genfromtxt(os.path.join(data_path, opt.train_data), delimiter=',',dtype=None)
    opt.test_data = np.genfromtxt(os.path.join(data_path, opt.test_data),  delimiter=',',dtype=None)

    # CREATE EVOLUTIONARY PT CLASS
    evo_pt = EvoPT(opt, results_dir)
    
    # INITIALIZE PT CHAINS
    np.random.seed(int(time.time()))
    evo_pt.initialize_chains()

    # RUN EVO PT
    if opt.problem_type == 'classification':
        rmse_train, rmse_test, acc_train, acc_test,accept_ratio = evo_pt.run_chains()
        print(np.mean(acc_train),np.std(acc_train),np.max(acc_train), np.mean(acc_test), np.std(acc_test),np.max(acc_test),accept_ratio)
    elif opt.problem_type == 'regression':
        rmse_train, rmse_test ,accept_ratio= evo_pt.run_chains()
        print(np.mean(rmse_train),np.std(rmse_train),np.min(rmse_train), np.mean(rmse_test), np.std(rmse_test),np.min(rmse_test),accept_ratio)
