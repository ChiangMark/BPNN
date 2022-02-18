import numpy as np 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import csv
import time as time
import math

class BPNN(object):
    def __init__(self):
        self.x1_min, self.x1_max = -5, 5
        self.x2_min, self.x2_max = -5, 5
        self.w_min, self.w_max = -0.3, 0.3 

        self.num_data = 400
        self.num_train_data = 300
        self.num_test_data = 100

        self.hidden_neuron = np.array([5, 10, 15])
        self.L_rate = np.array([0.1, 0.5, 5])
        self.alpha = np.array([0, 1])

        self.all_x1 = np.linspace(self.x1_min, self.x1_max, int((self.x1_max - self.x1_min)*1000) + 1)
        self.all_x2 = np.linspace(self.x2_min, self.x2_max, int((self.x2_max - self.x2_min)*1000) + 1)
        self.X1, self.X2 = np.meshgrid(self.all_x1, self.all_x2)

        self.Y = self.function(self.X1, self.X2)
        self.Y_max = np.max(self.Y)
        self.Y_min = np.min(self.Y)
        

    def random_data(self, number):
        self.x1 = self.x1_min + (self.x1_max - self.x1_min)*np.random.rand(self.num_data,1)
        self.x2 = self.x2_min + (self.x2_max - self.x2_min)*np.random.rand(self.num_data,1)

        self.w_hidden = self.w_min + (self.w_max - self.w_min)*np.random.rand(3, self.hidden_neuron[-1])
        self.b = self.w_min + (self.w_max - self.w_min)*np.random.rand(1, self.hidden_neuron[-1] + 1)

        self.x1_train = self.x1[:self.num_train_data]
        self.x2_train = self.x2[:self.num_train_data]
        self.x1_test = self.x1[self.num_train_data:]
        self.x2_test = self.x2[self.num_train_data:]

        with open("Data/x-{}.csv".format(number), "w", newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(self.x1)

        with open("Data/x2-{}.csv".format(number), "w", newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(self.x2)

        with open("Data/w-{}.csv".format(number), "w", newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(self.w_hidden)

        with open("Data/b-{}.csv".format(number), "w", newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(self.b)

        self.normalization()

        f_train = plt.figure(1)
        ax = Axes3D(f_train)
        plt.title("train-{}".format(number))
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        ax.set_zlabel("y")
        ax.scatter(self.x1_train, self.x2_train, self.d[:self.num_train_data])
        plt.savefig("Picture/train-{}.png".format(number))

        f_test = plt.figure(2)
        ax = Axes3D(f_test)
        plt.title("test-{}".format(number))
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        ax.set_zlabel("y")
        ax.scatter(self.x1_test, self.x2_test, self.d[self.num_train_data:])
        plt.savefig("Picture/test-{}.png".format(number))

        with open("Data/d-{}.csv".format(number), "w", newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(self.d)

    def initialize_data(self, Hidden_N):
        self.w_input1 = self.w_hidden.copy()[0,:self.hidden_neuron[Hidden_N]].reshape(1, self.hidden_neuron[Hidden_N])
        self.w_input2 = self.w_hidden.copy()[1,:self.hidden_neuron[Hidden_N]].reshape(1, self.hidden_neuron[Hidden_N])
        self.w_output = self.w_hidden.copy()[2,:self.hidden_neuron[Hidden_N]].reshape(1, self.hidden_neuron[Hidden_N]).T

        self.bias = self.b.copy()[0, :self.hidden_neuron[Hidden_N]].reshape(1, self.hidden_neuron[Hidden_N])
        self.bias_out = self.b.copy()[0, self.hidden_neuron[Hidden_N]].reshape(1,1)

    def function(self, x1, x2):
        y = pow(x1, 2) + pow(x2, 2)
        return y

    def activation_funation(self, V):
        phi_V = 1/(1 + np.exp(-V))
        return phi_V

    def normalization(self):
        self.d = self.function(self.x1, self.x2)
        self.norm_d = (self.d - self.Y_min)*(0.8 - 0.2)/(self.Y_max - self.Y_min) + 0.2
        self.y_train = self.norm_d[:self.num_train_data]
        self.y_test = self.norm_d[self.num_train_data:]

    def denormalization(self, norm):
        self.denorm = (norm - 0.2)*(self.Y_max - self.Y_min)/(0.8 - 0.2) + self.Y_min
        return self.denorm
        
    def train(self, number, Hidden_N, Learning_Rate, Momentum_Alpha):
        self.initialize_data(Hidden_N)
        result = []
        self.Eav = []
        self.E = 1
        self.epoch = 0
        self.last_delta_w_output = np.zeros([self.hidden_neuron[Hidden_N], 1])
        self.last_delta_w_input1 = np.zeros([1, self.hidden_neuron[Hidden_N]])
        self.last_delta_w_input2 = np.zeros([1, self.hidden_neuron[Hidden_N]])
        self.last_delta_bias_out = np.zeros([1, 1])
        self.last_delta_bias = np.zeros([1, self.hidden_neuron[Hidden_N]])
        time_start = time.time()

        while self.E > 0.0001:
            self.bias_ = np.tile(self.bias, [self.num_train_data, 1])
            self.bias_out_ = np.tile(self.bias_out, [self.num_train_data, 1])

            self.V_hidden = np.dot(self.x1_train, self.w_input1) + np.dot(self.x2_train, self.w_input2) + self.bias_
            self.y_hidden = self.activation_funation(self.V_hidden)

            self.V_output = np.dot(self.y_hidden, self.w_output) + self.bias_out_
            self.y_output = self.activation_funation(self.V_output)

            self.e = self.y_train - self.y_output

            self.w_output_ = np.tile(self.w_output, [1, self.num_train_data]).T
            self.x1_train_ = np.tile(self.x1_train, [1, self.hidden_neuron[Hidden_N]])
            self.x2_train_ = np.tile(self.x2_train, [1, self.hidden_neuron[Hidden_N]])
            self.Delta_output = self.e*self.y_output*(1 - self.y_output)
            self.Delta_hidden = self.Delta_output*self.w_output_*self.y_hidden*(1 - self.y_hidden)
            self.delta_w_output_ = self.L_rate[Learning_Rate]*self.Delta_output*self.y_hidden
            self.delta_w_input1_ = self.L_rate[Learning_Rate]*self.Delta_hidden*self.x1_train_
            self.delta_w_input2_ = self.L_rate[Learning_Rate]*self.Delta_hidden*self.x2_train_
            self.delta_bias_out_ = self.L_rate[Learning_Rate]*self.Delta_output*1
            self.delta_bias_ = self.L_rate[Learning_Rate]*self.Delta_hidden*1

            self.delta_w_output = np.mean(self.delta_w_output_, axis = 0).reshape(self.hidden_neuron[Hidden_N], 1)
            self.delta_w_input1 = np.mean(self.delta_w_input1_, axis = 0)
            self.delta_w_input2 = np.mean(self.delta_w_input2_, axis = 0)
            self.delta_bias_out = np.mean(self.delta_bias_out_, axis = 0)
            self.delta_bias = np.mean(self.delta_bias_, axis = 0)

            self.w_output += self.delta_w_output + self.last_delta_w_output*self.alpha[Momentum_Alpha]
            self.w_input1 += self.delta_w_input1 + self.last_delta_w_input1*self.alpha[Momentum_Alpha]
            self.w_input2 += self.delta_w_input2 + self.last_delta_w_input2*self.alpha[Momentum_Alpha]
            self.bias_out += self.delta_bias_out + self.last_delta_bias_out*self.alpha[Momentum_Alpha]
            self.bias += self.delta_bias + self.last_delta_bias*self.alpha[Momentum_Alpha]

            self.last_delta_w_output = self.delta_w_output.copy()
            self.last_delta_w_input1 = self.delta_w_input1.copy()
            self.last_delta_w_input2 = self.delta_w_input2.copy()
            self.last_delta_bias_out = self.delta_bias_out.copy()
            self.last_delta_bias = self.delta_bias.copy()

            self.E = np.mean(self.e**2)/2
            self.Eav.append(self.E)
            self.epoch += 1
            if self.epoch >= 2000000:
                break
            
        time_end = time.time()
        all_time = time_end - time_start
        self.Epoch = np.arange(0, self.epoch)

        print("{}-HN:{}-LR:{}-α:{}".format(number, self.hidden_neuron[Hidden_N], self.L_rate[Learning_Rate], self.alpha[Momentum_Alpha]))
        print("Training time: {}".format(all_time))
        print("Last Training Eav: {}".format(self.Eav[-1]))
        print("Epoch: {}".format(self.epoch))

        f1 = plt.figure(num="{}-HN:{}-LR:{}-α:{}".format(number, self.hidden_neuron[Hidden_N], self.L_rate[Learning_Rate], self.alpha[Momentum_Alpha]))
        plt.title("trainE-{}-HN:{}-LR:{}-α:{}".format(number, self.hidden_neuron[Hidden_N], self.L_rate[Learning_Rate], self.alpha[Momentum_Alpha]))
        plt.xlabel("epoch")
        plt.ylabel("Eav")
        plt.xlim([0, self.epoch])
        plt.plot(self.Epoch, self.Eav)
        plt.savefig("Picture/trainE-{}-HN={}-LR={}-α={}.png".format(number, self.hidden_neuron[Hidden_N], self.L_rate[Learning_Rate], self.alpha[Momentum_Alpha]))

        f2 = plt.figure(num="{}-HN={}-LR={}-α={}".format(number, self.hidden_neuron[Hidden_N], self.L_rate[Learning_Rate], self.alpha[Momentum_Alpha]))
        ax = Axes3D(f2)
        self.bias_ = np.tile(self.bias, [self.num_train_data, 1])
        self.bias_out_ = np.tile(self.bias_out, [self.num_train_data, 1])
        self.V_hidden = np.dot(self.x1_train, self.w_input1) + np.dot(self.x2_train, self.w_input2) + self.bias_
        self.y_hidden = self.activation_funation(self.V_hidden)
        self.V_output = np.dot(self.y_hidden, self.w_output) + self.bias_out_
        self.y_output = self.activation_funation(self.V_output)
        self.real_y_output = self.denormalization(self.y_output)
        plt.title("train-{}-HN={}-LR={}-α={}".format(number, self.hidden_neuron[Hidden_N], self.L_rate[Learning_Rate], self.alpha[Momentum_Alpha]))
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        ax.set_zlabel("y")
        ax.scatter(self.x1_train, self.x2_train, self.real_y_output)
        plt.savefig("Picture/train-{}-HN={}-LR={}-α={}.png".format(number, self.hidden_neuron[Hidden_N], self.L_rate[Learning_Rate], self.alpha[Momentum_Alpha]))

        f3 = plt.figure(num="{}-HN={}-LR={}-α={}".format(number, self.hidden_neuron[Hidden_N], self.L_rate[Learning_Rate], self.alpha[Momentum_Alpha]))
        ax = Axes3D(f3)
        self.bias__ = np.tile(self.bias, [self.num_test_data, 1])
        self.bias_out__ = np.tile(self.bias_out, [self.num_test_data, 1])
        self.V_hidden = np.dot(self.x1_test, self.w_input1) + np.dot(self.x2_test, self.w_input2) + self.bias__
        self.y_hidden = self.activation_funation(self.V_hidden)
        self.V_output = np.dot(self.y_hidden, self.w_output) + self.bias_out__
        self.y_output_test = self.activation_funation(self.V_output)
        self.real_y_output_test = self.denormalization(self.y_output_test)
        self.e_test = self.y_test - self.y_output_test
        self.E_test = np.mean(self.e_test**2)/2
        print("Test Eav: {}".format(self.E_test))
        print("-------------------------------------------------")
        plt.title("test-{}-HN={}-LR={}-α={}".format(number, self.hidden_neuron[Hidden_N], self.L_rate[Learning_Rate], self.alpha[Momentum_Alpha]))
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        ax.set_zlabel("y")
        ax.scatter(self.x1_test, self.x2_test, self.real_y_output_test)
        plt.savefig("Picture/test-{}-HN={}-LR={}-α={}.png".format(number, self.hidden_neuron[Hidden_N], self.L_rate[Learning_Rate], self.alpha[Momentum_Alpha]))

        result = [[self.epoch, all_time, self.Eav[-1], self.E_test]]

        with open("Result/{}-HN={}-LR={}-α={}.csv".format(number, self.hidden_neuron[Hidden_N], self.L_rate[Learning_Rate], self.alpha[Momentum_Alpha]), "w", newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(result)
        
    def start(self):
        print("start!")
        for i in range(3):
            self.random_data(i)
            for j in range(3):
                for k in range(3):
                    for l in range(2):
                        self.train(i, j, k, l)
        print("finish!")

if __name__ == "_main_":
    NN = BPNN()
    NN.random_data(0)
    NN.train(0, 0, 0, 1)