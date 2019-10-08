#import libraries basic to the need of the assignment
import sys,random
import numpy as np
import math
import random

class assignment:
    def __init__(self, lt):
        self.parsed_list = [lt[i][:-1] for i in range(len(lt))]

        #hold the problem parameters
        self.P = int(self.parsed_list[0])
        self.S = int(self.parsed_list[1])
        self.T = int(self.parsed_list[2])
        self.Z = int(self.parsed_list[3])

        #create a numpy array to hold the matrix of distances
        self.N = self.P * self.S * self.T
        self.distance_matrix = np.zeros((self.N,self.N), dtype=float)
        self.similarity_matrix = np.zeros((self.N,self.N), dtype=float)
        
        #generate the distance_matrix
        for j in range(4,len(self.parsed_list)):
            temp = self.parsed_list[j].split()
            for i in range(len(temp)):
                self.distance_matrix[j-4][i] = float(temp[i])
                self.similarity_matrix[j-4][i] = 1.0 - float(temp[i])

        #create a matrix of arranged presentations
        self.scheduled_curr = np.arange(self.S*self.T*self.P).reshape((self.S,self.T,self.P))
    
    #function that calculates the goodness measure
    def goodnessMeasure(self, sched_curr):
        dist_mat = self.distance_matrix
        simi_mat = self.similarity_matrix

        sum_u_s = 0.0 
        sum_d_s = 0.0

        for i in range(self.S):
            lst = []
            for j in range(self.T):
                lst = sched_curr[i][j]
                for k in range(len(lst)):
                    for l in range(k,len(lst)):
                        if k != l:
                            sum_u_s = sum_u_s + simi_mat[lst[k]][lst[l]]
        
        #compare the lists
        for i in range(self.T):
            extract = np.array([],dtype=int)

            for j in range(self.S):
                extract = np.append(extract,sched_curr[j][i])
            extract = np.reshape(extract,(self.S,self.P))
            
            for k in range(self.S - 1):
                count = k
                list_1 = np.array([],dtype=int)
                list_2 = np.array([],dtype=int)
                
                for l in range(k,self.S):        
                    if k==count:
                        list_1 = np.append(list_1,extract[l])
                    else:
                        list_2 = np.append(list_2,extract[l])
                    count = count + 1

                for u in range(len(list_1)):
                    for v in range(len(list_2)):
                        sum_d_s = sum_d_s + dist_mat[list_1[u]][list_2[v]]
        
        GS_sum = 0.0
        GS_sum = (sum_u_s + (self.Z * sum_d_s))
        return GS_sum

    def simulatedAnnealing(self):
        #variabled for annealing
        sched_curr = self.generateState(self.scheduled_curr)
        sched_next = np.array([],dtype=int)
        T = 10.0
        alpha = 0.95
        del_E = 0.0

        for i in range(1000):
            if T <= 0.0001:
                return sched_curr
            
            sched_next = self.generateState(sched_curr)
            del_E = self.goodnessMeasure(sched_next) - self.goodnessMeasure(sched_curr)

            if del_E > 0:
                sched_curr = sched_next
            else:
                if math.exp(del_E/T) > random.random():
                    sched_curr = sched_next
            T *= alpha

    def hillClimbing(self):
        #randomly selecting the best state in the consecuting schecule
        sched_curr = self.generateState_n(self.scheduled_curr)
        sched_next = np.array([],dtype=int)
        del_E = 0.0

        for i in range(1000):
            sched_next = self.generateState(sched_curr)
            del_E = self.goodnessMeasure(sched_next) - self.goodnessMeasure(sched_curr)
            if del_E > 0:
                sched_curr = sched_next
        return sched_curr

    #randomly generate states
    def generateState(self, sched_gen):
        sched_gen = np.reshape(sched_gen,(self.S*self.T*self.P)).tolist()
        
        random.shuffle(sched_gen)
        
        sched_gen = np.reshape(sched_gen,(self.S,self.T,self.P))
        return sched_gen
    
    #generating random states with an approximate goodness measure
    def generateState_n(self, sched_gen_n):
        sched_gen_n = np.reshape(sched_gen_n,(self.S*self.T*self.P)).tolist()
        ##############################
        ##############################

        #initialize
        new_lst = []
        ori_lst = sched_gen_n
        idx = 0

        for k in range(self.S):
            for j in range(self.T):
                for i in range(self.P - 1):
                    if i == 0:
                        temp = ori_lst[random.randint(0,len(ori_lst)-1)]
                        new_lst.append(temp)
                        ori_lst.remove(temp)
                        idx = new_lst[-1]
                    else:
                        idx = new_lst[0]
            
                    max_init = 0.0
                    max_idx = idx

                    for i in range(len(ori_lst)):
                        if self.similarity_matrix[idx][ori_lst[i]] > max_init:
                            max_idx = i
                    
                    #print(max_idx)
                    #print(new_lst)
                    #print(ori_lst)

                    new_lst.append(ori_lst[max_idx])
                    ori_lst.remove(ori_lst[max_idx])

        sched_gen_n = new_lst
        
        ##############################
        ##############################
        sched_gen_n = np.reshape(sched_gen_n,(self.S,self.T,self.P))
        return sched_gen_n

    
    #function that prints the different result for debugging
    def printResult(self, new_sched):
        print(round(self.goodnessMeasure(new_sched),1))
        print(new_sched)       

    def outputResult(self, new_sched, out_file):
        file1 = open(out_file, 'w')
        
        for i in range(self.S):
            str_row = ''
            for j in range(self.T):
                for k in range(self.P):
                    if k != (self.P - 1):
                        str_row += str(new_sched[i][j][k] + 1) + ' '
                    else:
                        str_row += str(new_sched[i][j][k] + 1)
                if j != (self.T - 1):
                    str_row += ' | '
            if i != (self.S - 1):
                str_row += '\n'
            #print(str_row)     
            file1.write(str_row)
        file1.close()

if __name__ == '__main__':

    #command line arguments should be 2
    in_file  = sys.argv[1]
    out_file = sys.argv[2]

    #extract the two different possible values form the input file
    in_file_data = open(in_file)
    lt = [l for l in in_file_data]
    initial = assignment(lt)

    max_sched = np.array([],dtype=int)
    max_goodn = 0.0
    
    ###########Simulated Annealing############
    new_sched = initial.simulatedAnnealing()
    
    max_goodn = initial.goodnessMeasure(new_sched)
    max_sched = new_sched 
    ###########Hill Climbing##################
    new_sched = initial.hillClimbing()

    if initial.goodnessMeasure(new_sched) > max_goodn:
        new_sched = max_sched
    
    #prints the result for debugging
    #initial.printResult(new_sched)
    
    
    #outputs the result
    initial.outputResult(new_sched, out_file)
