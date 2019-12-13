#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 20:59:22 2019

@author: pf
"""
import math
import numpy as np
from matplotlib import pyplot as plt
import time

## Hyper-parameters
R, S, T, P = [3, 0, 5, 1] # rewards
ref0 = 100 # init reference
size_row, size_col = [10, 10] # size of the grid
d =  0.5 # forgetting rate
s = 0.25 # activation noise parameter
window_size = int(1e4) # number of generations in 1 window
window_num = 20
num_generation = window_size * window_num #int(20e4) # number of generations in total
diff_tol = 1e-3 # asymptotically stable threshold

sigma = np.sqrt(np.pi * s / np.sqrt(3))

# chunk
class Chunk():
    def __init__(self, p_move, N_config):
        self.p_move = p_move
        self.N_config = N_config
        if p_move == 1: # player cooperates
            self.payoff = N_config * R + (1 - N_config) * S
        elif p_move == 0: # player defects
            self.payoff = N_config * T + (1 - N_config) * P

# Row 1 stands for C, 0 stands for D
# Col n stands for nC
chunks = np.array([[Chunk(0, N_config) for N_config in range(2)], 
                   [Chunk(1, N_config) for N_config in range(2)]])

def B_fun_mem(t1, tn, n, d):
    return math.log(math.pow(t1, -d) + (n - 1) * (math.pow(tn, 1-d) - math.pow(t1, 1-d)) / (1 - d) / (tn - t1))
    
# activations of a declarative chunks
def B_fun(t1, tn, n, d, s, all_radom = False):
    #memory = np.log(np.power(t1, -d) + (n - 1) * (np.power(tn, 1-d) - np.power(t1, 1-d)) / (1 - d) / (tn - t1))

    # may be faster
    memory = np.zeros([2, 2])
    for row_no in [0,1]:
        for col_no in [0,1]:
            memory[row_no][col_no] = B_fun_mem(t1[row_no][col_no], tn[row_no][col_no], n[row_no][col_no], d)

    if all_radom:
        #noise = np.random.randn(2, 2) * sigma
        # may be faster
        noise = np.random.normal(loc = 0, scale = sigma, size = (2, 2))
    else:
        noise = np.random.randn(1)[0] * sigma
    return memory + noise

# the reference of each chunk
class Chunks_ref():
    def __init__(self):
        self.n = ref0 * np.ones([2, 2]) # total number of references
        self.t1 = np.ones([2, 2]) # time since last reference
        self.tn = ref0 * 2 * 2 * np.ones([2, 2]) # time since the fist reference
        self.update_B() # activations of a declarative chunks
    def update_B(self):
        self.B = B_fun(self.t1, self.tn, self.n, d, s, True)
    # get the position of the most active chunk
    def most_active_chunk(self):
        #mac0_index = np.argmax(self.B[0])
        #mac1_index = np.argmax(self.B[1])
        # may be faster
        mac0_index = 0 if self.B[0][0] > self.B[0][1] else 1
        mac1_index = 0 if self.B[1][0] > self.B[1][1] else 1
        return [mac0_index, mac1_index]

class Player():
    def __init__(self, pos_index):
        self.sum_payoff = 0
        self.chunks_ref = Chunks_ref()
        self.position = np.array([int(pos_index / size_col), pos_index % size_col], dtype=np.int16)
        self.nei = np.array([], dtype = np.int16)
    def get_neighbors(self):
        # r(andom)
        self.num_nei = players_num_nei[self.position[0], self.position[1]]
        if self.num_nei == 0:
            players_num_nei[self.position[0], self.position[1]] = 1
            # no replace choose neighbors
            # nei = np.random.choice(players[players_num_nei == 0], 1, replace = True)[0]
            # may be faster
            aval_players = players[players_num_nei == 0].copy()
            np.random.shuffle(aval_players)
            nei = aval_players[0]
            self.nei = nei.position
            players_num_nei[nei.position[0], nei.position[1]] += 1
            players[nei.position[0], nei.position[1]].nei = self.position
            self.num_nei = 1

    # find the action with the larger probable payoff
    def act(self):
        mac = self.chunks_ref.most_active_chunk()
        action = 0 if (chunks[0, mac[0]].payoff > chunks[1, mac[1]].payoff) else 1
        return action
    def update_chunks_ref(self):
        # solve the current p-move and N-config
        p_move = state[self.position[0], self.position[1]]
        N_config = state[self.nei[0], self.nei[1]]
        # update chunks references
        self.chunks_ref.n[p_move, N_config] += 1 # total number of references
        self.chunks_ref.t1[p_move, N_config] = 0 # time since last reference
        self.chunks_ref.t1 += 1
        self.chunks_ref.tn += 1 # time since the fist reference
        self.chunks_ref.update_B() # activations of a declarative chunks
        return 0

# init
players = np.array([Player(pos_i) for pos_i in range(size_row * size_col)]).reshape(size_row, size_col)


state = np.zeros([size_row, size_col]) # is the player in this position cooperates
f_c = [] # fraction of cooperations in a state

if __name__ == '__main__':
    verbose = True
    start_time = time.time()
    fc_table = np.ones([window_num, window_size])
    for window in range(window_num):
        for gen in range(window_size):
            # init DP pairs
            players_num_nei = np.zeros([size_row, size_col], dtype = np.int16)
            for row_no in range(size_row):
                for col_no in range(size_col):
                    players[row_no, col_no].get_neighbors()
            # part 1, each player acts according to the state of last generation, 125/s
            generate_state = np.vectorize(Player.act)
            state = generate_state(players)
            fc_table[window, gen] = np.sum(state) / size_row / size_col
            #part 2, each player update the reference of chunks according to current state, 8/s
            for row_no in range(size_row):
                for col_no in range(size_col):
                    players[row_no, col_no].update_chunks_ref()
            if verbose and gen % 100 == 0:
                duration = time.time() - start_time
                expect_time = duration / (window * window_size + gen + 1) * (window_num * window_size)
                print('Win:{}/{}, Gen:{}/{}, Duration: {:.2f}s/{:.2f}s'.format(\
                    window + 1, window_num, gen + 1, window_size, duration, expect_time))
    # plot the f_c
    fc_mean = fc_table.mean(axis = 1)
    fc_diff = np.insert(np.diff(fc_mean), 0, 1, axis = 0)
    
    fc_dot_color = np.where(np.abs(fc_diff) > -1, 'r', 'r')
    ii = window_num - 1
    while (np.abs(fc_diff[ii]) < diff_tol and ii >= 0):
        fc_dot_color[ii] = 'g'
        ii -= 1
    #fc_dot_color = np.where(np.abs(fc_diff) < diff_tol, 'g', 'r')
        
    plt.figure(figsize = (10, 6))
    plt.plot(range(1, window_num + 1), fc_mean)
    plt.scatter(range(1, window_num + 1), fc_mean, color = fc_dot_color, s = 50)
    plt.xlabel('Generation Number(*{})'.format(window_size))
    plt.ylabel('Cooperation Rate')
    plt.title('Mean-Field_model')
    
    # find asymptotically stable value (if exists)
    if 'g' in fc_dot_color:
        fc_dot_color[-1] = 'g'
        asym_val = np.mean(fc_mean[fc_dot_color == 'g'])
        plt.axhline(y = asym_val, color = 'g', ls="--")
        plt.title('Mean-Field_model(asymptotically stable value:{:.4f})'.format(asym_val))
    
    plt.savefig('Mean-Field_model_fc.png', dpi = 300)
    plt.show()