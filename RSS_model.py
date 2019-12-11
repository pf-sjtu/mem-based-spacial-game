# -*- coding: utf-8 -*-
"""
Created on Sun Dec  8 10:43:23 2019

@author: PENG Feng
@email:  im.pengf@outlook.com
"""

import numpy as np
#import pandas as pd
from matplotlib import pyplot as plt
import time

## Hyper-parameters
R, S, T, P = [3, 0, 5, 1] # rewards
ref0 = 100 # init reference
num_nei = 4 # number of neighboors
type_nei = 'r' # r(andom) m(oore) v(on)
size_row, size_col = [20, 20] # size of the grid
d =  0.5 # forgetting rate
s = 0.25 # activation noise parameter
num_generation = 1000 #int(20e4) # number of generations in total
window_size = int(1e4) # number of generations in 1 window

# chunk
class Chunk():
    def __init__(self, p_move, N_config):
        self.p_move = p_move
        self.N_config = N_config
        if p_move == 1: # player cooperates
            self.payoff = N_config * R + (num_nei - N_config) * S
        elif p_move == 0: # player defects
            self.payoff = N_config * T + (num_nei - N_config) * P

# Row 1 stands for C, 0 stands for D
# Col n stands for nC
chunks = np.array([[Chunk(0, N_config) for N_config in range(num_nei + 1)], 
                   [Chunk(1, N_config) for N_config in range(num_nei + 1)]])

# activations of a declarative chunks
def B_fun(t1, tn, n, d, s, all_radom = False):
    memory = np.log(np.power(t1, -d) + 
                    (n - 1) * (np.power(tn, 1-d) - np.power(t1, 1-d)) 
                    / (1 - d) / (tn - t1))
    if all_radom:
        noise = np.random.randn(2, num_nei + 1) * np.sqrt(np.pi * s / np.sqrt(3))
    else:
        noise = np.random.randn(1)[0] * np.sqrt(np.pi * s / np.sqrt(3))
    return memory + noise

# the reference of each chunk
class Chunks_ref():
    def __init__(self):
        self.n = ref0 * np.ones([2, num_nei+1]) # total number of references
        self.t1 = np.ones([2, num_nei+1]) # time since last reference
        self.tn = ref0 * 2 * num_nei+1 * np.ones([2, num_nei+1]) # time since the fist reference
        self.update_B() # activations of a declarative chunks
    def update_B(self):
        self.B = B_fun(self.t1, self.tn, self.n, d, s, True)
    # get the position of the most active chunk
    def most_active_chunk(self):
        mac0_index = np.argmax(self.B[0])
        mac1_index = np.argmax(self.B[1])
        return [mac0_index, mac1_index]

class Player():
    def __init__(self, pos_index):
        self.sum_payoff = 0
        self.chunks_ref = Chunks_ref()
        self.position = np.array([int(pos_index / size_col), pos_index % size_col], dtype=np.int16)
        self.nei = np.array([], dtype = np.int16)
    def get_neighbors(self):
        if type_nei == 'r': # r(andom) m(oore) v(on)
            self.num_nei = players_num_nei[self.position[0], self.position[1]]
            players_num_nei[self.position[0], self.position[1]] = num_nei
            # no replace choose neighbors
            nei = np.random.choice(players[players_num_nei < num_nei], 
                                   num_nei - self.num_nei, 
                                   replace = False)
            nei_pos = np.array([each_nei.position for each_nei in nei], dtype=np.int16)
            self.nei = np.append(self.nei, nei_pos)
            for each_nei in nei:
                players_num_nei[each_nei.position[0], each_nei.position[1]] += 1
                players[each_nei.position[0], each_nei.position[1]].nei = \
                    np.append(players[each_nei.position[0], each_nei.position[1]].nei, [self.position])            
        elif type_nei == 'm': # r(andom) m(oore) v(on)
            return 0
        elif type_nei == 'v': # r(andom) m(oore) v(on)
            return 0
    # find the action with the larger probable payoff
    def act(self):
        mac = self.chunks_ref.most_active_chunk()
        action = 0 if (chunks[0, mac[0]].payoff > chunks[1, mac[1]].payoff) else 1
        return action
    def update_chunks_ref(self):
        # solve the current p-move and N-config
        p_move = state[self.position[0], self.position[1]]
        N_config = 0
        for nei_no in range(num_nei):
            N_config += state[self.nei[2 * nei_no], self.nei[2 * nei_no + 1]]
        # update chunks references
        self.chunks_ref.n[p_move, N_config] += 1 # total number of references
        self.chunks_ref.t1[p_move, N_config] = 0 # time since last reference
        self.chunks_ref.t1 += 1
        self.chunks_ref.tn += 1 # time since the fist reference
        self.chunks_ref.update_B() # activations of a declarative chunks
        return 0

# generate neighbor pairs
while (True):
    try:
        players = np.array([Player(pos_i) for pos_i in range(size_row * size_col)]).reshape(size_row, size_col)
        players_num_nei = np.zeros([size_row, size_col], dtype = np.int16)
        
        for row_no in range(size_row):
            for col_no in range(size_col):
                players[row_no, col_no].get_neighbors()
        print('Finished generating radom neighbors.')
        break
    except ValueError as e:
        print('Failed to generate radom neighbors (', e, '), redo...')
        continue

state = np.zeros([size_row, size_col]) # is the player in this position cooperates
f_c = [] # fraction of cooperations in a state

if __name__ == '__main__':
    verbose = True
    start_time = time.time()
    for generation in range(num_generation):
        # part 1, each player acts according to the state of last generation, 125/s
        generate_state = np.vectorize(Player.act)
        state = generate_state(players)
        f_c.append(np.sum(state) / size_row / size_col)
        #part 2, each player update the reference of chunks according to current state, 8/s
        update_mem = np.vectorize(Player.update_chunks_ref)
        update_mem(players)
        if verbose and generation % 100 == 0:
            print('Finished Generation: {}, duration: {:.2f}s'.format(generation, time.time() - start_time))
    
    plt.plot(range(num_generation), f_c)
    plt.savefig('f_c.png')