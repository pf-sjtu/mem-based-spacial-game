# -*- coding: utf-8 -*-
"""
Created on Sun Dec  8 10:43:23 2019

@author: PENG Feng
@email:  im.pengf@outlook.com
"""

import numpy as np
from matplotlib import pyplot as plt
import time


## Hyper-parameters
R, S, T, P = [3, 0, 5, 1] # rewards
ref0 = 100 # init reference
num_nei = 1 # number of neighboors
type_nei = 'v' # r(andom) m(oore) v(on)
size_row, size_col = [10, 10] # size of the grid
d =  0.5 # forgetting rate
s = 0.25 # activation noise parameter
window_size = int(1e2) # number of generations in 1 window
window_num = 20
diff_tol = 1e-3 # asymptotically stable threshold

num_generation = window_size * window_num #int(20e4) # number of generations in total
sigma = np.sqrt(np.pi * s / np.sqrt(3))
m_num_nei = num_nei
v_num_nei = num_nei


if type_nei != 'r' and (2*num_nei + 1 > size_row or 2*num_nei + 1 > size_col):
    print('ERROR in \'get_neighbors\', map size is too small...')
if type_nei == 'm':
    num_nei = (2 * num_nei + 1) ** 2 - 1
elif type_nei == 'v':
    num_nei = 2 * num_nei * (num_nei + 1)

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
    memory = np.log(np.power(t1, -d) + (n - 1) * (np.power(tn, 1-d) - np.power(t1, 1-d)) / (1 - d) / (tn - t1))
    if all_radom:
        #noise = np.random.randn(2, num_nei + 1) * sigma
        # may be faster
        noise = np.random.normal(loc = 0, scale = sigma, size = (2, num_nei + 1))
    else:
        noise = np.random.randn(1)[0] * sigma
    return memory + noise

# the reference of each chunk
class Chunks_ref():
    def __init__(self):
        #print([2, num_nei+1])
        self.n = ref0 * np.ones([2, num_nei+1]) # total number of references
        self.t1 = np.ones([2, num_nei+1]) # time since last reference
        self.tn = ref0 * 2 * num_nei+1 * np.ones([2, num_nei+1]) # time since the fist reference
        self.update_B() # activations of a declarative chunks
    def update_B(self):
        self.B = B_fun(self.t1, self.tn, self.n, d, s, True)
    # get the position of the most active chunk
    def most_active_chunk(self):
        #mac0_index = np.argmax(self.B[0])
        #mac1_index = np.argmax(self.B[1])
        # may be faster
        mac0_index, mac1_index = [0, 0]
        for ii in range(num_nei+1):
            if self.B[0][ii] > self.B[0][mac0_index]: mac0_index = ii
            if self.B[1][ii] > self.B[1][mac1_index]: mac1_index = ii
        return [mac0_index, mac1_index]

# if a position(x, y) is out of the range of the map, bound it to a periodic position
def bound(pos):
    if (pos[0] >= size_row):
        pos[0] -= size_row
    elif (pos[0] < 0):
        pos[0] += size_row
    if (pos[1] >= size_col):
        pos[1] -= size_col
    elif (pos[1] < 0):
        pos[1] += size_col

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
            self.num_nei = num_nei
        elif type_nei == 'm': # r(andom) m(oore) v(on)
            self.num_nei = num_nei
            for nei_row in self.position[0] - m_num_nei + range(2*m_num_nei + 1):
                for nei_col in self.position[1] - m_num_nei + range(2*m_num_nei + 1):
                    if nei_row != self.position[0] or nei_col != self.position[1]:
                        nei_pos = [nei_row, nei_col]
                        bound(nei_pos)
                        #print(self.position, nei_pos)
                        self.nei = np.append(self.nei, nei_pos)
        elif type_nei == 'v': # r(andom) m(oore) v(on)
            self.num_nei = num_nei
            for nei_row in self.position[0] - v_num_nei + range(2*v_num_nei + 1):
                for nei_col in self.position[1] - v_num_nei +  np.abs(nei_row - self.position[0]) + \
                    range(2*(v_num_nei - np.abs(nei_row - self.position[0])) + 1):
                    if nei_row != self.position[0] or nei_col != self.position[1]:
                        nei_pos = [nei_row, nei_col]
                        bound(nei_pos)
                        #print(self.position, nei_pos)
                        self.nei = np.append(self.nei, nei_pos)
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
        print('Finished generating neighbors.')
        break
    except ValueError as e:
        print('Failed to generate neighbors (', e, '), redo...')
        continue

state = np.zeros([size_row, size_col]) # is the player in this position cooperates
f_c = [] # fraction of cooperations in a state

if __name__ == '__main__':
    verbose = True
    start_time = time.time()
    fc_table = np.ones([window_num, window_size])
    for window in range(window_num):
        for gen in range(window_size):
            # part 1, each player acts according to the state of last generation, 125/s
            generate_state = np.vectorize(Player.act)
            state = generate_state(players)
            fc_table[window, gen] = np.sum(state) / size_row / size_col
            #part 2, each player update the reference of chunks according to current state, 8/s
            for row_no in range(size_row):
                for col_no in range(size_col):
                    players[row_no, col_no].update_chunks_ref()
            #update_mem = np.vectorize(Player.update_chunks_ref)
            #update_mem(players)
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
    #fc_dot_color = np.where(np.abs(fc_diff) < diff_tol, 'g', 'red')
        
    plt.figure(figsize = (10, 6))
    plt.plot(range(1, window_num + 1), fc_mean)
    plt.scatter(range(1, window_num + 1), fc_mean, color = fc_dot_color, s = 50)
    plt.xlabel('Generation Number(*{})'.format(window_size))
    plt.ylabel('Cooperation Rate')
    plt.title('RSS_model(neighbor type:{})'.format(type_nei))
    
    # find asymptotically stable value (if exists)
    if 'g' in fc_dot_color:
        fc_dot_color[-1] = 'g'
        asym_val = np.mean(fc_mean[fc_dot_color == 'g'])
        plt.axhline(y = asym_val, color = 'g', ls="--")
        plt.title('RSS_model(neighbor type:{}, asymptotically stable value:{:.4f})'.format(type_nei, asym_val))
    
    plt.savefig('RSS_fc.png', dpi = 300)
    plt.show()

