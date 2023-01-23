#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import string
from itertools import product, repeat, count, zip_longest
import pickle
import time
import os
from multiprocessing import Pool


# In[3]:


class shelf_graph:
    
    def __init__(self, order, nP = 4):
        
        self.order = order
        self.conditions = {}
        self.max_condition_id = -1
        self.generate_all_conditions()
        self.shelves = np.empty((0, order,order), int)
        self.init_matrix = np.ones((order, order,), dtype=int) *-1
        self.number_shelves = 0
        self.partition = 0
        self.bfs_out_candidates = np.empty((0, order,order), int)
        self.bfs_last_condition = 0
        self.shelve_files = []
        self.pool_size = nP
        self.range_order = np.arange(order, dtype=np.int64)
        self.prod2 = np.asarray(list(product(np.arange(order, dtype=np.int64), repeat=2)))
        self.time_dict = {}
        
    def generate_all_conditions(self):
        """
        Generates the n^3 conditions where n is the order
        The conditions are ordered according the the number of discting elements among the three elments.
        Those conditions with the most distinct elements are placed earlier because that proved faster complettion times.
        """
        conditions = []
        for x in range(self.order):
            for y in range(self.order):
                for z in range(self.order):
                    conditions.append([x,y,z])
        conditions = sorted(conditions, key=lambda x: len(np.unique(x)), reverse=True) #might speed up the alog, but not necessary
        self.conditions = dict(zip(range(self.order**3), conditions[0::2] + conditions[1::2]))     
        self.max_condition_id = self.order**3 - 1
        
    def check_candidate(self, matrix_in, c_id_in):
        """
        Returns true if the candidate matrix does not violate any of the remaining conditions.
        """
        for c_id in range(c_id_in+1, self.max_condition_id+1):
            condition = self.conditions[c_id]
            x = condition[0]
            y = condition[1]
            z = condition[2]
            if (matrix_in[x][y] == -1) | (matrix_in[x][z] == -1) | (matrix_in[y][z] == -1):
                continue
            elif (matrix_in[matrix_in[x][y]][z] == -1) | (matrix_in[matrix_in[x][z]][matrix_in[y][z]] == -1):
                continue
            elif (matrix_in[matrix_in[x][y]][z] == matrix_in[matrix_in[x][z]][matrix_in[y][z]]):
                continue
            else:
                return False
        return True
        
    def get_shelves(self):
        """
        Parallelize the shelf generation process.
        First generate a set of candidates by applying the first two conditions in a BFS manner.
        These candidates are then allocated multiple independent processes that run in parallel to apply the remaining conditions in a DFS manner.
        """
        # call BFS
        self.bfs(self.init_matrix)
        # randomize the bfs candidates. Why? because it was observed that candidates that generate a lot of child candidates may appear as negibors in BFS.
        np.random.shuffle(self.bfs_out_candidates)
        print(f"BFS found {self.bfs_out_candidates.shape} unique candidates")
        # After a partition_size number of sheves has been found write them to a file
        # this is to free up memeory when running for large orders
        partition_size = int(len(self.bfs_out_candidates)/10)
        
        try:            
            with Pool(self.pool_size) as p:
                # allocate 10 candidates at a time to each process
                partial_results = p.imap_unordered(self.get_children, zip(self.bfs_out_candidates, count(0), repeat(self.bfs_last_condition+1)), chunksize = 10)
                res_count = 0
                for i, res in enumerate(partial_results):
                    res_count += 1
                    self.shelves = np.append(self.shelves, res, axis=0)
                    self.number_shelves += res.shape[0]          
                    if ((i+1)%500) == 0:
                        print(f'Currently got {res_count} results and {self.number_shelves} shelves (may include duplicates which will be filtered out at the end)')
                    if ((i+1)%partition_size) == 0:
                        file_ = f"shelves_order_{self.order}_partition_{self.partition}.pkl"
                        # fileter out duplicates before writing to file
                        self.shelves = np.unique(self.shelves,axis=0)
                        print(f'Writing paritial results to {file_}, number of unique shelves in this set{self.shelves.shape}')
                        with open(file_, "wb") as fp:
                            self.shelve_files.append(file_)
                            pickle.dump(self.shelves, fp)
                            self.partition += 1
                            self.shelves = np.empty((0, self.order, self.order), int)
                            
        except Exception as e:
            print(e)
        
        print("Processing completed. Assembling results...")
        
        try:
            # read the files and assemble all, along with the final set that is in the memory
            # then delete the partition fiels and wirte the single results file
            for file in self.shelve_files:
                print('Partition file: ',file)
                with open(file, "rb") as fp:
                    self.shelves = np.append(self.shelves, pickle.load(fp), axis=0)
                    self.shelves = np.unique(self.shelves, axis=0)
            print(f"Number of unique shelves {self.shelves.shape[0]}")
            with open(f"shelves_order_{self.order}.pkl", "wb") as fp:   # Unpickling
                pickle.dump(self.shelves, fp)
            for file in self.shelve_files:
                os.remove(file)
        except Exception as e: 
            print(e)
            print("Saving unsaved Shelves")
            file_ = f"shelves_order_{self.order}_partition_{self.partition}.pkl"
            self.shelve_files.append(file_)
            with open(file_, "wb") as fp:   # Unpickling
                pickle.dump(self.shelves, fp)
                self.partition += 1
                self.shelves = np.empty((0, self.order, self.order), int) 
            print("All Shelves should be availble in .pkl files")
                              
    def get_children(self, inputs):
        """
        Recursively apply the next condition.
        when the last condition is successfully reached, that is a shelf
        """
        matrix, bfs_id, condition_id = inputs
        # condition_id is the current condition to apply
        if (condition_id > self.max_condition_id):
            return matrix.reshape(1,self.order, self.order)
        else:
            out_shelves = np.empty((0, order,order), int)
            for child_matrix in self.apply_condition(matrix, condition_id):
                out_shelfs = self.get_children((child_matrix, bfs_id,condition_id+1))
                out_shelves = np.append(out_shelves, out_shelfs, axis=0)
#             if (condition_id == (self.bfs_last_condition + 1)) & (out_shelves.shape[0]>0):
#                 out_shelves = np.unique(out_shelves, axis=0)
#                 print(f"Number of Shelves found in bfs candidate {bfs_id}: {out_shelves.shape[0]}\n")
            return out_shelves
            
    def apply_condition(self, matrix_in, condition_id):
        """
        condition = [x,y,z]
        M[M[x][y]][z] = M[M[x][z]][M[y][z]]
        lhs_1 = M[x][y]
        rhs_1 = M[x][z]
        rhs_2 = M[y][z]
        lhs_out = M[M[x][y]][z]
        rhs_out = M[M[x][z]][M[y][z]]
        
        Check the inner LHS condition

        matrix_in is not modified
        """
        condition = self.conditions[condition_id]
        x = condition[0]
        y = condition[1]
        z = condition[2]

        if matrix_in[x][y] == -1:
            possible_values_lhs_1 = self.range_order
            M = matrix_in.copy()
            for lhs_1 in possible_values_lhs_1:
                M[x][y] = lhs_1
                for i in self.rhs_check(M, x, y, z, condition_id):
                    yield i
        else:
            for i in self.rhs_check(matrix_in, x, y, z, condition_id):
                yield i

            
    def rhs_check(self, M, x, y, z, c_id):
        """
        Check the two inner RHS conditions
        """
        out_candidates = []

        if (M[x][z] == -1) & (M[y][z] == -1):
            possible_values_rhs_1_2 = self.prod2
            M_1 = M.copy()
            for rhs_1, rhs_2 in possible_values_rhs_1_2:
                M_1[x][z] = rhs_1
                M_1[y][z] = rhs_2
                yield from self.final_check(M_1, x, y, z, c_id)

        elif (M[x][z] == -1) & (M[y][z] != -1):
            possible_values_rhs_1 = self.range_order
            M_1 = M.copy()
            for rhs_1 in possible_values_rhs_1:
                M_1[x][z] = rhs_1
                yield from self.final_check(M_1, x, y, z, c_id)

        elif (M[x][z] != -1) & (M[y][z] == -1):
            possible_values_rhs_2 = self.range_order
            M_1 = M.copy()
            for rhs_2 in possible_values_rhs_2:
                M_1[y][z] = rhs_2
                yield from self.final_check(M_1, x, y, z, c_id)

        elif (M[x][z] != -1) & (M[y][z] != -1):
            yield from self.final_check(M, x, y, z, c_id)
            
    def final_check(self, M_1, x, y, z, c_id):
        """
        Match LHS and RHS
        M_1 is not modified
        
        """
        out_candidates = []

        if (M_1[M_1[x][y]][z] == -1) & (M_1[M_1[x][z]][M_1[y][z]] == -1):
            possible_values_lhs_rhs = self.range_order
            for lhs_rhs in possible_values_lhs_rhs:
                M_2 = M_1.copy()
                M_2[M_2[x][y]][z] = lhs_rhs
                M_2[M_2[x][z]][M_2[y][z]] = lhs_rhs
                if self.check_candidate(M_2, c_id):
                    yield M_2
        elif (M_1[M_1[x][y]][z] == - 1) & (M_1[M_1[x][z]][M_1[y][z]] != -1):
            M_2 = M_1.copy()
            M_2[M_2[x][y]][z] = M_2[M_2[x][z]][M_2[y][z]]
            if self.check_candidate(M_2, c_id):
                yield M_2
        elif (M_1[M_1[x][y]][z] != -1) & (M_1[M_1[x][z]][M_1[y][z]] == -1):
            M_2 = M_1.copy()
            M_2[M_2[x][z]][M_2[y][z]] = M_2[M_2[x][y]][z]
            if self.check_candidate(M_2, c_id):
                yield M_2
        elif M_1[M_1[x][y]][z] == M_1[M_1[x][z]][M_1[y][z]]:
            M_2 = M_1.copy()
            if self.check_candidate(M_2, c_id):
                yield M_2
            
    def bfs(self, initial_matrix):
        """
        Intial condidate generation applying the first two conditions.
        Could apply more conditions (or all conditions) for lower orders, but for higher orders >=5 BFS may run out of memory
        After a few conditions, becuase too many candidates are generated"""
        start_candidates = [initial_matrix]
        canditate_count = 0
        for condition_id in range(2):
            #print('Condition id ', key_)
            start_con = time.time()
            end_candidate = []
            for matrix in start_candidates:
                new_candidates = list(self.apply_condition(matrix, condition_id))
#                 print(new_candidates)
                canditate_count += len(new_candidates)
                end_candidate.extend(new_candidates)
            print(f'Condtion {self.conditions[condition_id]} Start candidates {len(start_candidates)} End candidates {len(end_candidate)} Time for condtion {time.time() - start_con} s')
            start_candidates = np.unique(end_candidate,axis=0)
#         print(type(start_candidates))
        self.bfs_out_candidates = np.append(self.bfs_out_candidates, start_candidates, axis=0)
        self.bfs_last_condition = condition_id


# In[4]:


if __name__ == "__main__":
    start_ = time.time()
    #order
    order = 4
    # number of parallel processes, set to number of cores
    nP = 4
    graph = shelf_graph(order,  nP)
    graph.get_shelves()
    end_ = time.time()
    print(f'Total time {end_ - start_} s')


# In[ ]:




