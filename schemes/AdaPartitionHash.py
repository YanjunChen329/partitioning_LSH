from sklearn.utils import murmurhash3_32 as mmh
import numpy as np
import time
import math
from LSH import LSH


class AdaPartition_Hash(LSH):
    def __init__(self, L, D, J=0.9, c=0.999, R=2**20, max_len=1000, seed=0):
        super(AdaPartition_Hash, self).__init__("AdaPartition_Hash")
        self.L = L          # Use L hash tables
        self.J = J          # Target Jaccard similarity to cover
        self.c = c          # Covering probability
        self.D = D          # dimensionality of dictionary
        self.R = R          # size of hash table
        self.prime = 1299709

        self.hash_tables = [{} for _ in range(self.L)]
        self.seed = [i + seed for i in range(L)]
        np.random.seed(seed)
        self.hash_params = np.random.randint(0, self.prime-1, (self.L, self.D))

        # For n estimation
        self.sliding_J = np.arange(J, 1./J, (1./J-J)/L)
        self.ratio_map = {}
        self.initialize_ratio_map(max_len)

        # Timing
        self.insert_time = np.zeros(2, dtype=float)
        self.query_time = np.zeros(2, dtype=float)
        self.time_details = {}
        self.initialize_time_details()

    def initialize_time_details(self):
        self.time_details["pre-calculate"] = 0.
        self.time_details["initialize"] = 0.
        self.time_details["one-pass"] = 0.
        self.time_details["densification"] = 0.

    def initialize_ratio_map(self, max_len):
        for n in range(2, max_len):
            self.get_target_ratio(n)

    def get_target_ratio(self, n):
        if n not in self.ratio_map:
            size = 1.2
            p = 1 - (1 - (1 - self.c)**(size/self.L))**(1./math.ceil((1-self.J)*n))
            # p = round(p, 2)
            # if p == 0:
            #     p = 0.01
            self.ratio_map[n] = p
        return self.ratio_map[n]

    def hash_func(self, seed):
        return lambda x: mmh(key=x, seed=seed, positive=True) % self.D

    def hash_to_bucket(self, key, table):
        # hashcode = "-".join(map(lambda x: str(x), key))
        # return mmh(key=hashcode, seed=self.seed[table], positive=True) % self.R
        hashcode = map(lambda x: self.hash_params[table][x], key)
        return sum(hashcode) % self.prime % self.R

    def ada_partition_hash(self, word_set, table, insert=True):
        stop1 = time.clock()
        seed = self.seed[table]
        estimated = (1 + self.J) / 2
        if insert:
            n_hat = int(len(word_set) / estimated / self.sliding_J[table])    # calculate estimated n (unions)
        else:
            n_hat = int(len(word_set) / estimated)
        stop11 = time.clock()
        # print(length, self.get_target_ratio(length), n_hat, self.get_target_ratio(n_hat))
        p = self.get_target_ratio(n_hat)
        cutoff = int(self.D * p)
        stop2 = time.clock()

        hash_func = self.hash_func(seed)
        bins = [[] for _ in range(int(1 / p) + 1)]

        for w in word_set:
            val = hash_func(w)
            bins[int(val / cutoff)].append(val)
        move = 0
        stop3 = time.clock()
        while len(bins[move]) == 0:
            move += 1
        hash_code = bins[move]
        stop4 = time.clock()

        self.time_details["pre-calculate"] += stop11 - stop1
        self.time_details["initialize"] += stop2 - stop11
        self.time_details["one-pass"] += stop3 - stop2
        self.time_details["densification"] += stop4 - stop3

        # hashcode = str(move) + "."
        return hash_code

    def insert(self, x, id):
        start = time.clock()
        for l in range(self.L):
            hashcode = self.ada_partition_hash(x, l)
            key = self.hash_to_bucket(hashcode, l)
            table_id = l
            if key not in self.hash_tables[table_id]:
                self.hash_tables[table_id][key] = set()
            self.hash_tables[table_id][key].add(id)
        self.insert_time[0] += time.clock() - start
        self.insert_time[1] += 1

    def query(self, x):
        start = time.clock()
        retrieved = []
        for l in range(self.L):
            hashcode = self.ada_partition_hash(x, l, insert=False)
            key = self.hash_to_bucket(hashcode, l)
            table_id = l
            retrieved.append(self.hash_tables[table_id].get(key, set()))
        result = set.union(*retrieved)
        self.query_time[0] += time.clock() - start
        self.query_time[1] += 1
        return result

    def get_insert_time(self):
        return self.insert_time[0] / self.insert_time[1]

    def get_query_time(self):
        return self.query_time[0] / self.query_time[1]

    def get_time_details(self):
        dic = dict(self.time_details)
        for key in dic:
            dic[key] /= self.insert_time[1]
        total = sum(list(dic.values()))
        # for key in dic:
        #     dic[key] /= total
        return total, dic


if __name__ == '__main__':
    from MinHash_test import MinHash_tester

    J = 0.9
    L = 50
    c = 0.9
    D = 10000
    LSH = AdaPartition_Hash(L=L, J=J, D=D, c=c, max_len=1000)
    # print(set(LSH.ratio_map.values()))
    # LSH.insert([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 1)
    # print(LSH.query([1, 2, 3, 4, 5, 6, 7, 8, 9]))

    tester = MinHash_tester(AdaPartition_Hash, seeds_per_table=L, L=L, J=J, D=D, max_len=20)
    tester.test_retrieval_prob()

