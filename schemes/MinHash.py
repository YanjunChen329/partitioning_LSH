from sklearn.utils import murmurhash3_32 as mmh
from LSH import LSH
import time
import numpy as np


class MinHash(LSH):
    def __init__(self, K, L, D, bbit=2, R=2**20, seed=0):
        super(MinHash, self).__init__("MinHash")
        self.K = K
        self.L = L
        self.D = D
        self.bbit = bbit
        self.R = R
        self.hash_tables = [{} for _ in range(L)]

        self.seed = range(seed, seed+K*L+1, K)

        # Timing
        self.insert_time = np.zeros(2, dtype=float)
        self.query_time = np.zeros(2, dtype=float)

    def hash_func(self, seed):
        return lambda x: mmh(key=x, seed=seed, positive=True) % self.D

    def hash_to_bucket_bbit(self, k_hashes):
        bbit_rep = map(lambda x: bin(x)[-self.bbit], k_hashes)
        return "".join(bbit_rep)

    def hash_to_bucket(self, k_hashes, seed):
        key = "-".join(map(lambda x: str(x), k_hashes))
        return mmh(key=key, seed=seed, positive=True) % self.R

    def min_hash(self, x, seed):
        hashfunc = self.hash_func(seed)
        return min(list(map(lambda a: hashfunc(a), x)))

    def generate_k_minhash(self, x, seed):
        k_hashes = []
        for k in range(self.K):
            k_hashes.append(self.min_hash(x, seed + k))
        return k_hashes

    def insert(self, x, id):
        start = time.clock()
        for l in range(self.L):
            k_hashes = self.generate_k_minhash(x, self.seed[l])
            key = self.hash_to_bucket(k_hashes, self.seed[l])
            if key not in self.hash_tables[l]:
                self.hash_tables[l][key] = set()
            self.hash_tables[l][key].add(id)
        self.insert_time[0] += time.clock() - start
        self.insert_time[1] += 1

    def query(self, x):
        start = time.clock()
        retrieved = []
        for l in range(self.L):
            k_hashes = self.generate_k_minhash(x, self.seed[l])
            key = self.hash_to_bucket(k_hashes, self.seed[l])
            retrieved.append(self.hash_tables[l].get(key, set()))
        result = set.union(*retrieved)
        self.query_time[0] += time.clock() - start
        self.query_time[1] += 1
        return result

    def get_insert_time(self):
        return self.insert_time[0] / self.insert_time[1]

    def get_query_time(self):
        return self.query_time[0] / self.query_time[1]


if __name__ == '__main__':
    from MinHash_test import MinHash_tester

    K = 10
    L = 50
    D = 1000
    tester = MinHash_tester(MinHash, seeds_per_table=K*L, K=K, L=L, D=D)
    tester.test_retrieval_prob()
    # tester.test_jaccard()
