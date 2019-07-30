from sklearn.utils import murmurhash3_32 as mmh
import numpy as np
import time
import random
from LSH import LSH


def chunk(xs, n):
    """Split the list, xs, into n evenly sized chunks"""
    L = len(xs)
    assert 0 < n <= L
    s, r = divmod(L, n)
    t = s + 1
    return [xs[p:p+t] for p in range(0, r*t, t)] + [xs[p:p+s] for p in range(r*t, L, s)]


class Densified_MinHash(LSH):
    def __init__(self, K, L, D, R=2**20, seed=0):
        super(Densified_MinHash, self).__init__("Densified_MinHash")
        self.K = K
        self.L = L
        self.D = D
        self.R = R
        self.bins = list(map(lambda x: [x[0], x[-1]], chunk(np.arange(0, D), K)))
        # print(self.bins)
        self.hash_tables = [{} for _ in range(L)]
        self.seed = [i + seed for i in range(L)]
        self.prime = 1299709
        np.random.seed(seed)
        self.hash_params = np.random.randint(0, self.prime-1, (self.L, self.K))

        # Timing
        self.insert_time = np.zeros(2, dtype=float)
        self.query_time = np.zeros(2, dtype=float)
        self.time_details = {}
        self.initialize_time_details()

    def initialize_time_details(self):
        self.time_details["get-min"] = 0.
        self.time_details["one-pass"] = 0.
        self.time_details["densification"] = 0.
        self.time_details["hash-bucket"] = 0.

    def hash_func(self, seed):
        return lambda x: mmh(key=x, seed=seed, positive=True) % self.D

    def hash_bin_to_bin(self, bin_id, attempt, seed):
        key = str(attempt) + "." + str(bin_id)
        return mmh(key=key, seed=seed, positive=True) % self.K

    def hash_to_bucket(self, k_hashes, table):
        key = "-".join(map(lambda x: str(x), k_hashes))
        return mmh(key=key, seed=self.seed[table], positive=True) % self.R
        # bucket = np.inner(np.array(k_hashes), self.hash_params[table]) % self.prime % self.R
        # return bucket

    def one_permutation_hash(self, word_set, seed):
        stop1 = time.clock()
        k_hashes = [-1 for _ in range(self.K)]
        hash_func = self.hash_func(seed)
        for w in word_set:
            hash_val = hash_func(w)
            idx = int(1. * hash_val * self.K / self.D)
            if hash_val > k_hashes[idx]:
                k_hashes[idx] = hash_val
            # for idx in range(self.k):
            #     if self.bins[idx][0] <= hash_val <= self.bins[idx][-1]:
            #         if hash_val > k_hashes[idx]:
            #             k_hashes[idx] = hash_val
            #         break
        stop2 = time.clock()

        # k_hashes = list(map(lambda x: min(x) if len(x) != 0 else -1, k_bins))  # get min in each bin
        stop3 = time.clock()

        # optimal densified hashing for empty bins
        for idx in range(self.K):
            if k_hashes[idx] == -1:
                attempt = 1
                new_bin = self.hash_bin_to_bin(idx, attempt, seed)
                while k_hashes[new_bin] == -1:
                    attempt += 1
                    new_bin = self.hash_bin_to_bin(idx, attempt, seed)
                k_hashes[idx] = k_hashes[new_bin]
        # print(k_hashes)
        stop4 = time.clock()
        self.time_details["one-pass"] += stop2 - stop1
        self.time_details["get-min"] += stop3 - stop2
        self.time_details["densification"] += stop4 - stop3
        return k_hashes

    def insert(self, x, id):
        start = time.clock()
        for l in range(self.L):
            k_hashes = self.one_permutation_hash(x, l)
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
            k_hashes = self.one_permutation_hash(x, l)
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

    K = 10
    L = 50
    D = 1000
    # Densified_MinHash(K, L, D, overlap=True)

    tester = MinHash_tester(Densified_MinHash, seeds_per_table=L, K=K, L=L, D=D)
    tester.test_retrieval_prob()
    # tester.test_jaccard()





