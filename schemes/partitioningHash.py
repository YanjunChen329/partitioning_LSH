from sklearn.utils import murmurhash3_32 as mmh
import numpy as np
import random
import math
from itertools import combinations
from LSH import LSH


def chunk(xs, n):
    """Split the list, xs, into n evenly sized chunks"""
    L = len(xs)
    assert 0 < n <= L
    s, r = divmod(L, n)
    t = s + 1
    return [xs[p:p+t] for p in range(0, r*t, t)] + [xs[p:p+s] for p in range(r*t, L, s)]


def nCr(k, r):
    f = math.factorial
    return f(k) / f(r) / f(k - r)


class Partitioning_Hash(LSH):
    def __init__(self, K, r, D, t=1, bbit=2, seed=0):
        super(Partitioning_Hash, self).__init__("Partitioning_Hash")
        self.K = K          # split the dictionary into K partitions
        self.r = r          # use r partitions as hash codes
        self.L = 1  # L hash tables as a unit
        self.t = t          # use t hash table units (t*L tables in total)
        self.D = D          # dimensionality of dictionary
        self.bbit = bbit

        self.bins = list(map(lambda x: x[-1], chunk(np.arange(0, D), K)))
        self.combinations = [list(combinations(range(K), r))[0]]

        self.hash_tables = [{} for _ in range(self.L * self.t)]

        self.seed = [i + seed for i in range(t)]

    def hash_func(self, seed):
        return lambda x: mmh(key=x, seed=seed, positive=True) % self.D

    def hash_bin_to_bin(self, bin_id, seed):
        return mmh(key=bin_id, seed=seed) % self.K

    def hash_to_bucket_bbit(self, k_hashes):
        bbit_rep = map(lambda x: bin(x)[-self.bbit], k_hashes)
        return "".join(bbit_rep)

    def hash_to_bucket(self, k_hashes):
        return ".".join(map(lambda x: str(x), k_hashes))

    def hash_to_string(self, hashes):
        return "".join(map(lambda y: str(y), hashes))

    def one_permutation_exact_hash(self, word_set, seed):
        k_bins = [[] for _ in range(self.K)]
        hash_values = list(map(self.hash_func(seed), word_set))  # One permutation of data
        # put hash values into k bins
        for h in hash_values:
            for idx in range(self.K):
                if h <= self.bins[idx]:
                    k_bins[idx].append(h)
                    break
        # print(k_bins)
        k_hashes = list(map(lambda x: self.hash_to_string(x) if len(x) != 0 else "", k_bins))  # get min in each bin
        # print(k_hashes)

        # optimal densified hashing for empty bins
        for idx in range(self.K):
            if k_hashes[idx] == "":
                new_bin = self.hash_bin_to_bin(idx, seed)
                random.seed(seed)
                while k_hashes[new_bin] == -1:
                    new_bin = random.choice(range(self.K))
                k_hashes[idx] = k_hashes[new_bin]
        # print(k_hashes)
        return k_hashes

    def insert(self, x, id):
        for t_i in range(self.t):
            k_hashes = self.one_permutation_exact_hash(x, self.seed[t_i])
            for l in range(self.L):
                comb = self.combinations[l]
                hashcode = []
                for idx in comb:
                    hashcode.append(k_hashes[idx])
                key = self.hash_to_bucket(hashcode)
                table_id = self.L * t_i + l
                if key not in self.hash_tables[table_id]:
                    self.hash_tables[table_id][key] = set()
                self.hash_tables[table_id][key].add(id)

    def query(self, x):
        retrieved = []
        for t_i in range(self.t):
            k_hashes = self.one_permutation_exact_hash(x, self.seed[t_i])
            for l in range(self.L):
                comb = self.combinations[l]
                hashcode = []
                for idx in comb:
                    hashcode.append(k_hashes[idx])
                key = self.hash_to_bucket(hashcode)
                table_id = self.L * t_i + l
                retrieved.append(self.hash_tables[table_id].get(key, set()))
        return set.union(*retrieved)


if __name__ == '__main__':
    from MinHash_test import MinHash_tester

    K = 2
    r = 1
    t = 20
    D = 1000
    LSH = Partitioning_Hash(K=K, r=r, t=t, D=D)

    # x = [1, 2, 3, 4, 5]
    # LSH.ada_partition_hash([1, 2, 3, 4, 5], 0)
    # LSH.insert(x, 1)

    tester = MinHash_tester(Partitioning_Hash, seeds_per_table=t, K=K, r=r, t=t, D=D)
    tester.test_retrieval_prob()
