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


class Partitioning_MinHash2(LSH):
    def __init__(self, K, r, C, L, D, bbit=2, seed=0):
        super(Partitioning_MinHash2, self).__init__("Partitioning_MinHash2")
        self.K = K          # split the dictionary into K partitions
        self.r = r          # use r partitions as hash codes
        self.C = C          # each code are copied c times for a table
        self.L = L          # use t hash table units (t*L tables in total)
        self.D = D          # dimensionality of dictionary
        self.bbit = bbit

        self.bins = list(map(lambda x: x[-1], chunk(np.arange(0, D), K)))
        self.combinations = list(combinations(range(K), r))[0]

        self.hash_tables = [{} for _ in range(self.L)]

        self.seed = [i + seed for i in range(C * L)]

    def hash_func(self, seed):
        return lambda x: mmh(key=x, seed=seed, positive=True) % self.D

    def hash_bin_to_bin(self, bin_id, seed):
        return mmh(key=bin_id, seed=seed) % self.K

    def hash_to_bucket_bbit(self, k_hashes):
        bbit_rep = map(lambda x: bin(x)[-self.bbit], k_hashes)
        return "".join(bbit_rep)

    def hash_to_bucket(self, k_hashes):
        return "".join(map(lambda x: str(x), k_hashes))

    def one_permutation_hash(self, word_set, seed):
        k_bins = [[] for _ in range(self.K)]
        hash_values = list(map(self.hash_func(seed), word_set))  # One permutation of data
        # put hash values into k bins
        for h in hash_values:
            for idx in range(self.K):
                if h <= self.bins[idx]:
                    k_bins[idx].append(h)
                    break
        # print(k_bins)
        k_hashes = list(map(lambda x: min(x) if len(x) != 0 else -1, k_bins))  # get min in each bin
        # print(k_hashes)

        # optimal densified hashing for empty bins
        for idx in range(self.K):
            if k_hashes[idx] == -1:
                new_bin = self.hash_bin_to_bin(idx, seed)
                random.seed(seed)
                while k_hashes[new_bin] == -1:
                    new_bin = random.choice(range(self.K))
                k_hashes[idx] = k_hashes[new_bin]
        # print(k_hashes)
        return k_hashes

    def insert(self, x, id):
        for l in range(self.L):
            hash_matrix = []
            for c in range(self.C):
                hash_matrix.append(self.one_permutation_hash(x, self.seed[self.C*l + c]))
            hash_matrix = np.array(hash_matrix)

            hashcode = []
            for idx in self.combinations:
                hashcode += list(hash_matrix[:, idx])
            key = self.hash_to_bucket(hashcode)
            table_id = l
            if key not in self.hash_tables[table_id]:
                self.hash_tables[table_id][key] = set()
            self.hash_tables[table_id][key].add(id)

    def query(self, x):
        retrieved = []
        for l in range(self.L):
            hash_matrix = []
            for c in range(self.C):
                hash_matrix.append(self.one_permutation_hash(x, self.seed[self.C*l + c]))
            hash_matrix = np.array(hash_matrix)

            hashcode = []
            for idx in self.combinations:
                hashcode += list(hash_matrix[:, idx])
            key = self.hash_to_bucket(hashcode)
            table_id = l
            retrieved.append(self.hash_tables[table_id].get(key, set()))
        return set.union(*retrieved)


if __name__ == '__main__':
    from MinHash_test import MinHash_tester

    K = 5
    r = 4
    C = 1
    L = 10
    D = 1000
    # LSH = Partitioning_MinHash(K=K, r=r, C=C, D=D)
    tester = MinHash_tester(Partitioning_MinHash2, seeds_per_table=C*L, K=K, r=r, C=C, L=L, D=D)
    tester.test_retrieval_prob()
