from sklearn.utils import murmurhash3_32 as mmh
import numpy as np
import random


def get_Ngram(A, n=3):
    ngram = set()
    for i in range(len(A) - n + 1):
        ngram.add(A[i:i+n])
    return list(ngram)


def Jaccard_similarity(s1, s2):
    intersection = len(set(s1).intersection(set(s2)))
    union = len(s1) + len(s2) - intersection
    return float(intersection) / union


def chunk(xs, n):
    """Split the list, xs, into n evenly sized chunks"""
    L = len(xs)
    assert 0 < n <= L
    s, r = divmod(L, n)
    t = s + 1
    return [xs[p:p+t] for p in range(0, r*t, t)] + [xs[p:p+s] for p in range(r*t, L, s)]


class Densified_MinHash:
    def __init__(self, k, L, D, bbit=2, seed=0):
        self.k = k
        self.L = L
        self.D = D
        self.bbit = bbit
        self.bins = list(map(lambda x: x[-1], chunk(np.arange(0, D), k)))
        self.hash_tables = [{} for _ in range(L)]

        self.seed = [i + seed for i in range(L)]

    def hash_func(self, seed):
        return lambda x: mmh(key=x, seed=seed) % self.D

    def universal_hashing(self, bin_id, seed):
        return mmh(key=bin_id, seed=seed) % self.k

    def b_bit_hashing(self, k_hashes):
        bbit_rep = map(lambda x: bin(x)[-self.bbit], k_hashes)
        return "".join(bbit_rep)

    def hash_to_bucket(self, k_hashes):
        return "".join(map(lambda x: str(x), k_hashes))

    def one_permutation_hash(self, word_set, seed):
        k_bins = [[] for _ in range(self.k)]
        hash_values = list(map(self.hash_func(seed), word_set))  # One permutation of data
        # put hash values into k bins
        for h in hash_values:
            for idx in range(self.k):
                if h <= self.bins[idx]:
                    k_bins[idx].append(h)
                    break
        # print(k_bins)
        k_hashes = list(map(lambda x: min(x) if len(x) != 0 else -1, k_bins))  # get min in each bin
        # print(k_hashes)

        # optimal densified hashing for empty bins
        for idx in range(self.k):
            if k_hashes[idx] == -1:
                new_bin = self.universal_hashing(idx, seed)
                random.seed(seed)
                while k_hashes[new_bin] == -1:
                    new_bin = random.choice(range(self.k))
                k_hashes[idx] = k_hashes[new_bin]
        # print(k_hashes)
        return k_hashes

    def insert(self, x, id):
        for l in range(self.L):
            k_hashes = self.one_permutation_hash(x, self.seed[l])
            key = self.hash_to_bucket(k_hashes)
            if key not in self.hash_tables[l]:
                self.hash_tables[l][key] = set()
            self.hash_tables[l][key].add(id)

    def query(self, x):
        retrieved = []
        for l in range(self.L):
            k_hashes = self.one_permutation_hash(x, self.seed[l])
            key = self.hash_to_bucket(k_hashes)
            retrieved.append(self.hash_tables[l].get(key, set()))
        return set.union(*retrieved)


if __name__ == '__main__':
    s1 = "cdefghijklmn"
    s2 = "defghijklmno"
    x1 = get_Ngram(s1)
    x2 = get_Ngram(s2)
    J = Jaccard_similarity(x1, x2)
    print("Jaccard: {}".format(J))

    K = 5
    L = 10
    D = 1000
    # densified = Densified_MinHash(K, L, D)
    # print(densified.bins)
    # data = np.random.choice(np.arange(0, 1000), 10, replace=False)
    # densified.one_permutation_hash(data, 0)

    # Test for Retrieval probability
    counter = 0
    for i in range(1000):
        densified = Densified_MinHash(K, L, D, seed=i*(L+1))
        densified.insert(x1, 1)
        counter += int(len(densified.query(x2)) > 0)
        # print()
    print("P(Retrieval) -- theoretical: {:.3f}; actual:{}".format(1-(1-J**K)**L, counter/1000.))

    # Test for Jaccard Similarity
    counter2 = np.zeros(K)
    for i in range(10000):
        densified = Densified_MinHash(K, L, D, seed=i*L)
        h1 = densified.one_permutation_hash(x1, i)
        h2 = densified.one_permutation_hash(x2, i)
        for j in range(K):
            counter2[j] += int(h1[j] == h2[j])
        # counter2 += Jaccard_similarity(h1, h2)
    print("Actual Jaccard: {}".format(counter2/10000.))




