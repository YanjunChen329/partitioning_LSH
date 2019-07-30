import numpy as np
import math

def get_Ngram(A, n=3):
    ngram = set()
    for i in range(len(A) - n + 1):
        ngram.add(A[i:i+n])
    return list(ngram)


def Jaccard_similarity(s1, s2):
    intersection = len(set(s1).intersection(set(s2)))
    union = len(s1) + len(s2) - intersection
    return float(intersection) / union


def OPH_retreival_prob(m, n, K, L):
    prob = 1
    for i in range(K):
        if m-i <= 0:
            prob *= float(m) / n
        else:
            prob *= float(m-i) / (n-i)
    return 1 - (1 - prob)**L


def partitioning_retrieval_prob(m, n, K, c, t):
    r = K-1
    p1 = 1
    for i in range(r):
        p1 *= 1. * (m-i)/(n-i)
    p1 **= c
    p2 = (K * (1.*(n-m)/(n-K)))**c + (1. * (m-K)/(n-K))**c
    p_match = p1 * p2
    # print(p1, p2, p_match)
    P = 1 - (1 - p_match)**t
    return P


def partitioning2_retrieval_prob(m, n, K, r, c, L):
    p1 = 1
    for i in range(r-1):
        p1 *= 1. * (m-i)/(n-i)
    p1 *= 1. * m / n
    p_match = p1 ** c
    # print(p1, p2, p_match)
    P = 1 - (1 - p_match)**L
    return P


def partitionhash_retrieval_prob(m, n, K, r, t):
    ratio = float(r) / K
    p_match = (1 - ratio)**(n-m)
    return 1 - (1 - p_match)**t


def partitionhash_retrieval_prob2(m, n, ratio, L):
    p_match = (1 - ratio)**(n-m)
    return 1 - (1 - p_match)**L


def autoparthash_retrieval_prob(m, n, r, L):
    d = int(n * r)
    print(m, n, d)
    p_match = (1. * math.factorial(m) / math.factorial(m-d)) * (1. * math.factorial(n-d) / math.factorial(n))\
        if d < m else 0

    return 1 - (1 - p_match)**L


def adaPartHash_get_ratio(n, J, L, c):
    return round(1 - (1 - (1 - c)**(1./L))**(1./math.ceil((1-J)*n)), 2)


class MinHash_tester:
    def __init__(self, LSH_class, seeds_per_table, **kwargs):
        self.LSH_class = LSH_class
        self.seeds_per_table = seeds_per_table
        self.kwargs = kwargs

        print(LSH_class, kwargs)

    def test_retrieval_prob(self):
        s1 = "cdefghijklmn"
        s2 = "cdefghijklm"
        s3 = "defghijklmno"
        s4 = "fghijklmn"
        s5 = "fghijklmnopq"
        s6 = "ghijklmnopqr"
        s_set = [s2, s3, s4, s5, s6]
        x1 = get_Ngram(s1)
        x_set = list(map(lambda s: get_Ngram(s), s_set))
        intersection_set = list(map(lambda x: len(set(x1).intersection(set(x))), x_set))
        union_set = list(map(lambda x: len(set(x1).union(set(x))), x_set))
        J_set = [float(x) / y for x, y in zip(intersection_set, union_set)]
        for i in range(len(J_set)):
            print("Jaccard 1&{} = {}/{} = {:.4f}".format(i+2, intersection_set[i], union_set[i], J_set[i]))

        # Test for Retrieval probability
        counters = [0] * len(s_set)
        n1 = 1000
        for i in range(n1):
            LSH = self.LSH_class(**self.kwargs, seed=i*100*self.seeds_per_table)
            LSH.insert(x1, 1)
            for j in range(len(counters)):
                counters[j] += int(len(LSH.query(x_set[j])) > 0)

        p_theory = [0] * len(s_set)
        if LSH.__name__ == "Densified_MinHash":
            K, L = self.kwargs["K"], self.kwargs["L"]
            p_theory = [OPH_retreival_prob(m, n, K, L) for m, n in zip(intersection_set, union_set)]
        elif LSH.__name__ == "MinHash":
            K, L = self.kwargs["K"], self.kwargs["L"]
            p_theory = [1 - (1 - J**K)**L for J in J_set]
        elif LSH.__name__ == "Partitioning_MinHash":
            K, C, t = self.kwargs["K"], self.kwargs["C"], self.kwargs["t"]
            p_theory = [partitioning_retrieval_prob(m, n, K, C, t) for m, n in zip(intersection_set, union_set)]
        elif LSH.__name__ == "Partitioning_MinHash2":
            K, r, C, L = self.kwargs["K"], self.kwargs["r"], self.kwargs["C"], self.kwargs["L"]
            p_theory = [partitioning2_retrieval_prob(m, n, K, r, C, L) for m, n in zip(intersection_set, union_set)]
        elif LSH.__name__ == "Partitioning_Hash":
            K, r, t = self.kwargs["K"], self.kwargs["r"], self.kwargs["t"]
            p_theory = [partitionhash_retrieval_prob(m, n, K, r, t) for m, n in zip(intersection_set, union_set)]
        elif LSH.__name__ == "AdaPartition_Hash":
            L, J, c = LSH.L, LSH.J, LSH.c
            p_theory = [partitionhash_retrieval_prob2(m, n, adaPartHash_get_ratio(n, J, L, c), L)
                        for m, n in zip(intersection_set, union_set)]

        for z in range(len(s_set)):
            print("P({:.4f} Retrieval) -- theoretical: {:.3f}; actual:{:.3f}".format(J_set[z], p_theory[z], counters[z]/float(n1)))

    def test_jaccard(self):
        s1 = "cdefghijklmn"
        s2 = "defghijklmno"
        x1 = get_Ngram(s1)
        x2 = get_Ngram(s2)

        # Test for Jaccard Similarity
        K = self.kwargs['K']
        counter2 = np.zeros(K)
        for i in range(10000):
            LSH = self.LSH_class(**self.kwargs, seed=i*self.seeds_per_table)
            if LSH.__name__ == "Densified_MinHash":
                h1 = LSH.ada_partition_hash(x1, i)
                h2 = LSH.ada_partition_hash(x2, i)
            elif LSH.__name__ == "MinHash":
                h1 = LSH.generate_k_minhash(x1, i)
                h2 = LSH.generate_k_minhash(x2, i)
            for j in range(K):
                counter2[j] += int(h1[j] == h2[j])
            # counter2 += Jaccard_similarity(h1, h2)
        print("Actual Jaccard: {}".format(counter2/10000.))

