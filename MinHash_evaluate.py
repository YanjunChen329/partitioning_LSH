import sys
sys.path.append("./schemes")
from schemes.densifiedMinHash import Densified_MinHash
from schemes.partitioningMinHash import Partitioning_MinHash
from schemes.MinHash import MinHash
from schemes.partitioningHash import Partitioning_Hash
from schemes.AdaPartitionHash import AdaPartition_Hash

sys.path.append("./data")
from data.TestDataLoader import TestDataLoader
from data.WebspamDataLoader import WebspamDataLoader



import numpy as np
import pickle


D = 1000
query_n = 100


def Jaccard_similarity(s1, s2):
    intersection = len(set(s1).intersection(set(s2)))
    union = len(s1) + len(s2) - intersection
    return float(intersection) / union


def generate_datapoints(filename1, filename2):
    dictionary = np.arange(0, D)
    jaccard_sample = [0.3, 0.5, 0.65, 0.7, 0.75, 0.85, 0.90, 0.95]
    dataset = []
    query_set = []
    max_len = int(D * 0.8)
    min_len = int(D * 0.05)
    len_range = np.arange(min_len, max_len)

    for i in range(query_n):
        set_length = np.random.choice(len_range, 1)
        query_i = np.random.choice(dictionary, set_length, replace=False)
        query_set.append(str(list(query_i))[1:-1])

        # Generate sets with different jaccard distances from the query point
        for j in jaccard_sample:
            dups = np.random.choice(query_i, int(j * query_i.shape[0]), replace=False)

            added = np.random.choice(dictionary, int(np.random.uniform(0.02, 0.07)*query_i.shape[0]), replace=False)
            data = np.unique(np.array(list(dups) + list(added)))
            dataset.append(str(list(data))[1:-1])

    with open(filename1, 'w') as output:
        for line in query_set:
            output.write(line + "\n")

    with open(filename2, 'w') as output:
        for line in dataset:
            output.write(line + "\n")
    return query_set, dataset


def test_jaccard_freq(query_set, dataset):
    j_steps = np.arange(0.1, 1.01, 0.1)
    j_dict = {}
    for step in j_steps:
        j_dict[step] = 0.

    for q in query_set:
        for x in dataset:
            jaccard = Jaccard_similarity(q.split(", "), x.split(", "))
            for j in j_steps:
                if jaccard <= j:
                    j_dict[j] += 1./query_n
                    break
    for k in j_dict:
        print("{:.1f}: {:.3f}".format(k, j_dict[k]))


def open_dataset_files(query, data):
    query_set, dataset = [], []
    with open(query, 'r') as f1:
        for line in f1.readlines():
            query_set.append(line.split(", "))

    with open(data, 'r') as f2:
        for line in f2.readlines():
            dataset.append(line.split(", "))
    return query_set, dataset


def bruteForce_NN(q, S, jaccard):
    answer = set()
    for s_id in range(len(S)):
        if Jaccard_similarity(q, S[s_id]) >= jaccard:
            answer.add(s_id)
    return answer


def initialize_minhash(K, L, D, dataset=""):
    densified = MinHash(K, L, D)

    for s_id in range(len(S)):
        densified.insert(S[s_id], s_id)

    with open("./hashtable/minhash{}_K{}_L{}.pkl".format(dataset, K, L), 'wb') as output:
        pickle.dump(densified, output, pickle.HIGHEST_PROTOCOL)
    print("vanilla MinHash Initialized")


def initialize_densified(K, L, D, dataset=""):
    densified = Densified_MinHash(K, L, D)

    for s_id in range(len(S)):
        densified.insert(S[s_id], s_id)

    with open("./hashtable/densified{}_K{}_L{}.pkl".format(dataset, K, L), 'wb') as output:
        pickle.dump(densified, output, pickle.HIGHEST_PROTOCOL)
    print("densified MinHash Initialized")


def initialize_partitioning(K, r, C, t, D, dataset=""):
    partitioning = Partitioning_MinHash(K, r, C, D, t=t)

    for s_id in range(len(S)):
        partitioning.insert(S[s_id], s_id)

    with open("./hashtable/partitioning{}_K{}_r{}_C{}_t{}.pkl".format(dataset, K2, r, C, t), 'wb') as output:
        pickle.dump(partitioning, output, pickle.HIGHEST_PROTOCOL)
    print("partitioning MinHash Initialized")


def initialize_adaPartHash(J, L, D, dataset=""):
    adaPartHash = AdaPartition_Hash(J=J, L=L, D=D)

    for s_id in range(len(S)):
        adaPartHash.insert(S[s_id], s_id)

    with open("./hashtable/adaPartHash{}_J{}_L{}.pkl".format(dataset, int(100*J), L), 'wb') as output:
        pickle.dump(adaPartHash, output, pickle.HIGHEST_PROTOCOL)
    print("AdaPartition_Hash Initialized")


def comparison(minhash=None, densified=None, partitioning=None, adaPartHash=None):
    jaccard_lst = [0.7, 0.8, 0.9]

    m_recall, d_recall, p_recall, aph_recall = np.zeros(len(jaccard_lst)), np.zeros(len(jaccard_lst)), \
                                               np.zeros(len(jaccard_lst)), np.zeros(len(jaccard_lst))
    m_collisions, d_collisions, p_collisions, aph_collisions = 0, 0, 0, 0
    counter = np.zeros(len(jaccard_lst))

    for q_id in range(len(Q)):
        print(q_id)
        if minhash is not None:
            minhash_nn = minhash.query(Q[q_id])
            m_collisions += len(minhash_nn)
        if densified is not None:
            densified_nn = densified.query(Q[q_id])
            d_collisions += len(densified_nn)
        if partitioning is not None:
            partition_nn = partitioning.query(Q[q_id])
            p_collisions += len(partition_nn)
        if adaPartHash is not None:
            partitionHash_nn = adaPartHash.query(Q[q_id])
            aph_collisions += len(partitionHash_nn)

        for idx in range(len(jaccard_lst)):
            actual_nn = bruteForce_NN(Q[q_id], S, jaccard_lst[idx])
            if len(actual_nn) == 0:
                continue

            if minhash is not None:
                m_intersection = len(actual_nn.intersection(minhash_nn))
                m_recall[idx] += m_intersection / float(len(actual_nn))

            if densified is not None:
                d_intersection = len(actual_nn.intersection(densified_nn))
                d_recall[idx] += d_intersection / float(len(actual_nn))

            if partitioning is not None:
                p_intersection = len(actual_nn.intersection(partition_nn))
                p_recall[idx] += p_intersection / float(len(actual_nn))

            if adaPartHash is not None:
                aph_intersection = len(actual_nn.intersection(partitionHash_nn))
                aph_recall[idx] += aph_intersection / float(len(actual_nn))

            counter[idx] += 1

    for idx in range(len(jaccard_lst)):
        print("Target Jaccard: {}".format(jaccard_lst[idx]))
        print("Vanilla_MinHash -- recall: {:.4f}; avg collisions: {:.2f}"
              .format(m_recall[idx] / counter[idx], m_collisions / len(Q)))
        print("Densified_MinHash -- recall: {:.4f}; avg collisions: {:.2f}"
              .format(d_recall[idx] / counter[idx], d_collisions / len(Q)))
        print("Partitioning_MinHash -- recall: {:.4f}; avg collisions: {:.2f}"
              .format(p_recall[idx] / counter[idx], p_collisions / len(Q)))
        print("AdaPartition_Hash -- recall: {:.4f}; avg collisions: {:.2f}"
              .format(aph_recall[idx] / counter[idx], aph_collisions / len(Q)))
        print()


if __name__ == '__main__':
    dataset = "2"
    # Q, S = generate_uniform_datapoints("./data/testing/query_set2", "./data/testing/dataset2")
    # test_jaccard_freq(Q, S)
    Q, S = open_dataset_files("./data/testing/query_set" + dataset, "./data/testing/dataset" + dataset)
    print(len(Q), len(S))

    D = 1000    # dimensions of dictionary

    K = 10      # K hashes per hash table
    L = 50     # L hash tables
    # initialize_minhash(K, L, D, dataset)
    # initialize_densified(K, L, D, dataset)

    K2 = 5      # split the dictionary into K partitions
    r = 4       # use r partitions as hash codes
    C = 2       # each code are copied c times for a table
    t = 20      # use t hash table units [t*(k choose r) tables in total]
    # initialize_partitioning(K2, r, C, t, D, dataset)

    J = 0.9
    L2 = 50
    # initialize_adaPartHash(J, L2, D, dataset)

    minhash, densified, partitioning, adaPartHash = None, None, None, None
    with open("./hashtable/testing/minhash{}_K{}_L{}.pkl".format(dataset, K, L), 'rb') as input1:
        minhash = pickle.load(input1)

    # with open("./hashtable/testing/densified{}_K{}_L{}.pkl".format(dataset, K, L), 'rb') as input2:
    #     densified = pickle.load(input2)

    # with open("./hashtable/testing/partitioning{}_K{}_r{}_C{}_t{}.pkl".format(dataset, K2, r, C, t), 'rb') as input3:
    #     partitioning = pickle.load(input3)

    with open("./hashtable/testing/adaPartHash{}_J{}_L{}.pkl".format(dataset, int(100*J), L2), 'rb') as input4:
        adaPartHash = pickle.load(input4)

    # comparison(minhash, densified, partitioning, adaPartHash)



