from densifiedMinHash import Densified_MinHash
from partitioningMinHash import Partitioning_MinHash
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
    jaccard_sample = np.arange(0.15, 1, 0.1)
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


def initialize_densified():
    densified = Densified_MinHash(K, L, D)

    for s_id in range(len(S)):
        densified.insert(S[s_id], s_id)

    with open("./hashtable/densified_K{}_L{}.pkl".format(K, L), 'wb') as output:
        pickle.dump(densified, output, pickle.HIGHEST_PROTOCOL)
    print("densified MinHash Initialized")


def initialize_partitioning():
    partitioning = Partitioning_MinHash(K2, r, C, D, t=t)

    for s_id in range(len(S)):
        partitioning.insert(S[s_id], s_id)

    with open("./hashtable/partitioning_K{}_r{}_C{}_t{}.pkl".format(K2, r, C, t), 'wb') as output:
        pickle.dump(partitioning, output, pickle.HIGHEST_PROTOCOL)
    print("partitioning MinHash Initialized")


def comparison(densified, partitioning):
    jaccard = 0.75

    d_recall, p_recall = 0, 0
    d_collisions, p_collisions = 0, 0
    counter = 0

    for q_id in range(len(Q)):
        print(q_id)
        actual_nn = bruteForce_NN(Q[q_id], S, jaccard)
        if len(actual_nn) == 0:
            continue
        densified_nn = densified.query(Q[q_id])
        partition_nn = partitioning.query(Q[q_id])

        d_intersection = len(actual_nn.intersection(densified_nn))
        d_recall += d_intersection / float(len(actual_nn))
        d_collisions += len(densified_nn)

        p_intersection = len(actual_nn.intersection(partition_nn))
        p_recall += p_intersection / float(len(actual_nn))
        p_collisions += len(partition_nn)

        counter += 1

    print("Densified_MinHash -- recall: {:.4f}; avg collisions: {:.2f}"
          .format(d_recall / counter, d_collisions / counter))
    print("Partitioning_MinHash -- recall: {:.4f}; avg collisions: {:.2f}"
          .format(p_recall / counter, p_collisions / counter))


if __name__ == '__main__':
    # Q, S = generate_datapoints("./data/query_set", "./data/dataset")
    # test_jaccard_freq(Q, S)
    Q, S = open_dataset_files("./data/query_set", "./data/dataset")
    print(len(Q), len(S))

    D = 1000

    K = 10
    L = 100
    # initialize_densified()

    K2 = 5
    r = 4
    C = 3
    t = 20
    # initialize_partitioning()

    with open("./hashtable/densified_K{}_L{}.pkl".format(K, L), 'rb') as input1:
        densified = pickle.load(input1)

    with open("./hashtable/partitioning_K{}_r{}_C{}_t{}.pkl".format(K2, r, C, t), 'rb') as input2:
        partitioning = pickle.load(input2)

    comparison(densified, partitioning)


