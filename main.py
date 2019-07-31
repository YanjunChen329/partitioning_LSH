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

from LSH_evaluator import LSH_evaluator

import numpy as np
import pickle
import sys


def initialize_LSH(filename, dataloader, LSH, **kwargs):
    lsh = LSH(**kwargs)

    for s_id in range(dataloader.get_size()):
        sys.stdout.write("\rInitializing {}: {:.3f}%".format(lsh.__name__, 100. * s_id / dataloader.get_size()))
        sys.stdout.flush()
        lsh.insert(dataloader.get_item(s_id), s_id)

    with open(filename, 'wb') as output:
        pickle.dump(lsh, output, pickle.HIGHEST_PROTOCOL)
    print()


def load_LSH(filename):
    with open(filename, 'rb') as input1:
        lsh = pickle.load(input1)
    print("{} Loaded".format(lsh.__name__))
    return lsh


def testing_experiment():
    # ------------------- Data Preparation ---------------------
    dataset_id = "3"
    test_dataloader = TestDataLoader("dataset" + dataset_id)
    query_set = list(range(100))
    LSH_list = []
    D = 10000

    # ----------------------- MinHash ----------------------
    mh_kwargs = {"K": 30, "L": 50, "D": D}
    mh_name = "./hashtable/testing/minhash{}_K{K}_L{L}.pkl".format(dataset_id, **mh_kwargs)
    # initialize_LSH(mh_name, test_dataloader, MinHash, **mh_kwargs)
    # minhash = load_LSH(mh_name)
    # LSH_list.append(minhash)

    # ------------------- Densified MinHash --------------------
    d_kwargs = {"K": 30, "L": 50, "D": D}
    d_name = "./hashtable/testing/time_densified{}_K{K}_L{L}.pkl".format(dataset_id, **d_kwargs)
    # initialize_LSH(d_name, test_dataloader, Densified_MinHash, **d_kwargs)
    densified = load_LSH(d_name)
    print(densified.get_time_details())
    LSH_list.append(densified)

    # ------------------- AdaPartition_Hash --------------------
    a_kwargs = {"L": 50, "J": 0.9, "c": 0.95, "D": D}
    a_name = "./hashtable/testing/adaPartHash{}_J{}_L{}_c{}.pkl".format(dataset_id, int(100 * a_kwargs["J"]),
                                                                           a_kwargs["L"], len(str(a_kwargs["c"]))-2)
    # initialize_LSH(a_name, test_dataloader, AdaPartition_Hash, **a_kwargs)
    adapart = load_LSH(a_name)
    print(adapart.get_time_details())
    LSH_list.append(adapart)

    # ---------------------- Experiment -------------------------
    evaluator = LSH_evaluator(LSH_list, query_set, test_dataloader)
    evaluator.experiment([0.7, 0.8, 0.9])


def webspam_unigram_experiment():
    # ------------------- Data Preparation ---------------------
    ratio = 0.2
    webspam_dataloader = WebspamDataLoader(ratio=ratio)
    np.random.seed(0)
    query_set = np.random.choice(range(webspam_dataloader.get_size()), size=100, replace=False)
    LSH_list = []
    D = 10000

    # ----------------------- MinHash ----------------------
    mh_kwargs = {"K": 30, "L": 50, "D": D}
    mh_name = "./hashtable/webspam/unigram/minhash{}_K{K}_L{L}.pkl".format(int(ratio*100), **mh_kwargs)
    # initialize_LSH(mh_name, webspam_dataloader, MinHash, **mh_kwargs)
    # minhash = load_LSH(mh_name)
    # LSH_list.append(minhash)

    # ------------------- Densified MinHash --------------------
    d_kwargs = {"K": 30, "L": 50, "D": D}
    d_name = "./hashtable/webspam/unigram/densified{}_K{K}_L{L}.pkl".format(int(ratio*100), **d_kwargs)
    # initialize_LSH(d_name, webspam_dataloader, Densified_MinHash, **d_kwargs)
    densified = load_LSH(d_name)
    print(densified.get_time_details())
    LSH_list.append(densified)

    # ------------------- AdaPartition_Hash --------------------
    a_kwargs = {"L": 50, "J": 0.9, "c": 0.95, "D": D}
    a_name = "./hashtable/webspam/unigram/adaPartHash{}_J{}_L{}_c{}.pkl".format(
        int(ratio*100), int(100 * a_kwargs["J"]), a_kwargs["L"], int(10000 * a_kwargs["c"]))
    # initialize_LSH(a_name, webspam_dataloader, AdaPartition_Hash, **a_kwargs)
    adapart = load_LSH(a_name)
    print(adapart.get_time_details())
    LSH_list.append(adapart)

    # ---------------------- Experiment -------------------------
    evaluator = LSH_evaluator(LSH_list, query_set, webspam_dataloader)
    evaluator.experiment([0.7, 0.8, 0.9])


def webspam_trigram_experiment():
    # ------------------- Data Preparation ---------------------
    ratio = 0.1
    webspam_dataloader = WebspamDataLoader(ratio=ratio, unigram=False)
    np.random.seed(0)
    query_set = np.random.choice(range(webspam_dataloader.get_size()), size=100, replace=False)
    LSH_list = []
    D = 100000

    # ----------------------- MinHash ----------------------
    mh_kwargs = {"K": 30, "L": 50, "D": D}
    mh_name = "./hashtable/webspam/trigram/minhash{}_K{K}_L{L}.pkl".format(int(ratio*100), **mh_kwargs)
    # initialize_LSH(mh_name, webspam_dataloader, MinHash, **mh_kwargs)
    # minhash = load_LSH(mh_name)
    # LSH_list.append(minhash)

    # ------------------- Densified MinHash --------------------
    d_kwargs = {"K": 30, "L": 50, "D": D}
    d_name = "./hashtable/webspam/trigram/densified{}_K{K}_L{L}.pkl".format(int(ratio*100), **d_kwargs)
    # initialize_LSH(d_name, webspam_dataloader, Densified_MinHash, **d_kwargs)
    densified = load_LSH(d_name)
    print(densified.get_time_details())
    print(densified.get_insert_time())
    LSH_list.append(densified)

    # ------------------- AdaPartition_Hash --------------------
    a_kwargs = {"L": 50, "J": 0.9, "c": 0.9, "D": D}
    a_name = "./hashtable/webspam/trigram/adaPartHash{}_J{}_L{}_c{}.pkl".format(
        int(ratio*100), int(100 * a_kwargs["J"]), a_kwargs["L"], int(10000 * a_kwargs["c"]))
    # initialize_LSH(a_name, webspam_dataloader, AdaPartition_Hash, **a_kwargs)
    adapart = load_LSH(a_name)
    print(adapart.get_time_details())
    print(adapart.get_insert_time())
    LSH_list.append(adapart)

    # ---------------------- Experiment -------------------------
    evaluator = LSH_evaluator(LSH_list, query_set, webspam_dataloader)
    evaluator.experiment([0.7, 0.8, 0.9])



if __name__ == '__main__':
    # testing_experiment()
    # webspam_unigram_experiment()
    webspam_trigram_experiment()



