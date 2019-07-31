import numpy as np
import sys
import os
import heapq


def Jaccard_similarity(s1, s2):
    intersection = len(set(s1).intersection(set(s2)))
    union = len(s1) + len(s2) - intersection
    return float(intersection) / union


def bruteForce_jaccard(q, data_loader, dist_lst, distance_metric):
    length = len(dist_lst)
    answer = [set() for _ in range(length)]
    for s_id in range(data_loader.get_size()):
        s_data = data_loader.get_item(s_id)
        similarity = distance_metric(q, s_data)
        for idx in range(length):
            if similarity >= dist_lst[idx]:
                answer[idx].add(s_id)
    return answer


def bruteForce_kNN(q_id, k, data_loader, distance_metric):
    max_heap = []
    q = data_loader.get_item(q_id)
    for s_id in range(data_loader.get_size()):
        s_data = data_loader.get_item(s_id)
        similarity = distance_metric(q, s_data)
        if len(max_heap) < k:
            heapq.heappush(max_heap, (-similarity, s_id))
        else:
            heapq.heappushpop(max_heap, (-similarity, s_id))
    return list(map(lambda x: x[1], max_heap))



class LSH_evaluator:
    def __init__(self, LSH_list, query_set, data_loader, distance_metric="Jaccard"):
        self.LSH_list = LSH_list
        self.query_set = query_set
        self.data_loader = data_loader
        if distance_metric == "Jaccard":
            self.metric = Jaccard_similarity

    def build_groundtruth_k(self, k, filename):
        groundtruth = np.zeros((len(self.query_set), k), dtype=np.int)
        for i in range(len(self.query_set)):
            sys.stdout.write("\r*** Building groundtruth: {:.2f}% ***".format(100. * i / k))
            sys.stdout.flush()
            q = self.query_set[i]
            kNN = bruteForce_kNN(q, k, self.data_loader, self.metric)
            groundtruth[i, :] = kNN
            break
        np.savetxt(filename, groundtruth)
        print()
        return groundtruth

    def experiment_kNN(self, k, groundtruth_file):
        print("***********************")
        print("Experiment begins")
        if os.path.exists(groundtruth_file):
            groundtruth = np.loadtxt(groundtruth_file, dtype=np.int)
        else:
            groundtruth = self.build_groundtruth_k(k, groundtruth_file)

        # print out details of input LSH
        recall = np.zeros(len(self.LSH_list), dtype=float)
        precision = np.zeros(len(self.LSH_list), dtype=float)
        collisions = np.zeros(len(self.LSH_list), dtype=float)

        query_size = len(self.query_set)
        for idx in range(query_size):
            sys.stdout.write("\rQuerying: {:.2f}%".format(100. * idx / query_size))
            sys.stdout.flush()
            q_data = self.data_loader.get_item(self.query_set[idx])

            actual_nn = set(groundtruth[idx, :])
            for i in range(len(self.LSH_list)):
                LSH = self.LSH_list[i]
                result_i = LSH.query(q_data)
                intersection = actual_nn.intersection(result_i)
                recall[i] += len(intersection) / float(len(actual_nn))
                precision[i] += len(intersection) / float(len(result_i))
                collisions[i] += len(result_i)

        print("\n")
        for i in range(len(self.LSH_list)):
            print("{} -- recall: {:.4f}; precision: {:.4f}; avg collisions: {:.2f}"
                  .format(self.LSH_list[i].__name__, recall[i] / query_size,
                          precision[i] / query_size, collisions[i] / query_size))

        # print time consumption
        print("\nTime Consumption")
        for lsh in self.LSH_list:
            print("{} -- insert time: {:.6f}; query time: {:.6f}"
                  .format(lsh.__name__, lsh.get_insert_time(), lsh.get_query_time()))


    def experiment_jaccard(self, jaccard_lst, average=True):
        if average:
            self.experiment_avg(jaccard_lst)
        else:
            self.experiment_sum(jaccard_lst)

    def experiment_sum(self, jaccard_lst):
        print("***********************")
        print("Experiment begins")

        intersection_count = np.zeros((len(jaccard_lst), len(self.LSH_list)), dtype=float)
        actual_count = np.zeros(len(jaccard_lst), dtype=float)
        collisions = np.zeros(len(self.LSH_list), dtype=float)

        query_size = len(self.query_set)
        for idx in range(query_size):
            sys.stdout.write("\rQuerying: {:.2f}%".format(100. * idx / query_size))
            sys.stdout.flush()
            q_data = self.data_loader.get_item(self.query_set[idx])

            lsh_nn_lst = []
            for i in range(len(self.LSH_list)):
                LSH = self.LSH_list[i]
                lsh_nn_lst.append(LSH.query(q_data))
                collisions[i] += len(lsh_nn_lst[i])

            actual_nn_lst = bruteForce_jaccard(q_data, self.data_loader, jaccard_lst, self.metric)
            for j in range(len(jaccard_lst)):
                actual_nn = actual_nn_lst[j]
                actual_count[j] += len(actual_nn)

                for k in range(len(self.LSH_list)):
                    intersection = actual_nn.intersection(lsh_nn_lst[k])
                    intersection_count[j][k] += len(intersection)

        print("\n")
        for i in range(len(jaccard_lst)):
            print("Target Jaccard: {}".format(jaccard_lst[i]))
            for j in range(len(self.LSH_list)):
                recall = intersection_count[i][j] / actual_count[i]
                precision = intersection_count[i][j] / collisions[j]
                print("{} -- recall: {:.4f}; precision: {:.4f}; avg collisions: {:.2f}"
                      .format(self.LSH_list[j].__name__, recall, precision, collisions[j] / query_size))

        # print time consumption
        print("\nTime Consumption")
        for lsh in self.LSH_list:
            print("{} -- insert time: {:.6f}; query time: {:.6f}"
                  .format(lsh.__name__, lsh.get_insert_time(), lsh.get_query_time()))

    def experiment_avg(self, jaccard_lst):
        print("***********************")
        print("Experiment begins")
        # print out details of input LSH
        recall = np.zeros((len(jaccard_lst), len(self.LSH_list)), dtype=float)
        precision = np.zeros((len(jaccard_lst), len(self.LSH_list)), dtype=float)
        collisions = np.zeros(len(self.LSH_list), dtype=float)
        counter = np.zeros(len(jaccard_lst), dtype=float)

        query_size = len(self.query_set)
        for idx in range(query_size):
            sys.stdout.write("\rQuerying: {:.2f}%".format(100. * idx / query_size))
            sys.stdout.flush()
            q_data = self.data_loader.get_item(self.query_set[idx])

            nn_lst = []
            for i in range(len(self.LSH_list)):
                LSH = self.LSH_list[i]
                nn_lst.append(LSH.query(q_data))
                collisions[i] += len(nn_lst[i])

            actual_nn_lst = bruteForce_jaccard(q_data, self.data_loader, jaccard_lst, self.metric)
            for j in range(len(jaccard_lst)):
                actual_nn = actual_nn_lst[j]
                if len(actual_nn) == 0:
                    continue

                for k in range(len(self.LSH_list)):
                    intersection = actual_nn.intersection(nn_lst[k])
                    recall[j][k] += len(intersection) / float(len(actual_nn))
                    precision[j][k] += len(intersection) / float(len(nn_lst[k]))

                counter[j] += 1

        print("\n")
        for i in range(len(jaccard_lst)):
            print("Target Jaccard: {}".format(jaccard_lst[i]))
            for j in range(len(self.LSH_list)):
                print("{} -- recall: {:.4f}; precision: {:.4f}; avg collisions: {:.2f}"
                      .format(self.LSH_list[j].__name__, recall[i][j] / counter[i],
                              precision[i][j] / counter[i], collisions[j] / query_size))

        # print time consumption
        print("\nTime Consumption")
        for lsh in self.LSH_list:
            print("{} -- insert time: {:.6f}; query time: {:.6f}"
                  .format(lsh.__name__, lsh.get_insert_time(), lsh.get_query_time()))
