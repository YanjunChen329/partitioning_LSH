from DataLoader import DataLoader
import numpy as np

D = 1000
query_n = 100


directory = "./data/testing/"


def Jaccard_similarity(s1, s2):
    intersection = len(set(s1).intersection(set(s2)))
    union = len(s1) + len(s2) - intersection
    return float(intersection) / union


def generate_uniform_datapoints(filename1, filename2):
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


def generate_large_dataset(filename):
    dim = 256
    dictionary = np.arange(0, dim)
    jaccard_sample = [0.55, 0.65, 0.75, 0.85, 0.95]
    points_per_j = [1, 1, 5, 9, 9]
    dataset = []
    query_set = []
    max_len = int(dim * 0.8)
    min_len = int(dim * 0.4)
    len_range = np.arange(min_len, max_len)

    for i in range(query_n):
        query_length = np.random.choice(len_range, 1)
        query_i = np.random.choice(dictionary, query_length, replace=False)
        query_set.append(str(list(query_i))[1:-1])

        # Generate sets with different jaccard distances from the query point
        for j in range(len(jaccard_sample)):
            jaccard = jaccard_sample[j]
            # the query point can be the longer/shorter/even-length vector
            ratio = jaccard / (float(query_length) / dim)
            ratio = 1. if ratio > 1 else ratio
            intersect_ratio = np.random.uniform(jaccard, ratio, size=points_per_j[j])
            for k in range(intersect_ratio.shape[0]):
                intersections = np.random.choice(query_i, int(intersect_ratio[k] * query_i.shape[0]), replace=False)
                added_length = int(intersections.shape[0] / jaccard) - query_length
                added = np.random.choice(dictionary, added_length, replace=False)
                data = np.unique(np.array(list(intersections) + list(added)))
                dataset.append(str(list(data))[1:-1])
                # print(intersect_ratio[k], query_length, intersections.shape[0], added_length)

    dataset = query_set + dataset
    with open(filename, 'w') as output:
        for line in dataset:
            output.write(line + "\n")
    return dataset


def test_jaccard_freq(filename):
    dataset = []
    with open(filename, 'r') as f2:
        for line in f2.readlines():
            dataset.append(line.split(", "))
    query_set = dataset[:query_n]
    dataset = dataset[query_n:]
    j_steps = np.arange(0.1, 1.01, 0.1)
    j_dict = {}
    for step in j_steps:
        j_dict[step] = 0.

    for q in query_set:
        for x in dataset:
            jaccard = Jaccard_similarity(q, x)
            for j in j_steps:
                if jaccard <= j:
                    j_dict[j] += 1./query_n
                    break
    for k in j_dict:
        print("{:.1f}-{:.1f}: {:.3f}".format(k-0.1, k, j_dict[k]))
    print("total:", len(dataset))


class TestDataLoader(DataLoader):
    def __init__(self, data_file):
        super(TestDataLoader, self).__init__("Test_set")
        print("Testset {} Loading...".format(data_file))

        self.dataset = []
        filename = directory + data_file
        with open(filename, 'r') as f2:
            for line in f2.readlines():
                self.dataset.append(line.split(", "))
        self.size = len(self.dataset)
        print("Testset {} Loaded".format(data_file))
        print("***********************")

    def get_size(self):
        return self.size

    def get_item(self, id):
        return self.dataset[id]

if __name__ == '__main__':
    # data_loader = TestDataLoader("dataset1")
    # print(data_loader.get_item(10))
    dataset = generate_large_dataset("./testing/dataset3")
    test_jaccard_freq("./testing/dataset3")
