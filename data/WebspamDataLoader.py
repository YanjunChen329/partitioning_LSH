from sklearn.datasets import load_svmlight_file
from DataLoader import DataLoader


class WebspamDataLoader(DataLoader):
    def __init__(self, ratio=1., unigram=True):
        super(WebspamDataLoader, self).__init__("Webspam")
        print("Webspam Loading...")
        if unigram:
            X, y = load_svmlight_file("partitioning_LSH/data/webspam/webspam_wc_normalized_unigram.svm", dtype=int)
            self.indices = X.indices.astype(int)
            self.indptr = X.indptr
            self.size = int((len(self.indptr) - 1) * ratio)
        else:
            counter = 0
            with open("D:/User/Documents/College/anshu_research/partitioning_lsh/"
                                      "data/webspam/webspam_wc_normalized_trigram.svm") as infile:
                for line in infile:
                    print(line)
                    print(len(line.split(":")))
                    counter += 1
                    if counter > 1:
                        exit()

        print("Webspam Loaded")
        print("***********************")

    def get_size(self):
        return self.size

    def get_item(self, id):
        start, end = self.indptr[id], self.indptr[id+1]
        return self.indices[start:end]


if __name__ == '__main__':
    data = WebspamDataLoader(unigram=False)
    # print(data.get_item(3))

