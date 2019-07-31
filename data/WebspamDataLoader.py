from sklearn.datasets import load_svmlight_file
from DataLoader import DataLoader


class WebspamDataLoader(DataLoader):
    def __init__(self, ratio=1., unigram=True):
        super(WebspamDataLoader, self).__init__("Webspam")
        self.unigram = unigram
        gram = "unigram" if unigram else "trigram"
        print("Webspam({}) Loading...".format(gram))
        if unigram:
            X, y = load_svmlight_file("./data/webspam/webspam_wc_normalized_unigram.svm", dtype=int)
            self.indices = X.indices.astype(int)
            self.indptr = X.indptr
            self.size = int((len(self.indptr) - 1) * ratio)
        else:
            self.filename = "./data/webspam/trigram{}.txt".format(int(ratio * 100))
            self.size = int(350000 * ratio)

        print("Webspam({}) Loaded".format(gram))
        print("***********************")

    def get_size(self):
        return self.size

    def get_item(self, id):
        if id > self.get_size():
            return None
        if self.unigram:
            start, end = self.indptr[id], self.indptr[id+1]
            return self.indices[start:end]
        else:
            with open(self.filename, 'r+') as infile:
                for i, line in enumerate(infile):
                    if i == id:
                        data = list(map(lambda x: int(x), line.split(",")))
                        return data


if __name__ == '__main__':
    data = WebspamDataLoader(unigram=False, ratio=0.1)
    print(data.get_item(3))

