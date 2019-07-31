import sys

trigram_file = "./webspam_wc_normalized_trigram.svm"

trigram10 = "./trigram10.txt"
trigram20 = "./trigram20.txt"
trigram50 = "./trigram50.txt"
trigram100 = "./trigram100.txt"

N = 350000
counter = 0
with open(trigram_file, "r+") as infile, open(trigram10, "w+") as out10, open(trigram20, "w+") as out20, open(trigram50, "w+") as out50, open(trigram100, "w+") as out100:
    for line in infile:
        sys.stdout.write("\r{:.3f}%".format(100. * counter / N))
        sys.stdout.flush()
        arr = line.split(" ")
        vector = list(map(lambda x: x.split(":")[0], arr[1:]))
        if counter < 0.1 * N:
            out10.write(",".join(vector) + "\n")
        if counter < 0.2 * N:
            out20.write(",".join(vector) + "\n")
        if counter < 0.5 * N:
            out50.write(",".join(vector) + "\n")
        out100.write(",".join(vector) + "\n")


