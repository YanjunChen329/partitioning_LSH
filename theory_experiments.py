import numpy as np
import itertools
from math import factorial, log
import matplotlib.pyplot as plt
import math


def permutation_exp():
    n = factorial(8)
    s1 = [1, 1, 1, 1, 0, 1, 1, 0]
    s2 = [0, 1, 1, 0, 1, 1, 1, 1]
    same1, same2, same3, same4 = 0., 0., 0., 0.
    same12, same23, same34, same41 = 0., 0., 0., 0.
    same123, same234, same134, same124 = 0., 0., 0., 0.
    same_trio = 0.
    same1234 = 0.
    x1 = [0] * 8
    x2 = [0] * 8
    for permu in itertools.permutations(range(8)):
        idx0, idx2, idx4, idx6 = permu[0], permu[2], permu[4], permu[6]
        x1[0], x2[0] = s1[idx0], s2[idx0]
        x1[2], x2[2] = s1[idx2], s2[idx2]
        x1[4], x2[4] = s1[idx4], s2[idx4]
        x1[6], x2[6] = s1[idx6], s2[idx6]
        # np.random.seed(i)
        # x1 = np.random.permutation(s1)
        # np.random.seed(i)
        # x2 = np.random.permutation(s2)
        same1 += int(x1[0] == x2[0])
        same2 += int(x1[2] == x2[2])
        same3 += int(x1[4] == x2[4])
        same4 += int(x1[6] == x2[6])

        same12 += int(x1[0] == x2[0] and x1[2] == x2[2])
        same23 += int(x1[2] == x2[2] and x1[4] == x2[4])
        same34 += int(x1[4] == x2[4] and x1[6] == x2[6])
        same41 += int(x1[6] == x2[6] and x1[0] == x2[0])

        same123 += int(x1[0] == x2[0] and x1[2] == x2[2] and x1[4] == x2[4])
        same234 += int(x1[2] == x2[2] and x1[4] == x2[4] and x1[6] == x2[6])
        same134 += int(x1[4] == x2[4] and x1[6] == x2[6] and x1[0] == x2[0])
        same124 += int(x1[0] == x2[0] and x1[2] == x2[2] and x1[6] == x2[6])

        same_trio += ((x1[0] == x2[0] and x1[2] == x2[2] and x1[4] == x2[4]) or
                      (x1[2] == x2[2] and x1[4] == x2[4] and x1[6] == x2[6]) or
                      (x1[4] == x2[4] and x1[6] == x2[6] and x1[0] == x2[0]) or
                      (x1[0] == x2[0] and x1[2] == x2[2] and x1[6] == x2[6]))

        same1234 += int(x1[0] == x2[0] and x1[2] == x2[2] and x1[4] == x2[4] and x1[6] == x2[6])
        # break
    print(same1 / n, same2 / n, same3 / n, same4 / n)
    print(same12 / n, same23 / n, same34 / n, same41 / n)
    print(same123 / n, same234 / n, same134 / n, same124 / n)
    print(same1234 / n)
    print(same_trio / n)
    a = 1 - (1 - (same123 / n))**4
    # print(a - (same1234 / n))
    print((same123 / n) * (4 * 4./5) + (same1234 / n))


def retrieval_prob_partitioning():
    n = 100
    k = 6
    r = k-1
    c = 3
    t = 15

    P_arr = []
    for m in range(k, n):
        p1 = 1
        for i in range(r):
            p1 *= 1. * (m-i)/(n-i)
        p1 **= c
        p2 = (k * (1.*(n-m)/(n-k)))**c + (1. * (m-k)/(n-k))**c
        p_match = p1 * p2
        # print(p1, p2, p_match)
        P = 1 - (1 - p_match)**t
        P_arr.append(P)

    x = np.arange(float(k)/n, 1, 1./n)
    plt.plot(x, P_arr)
    plt.plot(x, 1 - (1 - x**20)**100)
    plt.show()


def retrieval_prob_partitioning2():
    n = 40
    k = 10
    r = 12
    c = 1
    t = 20

    P_arr = []
    for m in range(k, n):
        p1 = 1
        for i in range(r):
            p1 *= 1. * (m-i)/(n-i)
        # p1 *= 1. * m/n
        p_match = p1 ** c
        P = 1 - (1 - p_match)**t
        P_arr.append(P)

    x = np.arange(float(k)/n, 1, 1./n)
    plt.plot(x, P_arr, label="partitioning2")
    plt.plot(x, 1 - (1 - x**10)**20, label="MinHash")
    plt.legend()
    plt.show()


def retrieval_prob_one():
    n = 100
    K = 5
    L = 20

    P_arr = []
    P_match_arr = []
    P_match_arr2 = []
    for m in range(K, n):
        p_match = 1
        p_match2 = 1
        for k in range(K):
            p_match *= float(m-k) / (n-k)
            p_match2 *= float(m+1-k) / (n+1-k)
        P_match_arr.append(p_match)
        P_match_arr2.append(p_match2)
        P = 1 - (1 - p_match)**L
        P_arr.append(P)

    x = np.arange(float(K)/n, 1, 1./n)
    plt.plot(x, P_arr, label="one permutation")
    # plt.plot(x, 1 - (1 - np.array(P_match_arr2))**L, label="h")
    plt.plot(x, 1 - (1 - x**10)**50, label="vanilla")
    # plt.plot(x, 1 - (1 - x ** K) ** (L+50), label="+50")
    # plt.plot(x, 1 - (1 - x ** K) ** (L-50), label="-50")

    # plt.plot(x, 1 - x**K, label="minhash")
    # plt.plot(x, 1 - np.array(P_match_arr), label="one permu")
    plt.legend()
    plt.show()


def adaPartHash_get_ratio(n, J, L, c):
    return 1 - (1 - (1 - c)**(1./L))**(1./math.ceil((1-J)*n))


def autoKMin_collision_prob(n, m, r):
    d = int(n * r)
    prob = (1. * factorial(m) / factorial(m-d)) * (1. * factorial(n-d) / factorial(n))
    return prob


def APH_prob_curve():
    n = 1000
    L = 50
    J = 0.9
    c = 0.99

    p = adaPartHash_get_ratio(n, J, L, c)

    K = round(calculate_optimal_k(1 - c, J, L))
    L2 = L
    P_arr, P_arr2 = [], []
    x = range(1, n)
    for m in x:
        p_match = (1 - p)**(n-m)
        # p_match2 = autoKMin_collision_prob(n, m, p - 0.01) if n*p < m else 0
        P_ret = 1 - (1 - p_match)**L
        # P_ret2 = 1 - (1 - p_match2)**L
        P_arr.append(P_ret)
        # P_arr2.append(P_ret2)
        J = float(m)/n
        # print("J = {:.2f}; P(MH) = {:.4f}; P(APH) = {:.4f}; P(?) = {:.4f}".format(J, 1-(1-J**K)**L2, P_ret, P_ret2))

    x = np.array(x) / float(n)
    plt.plot(x, P_arr, label="partitioning Hash (p={})".format(p))
    # plt.plot(x, P_arr2, label="partitioning Hash2 (p={})".format(p))
    plt.plot(x, 1 - (1 - x**K)**L2, label="MinHash (K={})".format(K))
    plt.legend()
    plt.show()


def count_ratio_freq():
    J = 0.9
    L = 500
    c = 0.999
    dic = {}

    for n in range(1, 1000):
        p = adaPartHash_get_ratio(n, J, L, c)
        if p not in dic:
            dic[p] = 0
        dic[p] += 1
        print(n, p)
    print("K:", calculate_optimal_k(1-c, J, L))
    print(dic)


def calculate_optimal_k(delta, p, L):
    log1 = math.log(1 - delta**(1./L))
    log2 = math.log(p)
    return log1 / log2


def check_ratio_distribution():
    J = 0.9
    L = 50
    c = 0.999

    J_arr = np.arange(J, 1./J, (1./J-J)/100)

    for x in range(10, 500):
        ratio_set = set()
        for jj in J_arr:
            n_hat = int(x / jj)
            ratio = adaPartHash_get_ratio(n_hat, J, L, c)
            ratio_set.add(round(ratio, 2))
        print(x, len(ratio_set))
    print(ratio_set)


def runtime_trial():
    d = 100
    delta = 0.01
    J = 0.9
    L = 50

    k = calculate_optimal_k(delta, J, L)
    r = math.exp(-d / float(k)) / (1 - math.exp(-d / float(k)))
    p = adaPartHash_get_ratio(d, J, L, 1-delta)

    print("K={}, ratio={}".format(k, p))
    print("MH:  dK = {}".format(d*k))
    print("DMH: d+(r+2)k = {}".format(d+(r+2)*k))
    print("APH: d+pdlog(pd) = {}".format(d+p*d*log(p*d, 2)))
    print("APH: d+pd = {}".format(d + p * d))


def runtime_plot():
    n = 1000
    delta = 0.01
    J = 0.9
    L = 50

    k = calculate_optimal_k(delta, J, L)
    print("K={}".format(k))
    DMH_runtime = []
    APH_runtime = []
    KAPH_runtime = []
    for d in range(5, n):
        r = math.exp(-d / float(k)) / (1 - math.exp(-d / float(k)))
        p = adaPartHash_get_ratio(d, J, L, 1 - delta)
        DMH_runtime.append((r+2)*k)
        APH_runtime.append(p*d*log(p*d, 2))
        KAPH_runtime.append(p*d)

    plt.plot(list(range(5, n)), DMH_runtime, label="DMH")
    plt.plot(list(range(5, n)), APH_runtime, label="APH")
    plt.plot(list(range(5, n)), KAPH_runtime, label="KAPH")
    plt.legend()
    plt.show()


def rou_analysis():
    R = 0.9
    up, down = 1, 0.2
    c = np.arange(up, down, -0.001)

    # MinHash
    MH_rou = np.divide(np.log(R), np.log(c*R))

    # APH
    APH_rou = np.divide(1 - R, 1 - (c*R))

    plt.plot(c, MH_rou, label="MH")
    plt.plot(c, APH_rou, label="APH")
    plt.xlim(up+0.05, down-0.05)
    plt.xlabel("c")
    plt.ylabel("ρ")
    plt.title("ρ analysis")
    plt.legend()
    plt.show()


def rou_analysis2():
    D = 0.1
    c = np.arange(1, 1/D, 1)

    # MinHash
    MH_rou = np.divide(np.log(1 - D), np.log(1 - c*D))
    MH_rou = np.divide(1, c)

    # APH
    APH_rou = np.divide(D, c*D)

    plt.plot(c, MH_rou, label="MH")
    plt.plot(c, APH_rou, label="APH")
    plt.xlabel("c")
    plt.ylabel("ρ")
    plt.title("ρ analysis")
    plt.legend()
    plt.show()


def rd_k_relationship():
    J = 0.2
    K = 500

    def r2k(r, J, d):
        s1 = (1.-J) * d / J
        s2 = log(1-r) / log(J)
        return s1 * s2

    def k2r(k, J, d):
        s1 = (1. - J) * d / J
        return 1 - J**(k / s1)

    d_arr = np.array(list(range(10, 10000)))
    rd_arr, rdlog_arr = [], []
    exp = np.exp(-np.divide(d_arr, K))
    r = exp / (1 - exp)
    kr_arr = (r+2) * K
    for d in d_arr:
        rd = k2r(K, J, d)*d
        rd_arr.append(rd)
        rdlog_arr.append(rd * log(rd))

    plt.plot(d_arr, rd_arr, label="rd")
    plt.plot(d_arr, rdlog_arr, label="rdlog")
    plt.plot(d_arr, kr_arr, label="kr")
    plt.legend()
    plt.show()


def rou_analysis_full():
    R = 0.9
    lower = 0.7
    # c = np.divide(np.arange(R, lower, -0.001), R)
    up, down = 1, 0.7
    c = np.arange(up, down, -0.001)

    delta = 0.001
    L = 100
    J = R

    # MinHash
    k = calculate_optimal_k(delta, J, L)
    MH_p1 = 1 - np.power(1 - np.power(R, k), L)
    print(MH_p1)
    MH_p2 = 1 - np.power(1 - np.power(c*R, k), L)
    MH_rou = np.divide(np.log(MH_p1), np.log(MH_p2))
    # print(MH_rou)

    # APH
    APH_c1 = (1 - delta**(1./L))**((1.-R)/(1.-J))
    APH_p1 = 1 - np.power(1 - APH_c1, L)
    print(APH_p1)
    APH_c2 = np.power(1 - delta**(1./L), np.divide(1. - (c*R), 1. - J))
    APH_p2 = 1 - np.power(1 - APH_c2, L)
    APH_rou = np.divide(np.log(APH_p1), np.log(APH_p2))

    plt.plot(c, MH_rou, label="MH")
    plt.plot(c, APH_rou, label="APH")
    plt.xlim(up+0.05, down-0.05)
    plt.xlabel("c")
    plt.ylabel("ρ")
    plt.title("ρ analysis (with L=100 tables)")
    plt.legend()
    plt.show()


def rou_comparison_hamming():
    d = 1
    c = np.arange(1, 100, 1)
    D = 10000

    H = 10

    print()
    # Hamming
    print(np.log(1 - d/D))
    print(np.log(1 - np.divide(c*d, D)))
    Hamming_rou = np.divide(np.log(1 - d/D), np.log(1 - np.divide(c*d, D)))
    # APH
    APH_rou = np.divide(1, c)

    plt.plot(c, Hamming_rou, label="Hamming")
    plt.plot(c, APH_rou, label="APH")
    plt.legend()
    plt.show()


if __name__ == '__main__':
    # APH_prob_curve()
    # count_ratio_freq()
    # print(calculate_optimal_k(0.1, 0.9, 100))
    # check_ratio_distribution()
    # runtime_trial()
    # runtime_plot()

    # rou_analysis()
    # rou_analysis2()
    # rou_analysis_full()

    # rd_k_relationship()
    rou_comparison_hamming()
