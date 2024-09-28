def rtd(ranks1, ranks2, alpha=1):
    n = len(ranks1)
    rtd = 0
    for i in range(n):
        rtd += abs(1 / (ranks1[i] + 1)**alpha - 1 / (ranks2[i] + 1)**alpha)**(1/(alpha + 1))
    return ((alpha + 1) / alpha) * rtd
