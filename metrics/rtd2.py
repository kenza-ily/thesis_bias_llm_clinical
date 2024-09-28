def rtd2(rtd_emp, rtd_emb):
    n = len(rtd_emp)
    ranks_emp = rankdata(rtd_emp)
    ranks_emb = rankdata(rtd_emb)
    return rtd(ranks_emp, ranks_emb)
