def findDiff(goup,godown,x):
    length = len(goup)
    res = []
    for i in xrange(0,length):
        if goup[i]!=godown[i] and godown[i]==x[i]:
            res.append(i+1)
    return res