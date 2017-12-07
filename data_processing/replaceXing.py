# -*- encoding:utf-8 -*-
import os
import gzip
dir = '../data/out_domain'
fileList = os.listdir(dir)
for fname in fileList:
    if fname.__contains__('all_review') and not fname.__contains__('Dict'):
        print fname
        theFile = gzip.open(dir+'/'+fname,'r')
        readAll = theFile.read()
        readAll = readAll.replace('*',' ')
        theFile.close()
        toWrite = gzip.open(dir+'/'+fname,'w')
        toWrite.write(readAll)
        toWrite.close()
