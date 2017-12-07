# -*- encoding:utf-8 -*-
import os
dir = '../data/topic/ready'
fileList = os.listdir(dir)
for fname in fileList:
    print fname
    theFile = open(dir+'/'+fname,'r')
    readAll = theFile.read()
    readAll = readAll.replace('   ','\t')
    theFile.close()
    toWrite = open(dir+'/'+'_'+fname,'w')
    toWrite.write(readAll)
    toWrite.close()

