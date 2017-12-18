
from sklearn.naive_bayes import MultinomialNB
import pandas as pd
import numpy as np
import datetime
import os
from experiment.expt_util import readTopicData,readNonTopicText,csv_to_train_test,classificationTest,saveResult

nonTopicData = readNonTopicText(addr = '../data/non_topic/10000_nontopicTrain.txt')
print 'nontopic indomain test'
resDict_non2topic = {}
count = 10
res=np.zeros(7)
for i in xrange(0,count):
    train_set, train_label, test_set, test_label = csv_to_train_test(nonTopicData,nonTopicData,ratio=4,times=10)
    res+=  np.array(classificationTest(train_set, train_label, test_set, test_label,classifier=MultinomialNB()))
res = res/count
print res
datestr = datetime.datetime.now().strftime('%y_%m_%d_%H_%M_%S')

allres = { 'non2non':
               {'avgRes_{}'.format(str(count)) : res}
      }
saveResult(allres, save_addr= '../data/result/' +datestr + '_non2nonRes.txt')