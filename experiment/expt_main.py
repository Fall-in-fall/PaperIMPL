# -*- encoding:utf-8 -*-
from final.TASC import TASC
from final.feature_handle import Source
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
import pandas as pd
import datetime
import os
import numpy as np
#from sentiment_classify_method import ngram_sa_method
from experiment.expt_util import readTopicData,readTopicData_final,readNonTopicText,csv_to_train_test,classificationTest,saveResult

# 这里放在console执行时，要显示设置路径。console设置的默认当前路径是项目所在路径而不是py文件路径
def run(instance_addr = './data/out_domain/all_review_no3.txt.gz',
                 vecModel_addr = './data/word_vector_data/word2vec_glove.twitter.27B.100d.txt',
                 para_w = [1,1,1],save_addr = './data/result/TASC_2_topic.txt',
                 topic_addr = './data/topic/final'):


    topicData = readTopicData_final()
    # ------------ non to topic
    nonTopicData = readNonTopicText(addr = './data/non_topic/nontopicTrain.txt')
    resDict_non2topic = {}
    for k, v in topicData.iteritems():
        print 'test in topic "{}"'.format(k)
        test_set, test_label = v['text'], v['label']
        train_set, train_label = nonTopicData['text'], nonTopicData['label']
        res = classificationTest(train_set, train_label, test_set, test_label,lowFreqK=10,classifier = MultinomialNB())
        resDict_non2topic[k] = res
    saveResult(resDict_non2topic, save_addr=topic_addr + '/result/' + 'non2topic.txt')
    # ------------ transfer
    # 这里放在console执行时，要显示设置路径。console设置的默认当前路径是项目所在路径而不是py文件路径

    tasc = TASC(instance_addr=instance_addr, vecModel_addr=vecModel_addr)
    resDict_tasc_mix = {}
    size = 1200
    for k, v in topicData.iteritems():
        print 'test in topic "{}"'.format(k)
        selected_num = len(v) * 5 if len(v)>size else size*5
        shortlist_num = selected_num * 2
        selected_instances = tasc.get_instance_TASC(k, v, selected_num, shortlist_num)
        print 'len(selected_instances): ',len(selected_instances)

        test_set, test_label = v['text'],v['label']
        train_set, train_label =  pd.concat([nonTopicData['text'],selected_instances['text']]), pd.concat([nonTopicData['label'],selected_instances['label']])
        res = classificationTest(train_set, train_label, test_set, test_label,lowFreqK=10,classifier = MultinomialNB())#
        resDict_tasc_mix[k] = res
    saveResult(resDict_tasc_mix, save_addr=topic_addr + '/result/all/' + str(size) + '_TASC_topic_transfer.txt')
    #-----------
    resDict_tasc_trans= {}
    for k, v in topicData.iteritems():
        print 'test in topic "{}"'.format(k)
        selected_num = len(v) * 5 if len(v) > size else size * 5
        shortlist_num = selected_num * 2
        selected_instances = tasc.get_instance_TASC(k, v, selected_num, shortlist_num)
        print 'len(selected_instances): ',len(selected_instances)
        test_set, test_label = v['text'], v['label']
        train_set, train_label = selected_instances['text'], selected_instances['label']
        res = classificationTest(train_set, train_label, test_set, test_label,lowFreqK=10,classifier =MultinomialNB())
        resDict_tasc_trans[k] = res

    saveResult(resDict_tasc_trans, save_addr=topic_addr + '/result/all/' + str(size) + '_TASC_topic_mix.txt')
    print resDict_non2topic
    print resDict_tasc_trans
    print resDict_tasc_mix
def run_indomain():
    nonTopicData = readNonTopicText(addr = '../data/non_topic/7500_nontopicTrain.txt')
    print 'nontopic indomain test'
    resDict_non2topic = {}
    count = 10
    res=np.zeros(7)
    for i in xrange(0,count):
        train_set, train_label, test_set, test_label = csv_to_train_test(nonTopicData,nonTopicData,ratio=10,times=10)
        res+=  np.array(classificationTest(train_set, train_label, test_set, test_label,classifier=MultinomialNB()))
    print res/count
    datestr = datetime.datetime.now().strftime('%y_%m_%d_%H_%M_%S')

    allres = { 'non2non':
                   {'_avgRes_{}'.format(str(count)) : res}
          }
    saveResult(allres, save_addr= '../data/result/' +datestr + '_non2nonres.txt')

def run_expt_final(topicDataFile  = '../data/topic/all_test_topicData/all_topic.csv',
             non_addr = '../data/non_topic/nontopicTrain.txt',
             result_addr = '../data/result/all',
             size = 1500):
    #topicData = readTopicData('../data/topic/all_test_topicData/single')
    topicData = readTopicData_final(topicDataFile  = topicDataFile )
    # ------------ non to topic
    nonTopicData = readNonTopicText(addr=non_addr)
    print 'nontopic test'
    resDict_non2topic = {}
    for k, v in topicData.iteritems():
        print 'test in topic "{}"'.format(k)
        test_set, test_label = v['text'], v['label']
        train_set, train_label = nonTopicData['text'], nonTopicData['label']
        res = classificationTest(train_set, train_label, test_set, test_label,classifier=MultinomialNB())
        resDict_non2topic[k] = res

    # 这里放在console执行时，要显示设置路径。console设置的默认当前路径是项目所在路径而不是py文件路径

    tasc = TASC(instance_addr='../data/out_domain/10000_review_no3.txt.gz',
                 vecModel_addr='../data/word_vector_data/word2vec_glove.twitter.27B.100d.txt')

    # tasc.get_instance_TASC('apple',topicData['pure_dealed2016all'],10000,15000)
    print 'mix test'
    resDict_mix = {}

    for k, v in topicData.iteritems():
        print 'test in topic "{}"'.format(k)
        selected_num = len(v) * 5 if len(v)>size else size*5
        shortlist_num = selected_num * 2
        selected_instances = tasc.get_instance_TASC(k, v, selected_num, shortlist_num)
        print 'len(selected_instances): ',len(selected_instances)
        test_set, test_label = v['text'],v['label']
        train_set, train_label =  pd.concat([nonTopicData['text'],selected_instances['text']]), pd.concat([nonTopicData['label'],selected_instances['label']])
        res = classificationTest(train_set, train_label, test_set, test_label,lowFreqK=10,classifier=MultinomialNB())#
        resDict_mix[k] = res
    print 'transfer test'
    resDict_transfer= {}
    for k, v in topicData.iteritems():
        print 'test in topic "{}"'.format(k)
        selected_num = len(v) * 5 if len(v) > size else size * 5
        shortlist_num = selected_num * 2
        selected_instances = tasc.get_instance_TASC(k, v, selected_num, shortlist_num)
        print 'len(selected_instances): ',len(selected_instances)
        test_set, test_label = v['text'], v['label']
        train_set, train_label = selected_instances['text'], selected_instances['label']
        res = classificationTest(train_set, train_label, test_set, test_label,lowFreqK=10,classifier = MultinomialNB())
        resDict_transfer[k] = res

    print 'resDict_non2topic'
    print resDict_non2topic
    print 'resDict_transfer'
    print resDict_transfer
    print 'resDict_mix'
    print resDict_mix

    datestr = datetime.datetime.now().strftime('%y_%m_%d_%H_%M_%S')
    topicAccuRes = { '*topic':resDict_non2topic.keys(),
                     'resDict_non2topic':[ v[0] for v in resDict_non2topic.values()],
                     'resDict_mix':[ v[0] for v in resDict_mix.values()],
                     'resDict_transfer':[ v[0] for v in resDict_transfer.values()]
                }

    def getAVGRes(x):
        return [  sum( [v[i] for v in x.values()] )/len(topicData) for i in range(0,7)]

    avgRes = { 'resDict_non2topic':getAVGRes(resDict_non2topic),
               'resDict_mix':getAVGRes(resDict_mix),
               'resDict_transfer':getAVGRes(resDict_transfer)
               }
    allres = { 'resDict_non2topic':resDict_non2topic,'resDict_mix':resDict_mix, 'resDict_transfer':resDict_transfer,
               'topicAccuRes':topicAccuRes,
               'avgRes':avgRes
               }
    print topicAccuRes.keys()
    for line in  [ [x[i] for x in topicAccuRes.values()] for i in range(0,len(topicAccuRes.values()[0])) ]:
        print line
    print avgRes
    if not os.path.isdir(result_addr):
        os.mkdir(result_addr)
    saveResult(allres, save_addr=result_addr  +datestr + '_allres.txt')

if __name__ =='__main__':
    run_expt_final()
    # run_expt(topic_addr = '../data/topic/final/revise5000',
    #          non_addr = '../data/non_topic/nontopicTrain.txt',
    #          size = 1500)
    # run_expt(topic_addr = '../data/topic/final/allfinal',
    #          non_addr = '../data/non_topic/15000_nontopicTrain.txt',
    #          size = 1500)