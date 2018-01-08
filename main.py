# -*- encoding:utf-8 -*-
__author__ = 'tyz'
# from datetime import datetime
# from final.feature_handle import Source
# from experiment import expt_main
#
# if __name__ =='__main__':
#     start = datetime.now()
#
#     expt_main.run(instance_addr = './data/out_domain/all_review_no3.txt.gz',
#                   vecModel_addr='./data/word_vector_data/word2vec_glove.twitter.27B.100d.txt',
#                   save_addr='./data/result/TASC_2_topic.txt',
#                   topic_addr='./data/topic/final')
#
#     end = datetime.now()
#     print (end-start).seconds,'s'

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
#from sentiment_classify_method import ngram_sa_method
from experiment.expt_util import readTopicData,readNonTopicText,csv_to_train_test,classificationTest,saveResult



def main_expt(topic_addr = './data/topic/final',
             non_addr = './data/non_topic/nontopicTrain.txt',
             size = 1500,
              instance_addr = './data/out_domain/10000_review_no3.txt.gz',
              vecModel_addr='./data/word_vector_data/word2vec_glove.twitter.27B.100d.txt'):
    topicData = readTopicData(topic_addr)
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

    tasc = TASC(instance_addr=instance_addr,
                 vecModel_addr=vecModel_addr)

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
    if not os.path.isdir(topic_addr + '/result'):
        os.mkdir(topic_addr + '/result')
    saveResult(allres, save_addr=topic_addr + '/result/' +datestr + '_allres.txt')

if __name__ =='__main__':
    main_expt(topic_addr = './data/topic/final/revise5000/england',
             non_addr = './data/non_topic/nontopicTrain.txt',
             size = 1500)
    # main_expt(topic_addr = './data/topic/final/revise5000/ipod',
    #          non_addr = './data/non_topic/nontopicTrain.txt',
    #          size = 1500)
    # main_expt(topic_addr = './data/topic/final/ultimate5000',
    #          non_addr = './data/non_topic/nontopicTrain.txt',
    #          size = 1500)