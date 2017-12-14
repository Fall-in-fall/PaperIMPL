# -*- encoding:utf-8 -*-
from final.TASC import TASC
from final.feature_handle import Source
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
import pandas as pd
import datetime
#from sentiment_classify_method import ngram_sa_method
from experiment.expt_util import readTopicData,readNonTopicText,csv_to_train_test,classificationTest,saveResult

# 这里放在console执行时，要显示设置路径。console设置的默认当前路径是项目所在路径而不是py文件路径

topic_addr = '../data/topic/final'
topicData = readTopicData(topic_addr)
# ------------ non to topic
nonTopicData = readNonTopicText(addr = '../data/non_topic/7500_nontopicTrain.txt')
resDict_non2topic = {}
for k, v in topicData.iteritems():
    print 'test in topic "{}"'.format(k)
    test_set, test_label = v['text'], v['label']
    train_set, train_label = nonTopicData['text'], nonTopicData['label']
    res = classificationTest(train_set, train_label, test_set, test_label,classifier=MultinomialNB())
    resDict_non2topic[k] = res

# ------------ transfer
# 这里放在console执行时，要显示设置路径。console设置的默认当前路径是项目所在路径而不是py文件路径

tasc = TASC(instance_addr='../data/out_domain/10000_review_no3.txt.gz',
             vecModel_addr='../data/word_vector_data/word2vec_glove.twitter.27B.100d.txt')

# tasc.get_instance_TASC('apple',topicData['pure_dealed2016all'],10000,15000)
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

resDict_tasc1 = {}
size = 1500
for k, v in topicData.iteritems():
    print 'test in topic "{}"'.format(k)
    selected_num = len(v) * 5 if len(v)>size else size*5
    shortlist_num = selected_num * 2
    selected_instances = tasc.get_instance_TASC(k, v, selected_num, shortlist_num)
    print 'len(selected_instances): ',len(selected_instances)
    test_set, test_label = v['text'],v['label']
    train_set, train_label =  pd.concat([nonTopicData['text'],selected_instances['text']]), pd.concat([nonTopicData['label'],selected_instances['label']])
    res = classificationTest(train_set, train_label, test_set, test_label,lowFreqK=10,classifier=MultinomialNB())#
    resDict_tasc1[k] = res

resDict_tasc2= {}
for k, v in topicData.iteritems():
    print 'test in topic "{}"'.format(k)
    selected_num = len(v) * 5 if len(v) > size else size * 5
    shortlist_num = selected_num * 2
    selected_instances = tasc.get_instance_TASC(k, v, selected_num, shortlist_num)
    print 'len(selected_instances): ',len(selected_instances)
    test_set, test_label = v['text'], v['label']
    train_set, train_label = selected_instances['text'], selected_instances['label']
    res = classificationTest(train_set, train_label, test_set, test_label,lowFreqK=10,classifier = MultinomialNB())
    resDict_tasc2[k] = res


print 'resDict_non2topic'
print resDict_non2topic
print 'resDict_tasc2'
print resDict_tasc2
print 'resDict_tasc1'
print resDict_tasc1


datestr = datetime.datetime.now().strftime('%y_%m_%d_%M_%S')
topicAccuRes = { '*topic':resDict_non2topic.keys(),
                 'resDict_non2topic':[ v[0] for v in resDict_non2topic.values()],
                 'resDict_tasc1':[ v[0] for v in resDict_tasc1.values()],
                 'resDict_tasc2':[ v[0] for v in resDict_tasc2.values()]
            }

def getAVGRes(x):
    return [  sum( [v[i] for v in x.values()] )/len(topicData) for i in range(0,7)]

avgRes = { 'resDict_non2topic':getAVGRes(resDict_non2topic),
           'resDict_tasc1':getAVGRes(resDict_tasc1),
           'resDict_tasc2':getAVGRes(resDict_tasc2)
           }
allres = { 'resDict_non2topic':resDict_non2topic,'resDict_tasc1':resDict_tasc1, 'resDict_tasc2':resDict_tasc2,
           'topicAccuRes':topicAccuRes,
           'avgRes':avgRes
          }
print topicAccuRes.keys()
print [ [x[i] for x in topicAccuRes.values()] for i in range(0,len(topicAccuRes.values()[0])) ]
print avgRes
saveResult(allres, save_addr=topic_addr + '/result/' +datestr + '_allres.txt')

#print resDict_tasc2
#saveResult(resDict_non2topic,saveAddr = topic_addr+'/result/'+'non2topic.txt')
#saveResult(resDict_tasc1,saveAddr = topic_addr+'/result/'+str(size)+'_TASC_topic_mix.txt')
#saveResult(resDict_tasc2,saveAddr = topic_addr+'/result/'+str(size)+'_TASC_2_topic2.txt')
#----------------------------
# k='twitter'
# v=topicData[k]
# resDict_tasc = {}
# print 'test in topic "{}"'.format(k)
# selected_num = len(v) * 4
# shortlist_num = selected_num * 4
# selected_instances = tasc.get_instance_TASC(k, v, selected_num, shortlist_num)
# print 'len(selected_instances): ', len(selected_instances)
# test_set, test_label = v['text'], v['label']
# train_set, train_label = selected_instances['text'], selected_instances['label']
# res = classificationTest(train_set, train_label, test_set, test_label, classifier = RandomForestClassifier(n_estimators=100))
# resDict_tasc[k] = res


 #a = pd.read_csv('./all_meta_no3.txt.gz',delimiter='\t',header=None,names = ['asin','title','cate'])

# {'twitter': [0.64835164835164838, 0.80000000000000004, 0.6923076923076924, array([[36,  9],
#        [23, 23]])], 'google': [0.82407407407407407, 0.89714285714285713, 0.89204545454545447, array([[157,  18],
#        [ 20,  21]])], 'apple': [0.63184079601990051, 0.66400000000000003, 0.52866242038216549, array([[ 83,  42],
#        [106, 171]])], 'microsoft': [0.58375634517766495, 0.75903614457831325, 0.60576923076923073, array([[63, 20],
#        [62, 52]])]}

