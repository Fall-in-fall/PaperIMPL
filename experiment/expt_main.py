# -*- encoding:utf-8 -*-
from final.TASC import TASC
from final.feature_handle import Source
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
import pandas as pd
#from sentiment_classify_method import ngram_sa_method
from experiment.expt_util import readTopicData,readNonTopicText,csv_to_train_test,classificationTest,saveResult

# 这里放在console执行时，要显示设置路径。console设置的默认当前路径是项目所在路径而不是py文件路径
def run(instance_addr = './data/out_domain/all_review_no3.txt.gz',
                 vecModel_addr = './data/word_vector_data/word2vec_glove.twitter.27B.100d.txt',
                 para_w = [1,1,1],save_addr = './data/result/TASC_2_topic.txt',
                 topic_addr = './data/topic/final'):
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import MultinomialNB

    topicData = readTopicData(topic_addr)
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
        res = classificationTest(train_set, train_label, test_set, test_label,lowFreqK=10,classifier=MultinomialNB())#
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


