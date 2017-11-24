# -*- encoding:utf-8 -*-
__author__ = 'tyz'
from datetime import datetime
from final.feature_handle import Source
from experiment import expt_main

if __name__ =='__main__':
    start = datetime.now()

    expt_main.run(instance_addr = './data/out_domain/all_review_no3.txt.gz',
                  vecModel_addr='./data/word_vector_data/word2vec_glove.twitter.27B.100d.txt',
                  save_addr='./data/result/TASC_2_topic.txt',
                  topic_addr='./data/topic/final')

    end = datetime.now()
    print (end-start).seconds,'s'