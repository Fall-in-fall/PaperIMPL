# -*- encoding:utf-8 -*-

import pandas as pd
import re
def twitterData():
    sent140 = pd.read_csv('../data/dealed_sent140-noNeu.txt',
                          names=['topic', 'label', 'id', 'text'], delimiter="\t", quoting=3)

    #注意sts的顺序和sent140不一样,要调整过来
    stsent = pd.read_csv('../data/sts_ent_topic.txt',
                         names=['topic', 'id', 'label', 'text'], delimiter="\t", quoting=3)
    stsent['label'] = stsent['label'].replace([0, 4], ['negative', 'positive'])
    stsent = stsent[['topic','label','id','text']]

    # se2016 = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data', 'dealed2016all.txt'),
    #                      names=['topic', 'label', 'id', 'text'], delimiter="\t", quoting=3)
    # --------------
    sent140_topic = ['kindle2','nike','dentist']
    sts_topic = ['iphone','ipod','england','london','facebook']
    for i in sent140_topic:
        res = sent140[sent140['topic'] == i]
        if i =='kindle2':
            res['topic']='kindle'
            i='kindle'
        to_thecsv(res ,'../data/topic/shortlist/shortlist_twitter/'+ i +'.txt_')
    for j in sts_topic:
        res = stsent[stsent['topic'] == j]
        to_thecsv( res , '../data/topic/shortlist/shortlist_twitter/' + j + '.txt_')
    #se2016_topic = ['Atheism']

def to_thecsv(df,fname):
    df.drop(['id'], axis=1, inplace=True)
    df['label'] = df['label'].replace(['negative', 'positive'], [0, 1])
    df.to_csv(fname, index=False, header=False, sep='\t')


def save_pure(fileAddr,newAddr):
    print 'processin '+fileAddr
    x = pd.read_csv(fileAddr,names=['topic','label','id','text'], delimiter="\t", quoting=3)
    x.drop([ 'id'], axis=1, inplace=True)
    x['label'] = x['label'].replace( ['negative', 'positive'],[0, 1])
    x.to_csv(newAddr, index=False,header = False, sep='\t')

if __name__ =='__main__':
    twitterData()
    # baseaddr = '../data/topic/'
    # nameList = ['atheism.txt']
    # for i in nameList:
    #     save_pure(baseaddr+i,baseaddr+i+'_')