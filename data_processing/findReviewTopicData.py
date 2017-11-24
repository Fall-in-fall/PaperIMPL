# -*- encoding:utf-8 -*-
import pandas as pd
import random
from collections import defaultdict


# meta_addr = '../data/out_domain/all_meta_no3.txt.gz'
# review_addr = '../data/out_domain/all_review_no3.txt.gz',
# meta = pd.read_csv(meta_addr,names=['asin','title','cate'],delimiter='\t', quoting=3)
# review = pd.read_csv(review_addr,delimiter='\t',header=None,names = ['asin','label','text'])

# 找到meta中所有title包含topicList中的topic的条目
def findMetaExist(topicList,meta):
    resIndexDict = defaultdict(list)
    resDict = {}
    for index,row in meta.iterrows():
        for topic in topicList:
            if str(row['title']).lower().__contains__(topic): # 因为会根据取出来的内容自动转换类型，所以这里必须强转str
                resIndexDict[topic].append( index )
    for k,v in resIndexDict.iteritems():
        resDict[k] = meta.iloc[v,:]
    return resDict

def findTopicReview(topicList,meta,review,base_toaddr = '../data/shortlist/',maxSize = 1000):
    asinResDict = defaultdict(list)
    print topicList
    for index,row in meta.iterrows():
        for topic in topicList:
            if str(row['title']).lower().__contains__(topic): # 因为会根据取出来的内容自动转换类型，所以这里必须强转str
                asinResDict[topic].append( str(row['asin']) )
    for k,v in asinResDict.iteritems():
        print k
        asinRes = set(v)
        allTopicReviewIndex = []
        for index2,row2 in review.iterrows():
            if str(row2['asin']) in asinRes:
                allTopicReviewIndex.append(index2)
        resDf = review.iloc[allTopicReviewIndex,:]
        posRes = resDf[resDf['label']==1]
        negRes = resDf[resDf['label']==0]
        if len(posRes)>len(negRes):
            posRes = posRes.sample(len(negRes))
        elif len(posRes)<len(negRes):
            negRes = negRes.sample(len(posRes))

        if len(posRes)>maxSize/2:
            posRes = posRes.sample(maxSize / 2)
            negRes = negRes.sample(maxSize / 2)

        res = pd.concat([posRes,negRes])
        res.insert(0,'topic',k)
        del res['asin']
        res.to_csv(base_toaddr+k+'.txt',index = False,sep='\t',header=False)

if __name__ =='__main__':
    meta_addr = '../data/out_domain/all_meta_no3.txt.gz'
    review_addr = '../data/out_domain/all_review_no3.txt.gz'
    meta = pd.read_csv(meta_addr,names=['asin','title','cate'],header=None,delimiter='\t', )
    review = pd.read_csv(review_addr,names = ['asin','label','text'],header=None,delimiter='\t',quoting=3)
    findTopicReview(['feminist','abortion','donald trump','hillary clinton','climate change'],meta,review,base_toaddr = '../data/topic/shortlist/',maxSize = 1000)
# ---------
# from gensim.models import KeyedVectors
# vecModel_addr='./data/word_vector_data/word2vec_glove.twitter.27B.100d.txt'
# vecModel = KeyedVectors.load_word2vec_format(vecModel_addr ,binary=False)