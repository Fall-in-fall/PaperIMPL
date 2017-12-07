# -*- encoding:utf-8 -*-
import cPickle,pickle
from gensim import corpora
from gensim.models import KeyedVectors
import numpy as np
import collections
from compiler.ast import flatten
import re
import gzip
class Source:
    asin =''
    topic_vec = []
    term_dict = {}
    instance_avglen = 0
    pos_instance_num = 0
    neg_instance_num = 0
    titleList = []
    cateList = []
    # def __init__(self,topic_vec=[],term_dict={},instance_avglen=0):
    #     self.topic_vec = topic_vec
    #     self.term_dict = term_dict
    #     self.instance_avglen = instance_avglen


def gen_feature(domain_addr = '../data/out_domain/100000_meta_no3.txt.gz',
                instance_addr = '../data/out_domain/100000_review_no3.txt.gz',
                vecModel_addr = '../data/word_vector_data/word2vec_glove.twitter.27B.100d.txt',
                lowFreqK = 10 ):
    print 'generate feature for '+domain_addr
    ## 读取所有实例，构建字典 allTermDict 并保存
    print ' 读取所有实例，构建字典 allTermDict 并保存 '
    allTermDict = collections.defaultdict(list)
    for line in gzip.open(instance_addr, 'r'):
        line_split = line.strip().split('\t')
        asin = line_split[0]
        allTermDict[asin].append(line_split[1:])
    cPickle.dump(allTermDict, file(instance_addr + '_allTermDict', 'w'))  # 保存 allTermDict

    ## 构建gensim词字典并保存，去停用词，去低频词
    print '构建gensim词字典'
    allTextList = flatten( [ [ i[1] for i in asinAllterm ]
                             for asinAllterm in allTermDict.values()]  )
    stoplist = set('for a of the and to in'.split(' '))
    allTexts = [ [word for word in
                  text.lower().replace(',','').replace('.','').split(' ')
                  if word not in stoplist]
                for text in allTextList]
    # allTexts = [ text.lower().split() for text in allTextList]
    del allTextList
    frequency = collections.defaultdict(int)
    for text in allTexts:
        for token in text:
            frequency[token] += 1
    allTexts = [[token for token in text if frequency[token] > lowFreqK]
             for text in allTexts]
    gensimDict = corpora.Dictionary(allTexts)
    del allTexts
    # gensimDict.save(file(instance_addr + '_gensimDict', 'w'))
    cPickle.dump(gensimDict, file(instance_addr + '_gensimDict', 'w') )

    ## 为每个source生成词分布 和 平均实例长度，放入allSourceDict
    #invalidList = set(['',''])
    print '为每个source生成词分布 和 平均实例长度，放入allSourceDict'
    allSourceDict = collections.defaultdict(Source)
    for k,v in allTermDict.iteritems():
        allSourceDict[k].term_dict = gensimDict.doc2bow(
            flatten( [ i[1].lower().replace(',','').replace('.','').split(' ')   for i in v ] )  )
        allSourceDict[k].instance_avglen = sum( [len(j[1]) for j in v] )/len(v)
        pos_shortlist_num = 0
        neg_shortlist_num = 0

        # 写成dictCount的形式实现，待实现
        for i in v:
            if i[0] == '1' :pos_shortlist_num += 1
            if i[0] == '0': neg_shortlist_num += 1
        allSourceDict[k].pos_instance_num = pos_shortlist_num
        allSourceDict[k].neg_instance_num = neg_shortlist_num

    del allTermDict
    del gensimDict

    ## 读取domain并生成 topicvec,然后保存allSourceDict
    print '读取domain并生成 topicvec,然后保存allSourceDict'
    model = KeyedVectors.load_word2vec_format(vecModel_addr,binary=False)
    vector_size= model.vector_size
    vec_dimention = 100
    for dline in gzip.open(domain_addr,'r'):
        dline_split = dline.strip().split('\t')
        source = Source
        asin = dline_split[0]
        titleList = re.sub('[^a-zA-Z]', ' ', dline_split[1].strip() ).split(' ')
        cateList = re.sub('[^a-zA-Z]', ' ', dline_split[2].strip() ).split(' ')
        avgvec_title = np.zeros(vector_size)
        title_validCount = 0
        title_invalid_count = 0
        for i in titleList: #实际上 title 可以看作是一个短句。直接相加也是建模句子的简单实现。
            if i in model.vocab:
                avgvec_title += model.wv[i]
                title_validCount += 1
            elif i.lower() in model.vocab:
                avgvec_title += model.wv[i.lower()]
                title_validCount += 1
            elif len(i)>1 and i[0].upper()+i[1:].lower() in model.vocab:
                avgvec_title += model.wv[i[0].upper()+i[1:].lower()]
                title_validCount += 1
            else:
                title_invalid_count+=1
        avgvec_cate = np.zeros(vector_size) # 任何向量与零向量的余弦相似度都是0
        cate_validCount = 0
        cate_invalid_count = 0
        for i in cateList:
            if i in model.vocab:
                avgvec_cate += model.wv[i]
                cate_validCount+=1
            elif i.lower() in model.vocab:
                avgvec_cate += model.wv[i.lower()]
                cate_validCount += 1
            elif len(i) > 1 and i[0].upper() + i[1:].lower() in model.vocab:
                avgvec_cate += model.wv[i[0].upper()+i[1:].lower()]
                cate_validCount += 1
            else:
                cate_invalid_count+=1
        allSourceDict[asin].titleList = titleList
        allSourceDict[asin].cateList = cateList
        avgvec_title = avgvec_title / (title_validCount if title_validCount > 0 else 1)
        avgvec_cate = avgvec_title / (cate_validCount if cate_validCount > 0 else 1)

        allSourceDict[asin].topic_vec = (avgvec_title+avgvec_cate)/ \
                                        ( 2 if title_invalid_count==0 and cate_invalid_count==0
                                          else 1)
    del model
    cPickle.dump(allSourceDict, file(instance_addr + '_allSourceDict', 'w'))  # 保存 allSourceDict

    print 'gen_feature finished'

    # model.wv['apple']: array([...,...])

def add_feature(domain_addr = '../data/out_domain/100000_meta_no3.txt.gz',
                instance_addr = '../data/out_domain/100000_review_no3.txt.gz',
                vecModel_addr = '../data/word_vector_data/word2vec_glove.twitter.27B.100d.txt',
                lowFreqK = 10):
    pass

def load_feature(instance_addr):
    allTermDict = cPickle.load(file(instance_addr+'_allTermDict','r')) # { asin: [ [label,text],...  ]  }
    gensimDict = cPickle.load(file(instance_addr+'_gensimDict','r'))# corpora.Dictionary.load(instance_addr+'_gensimDict')
    allSourceDict =  cPickle.load(file(instance_addr+'_allSourceDict','r')) # { asin: Source  }
    return allTermDict,gensimDict,allSourceDict

def loadAndToAllSource(instance_addr):
    print 'load allTermDict and gensimDict'
    allTermDict = cPickle.load(file(instance_addr+'_allTermDict','r')) # { asin: [ [label,text],...  ]  }
    gensimDict = cPickle.load(file(instance_addr+'_gensimDict','r'))# corpora.Dictionary.load(instance_addr+'_gensimDict')
    print '为每个source生成词分布 和 平均实例长度，放入allSourceDict'
    allSourceDict = collections.defaultdict(Source)
    for k,v in allTermDict.iteritems():
        allSourceDict[k].term_dict = gensimDict.doc2bow(
            flatten( [ i[1].split(' ') for i in v] )  )
        allSourceDict[k].instance_avglen = sum( [len(j[1]) for j in v] )/len(v)
        pos_shortlist_num = 0
        neg_shortlist_num = 0

        # 写成dictCount的形式实现，待实现
        for i in v:
            if i[0] == '1' :pos_shortlist_num += 1
            if i[0] == '0': neg_shortlist_num += 1
        allSourceDict[k].pos_instance_num = pos_shortlist_num
        allSourceDict[k].neg_instance_num = neg_shortlist_num

    del allTermDict
    del gensimDict

    ## 读取domain并生成 topicvec,然后保存allSourceDict
    print '读取domain并生成 topicvec,然后保存allSourceDict'
    vecModel_addr = '../data/word_vector_data/word2vec_glove.twitter.27B.100d.txt'
    model = KeyedVectors.load_word2vec_format(vecModel_addr,binary=False)
    vector_size= model.vector_size
    vec_dimention = 100
    for dline in gzip.open(domain_addr,'r'):
        dline_split = dline.strip().split('\t')
        source = Source
        asin = dline_split[0]
        titleList = re.sub('[^a-zA-Z]', ' ', dline_split[1].strip() ).split(' ')
        cateList = re.sub('[^a-zA-Z]', ' ', dline_split[2].strip() ).split(' ')
        avgvec_title = np.zeros(vector_size)
        title_validCount = 0
        title_invalid_count = 0
        for i in titleList: #实际上 title 可以看作是一个短句。直接相加也是建模句子的简单实现。
            if i in model.vocab:
                avgvec_title += model.wv[i]
                title_validCount += 1
            elif i.lower() in model.vocab:
                avgvec_title += model.wv[i.lower()]
                title_validCount += 1
            elif len(i)>1 and i[0].upper()+i[1:].lower() in model.vocab:
                avgvec_title += model.wv[i[0].upper()+i[1:].lower()]
                title_validCount += 1
            else:
                title_invalid_count+=1
        avgvec_cate = np.zeros(vector_size) # 任何向量与零向量的余弦相似度都是0
        cate_validCount = 0
        cate_invalid_count = 0
        for i in cateList:
            if i in model.vocab:
                avgvec_cate += model.wv[i]
                cate_validCount+=1
            elif i.lower() in model.vocab:
                avgvec_cate += model.wv[i.lower()]
                cate_validCount += 1
            elif len(i) > 1 and i[0].upper() + i[1:].lower() in model.vocab:
                avgvec_cate += model.wv[i[0].upper()+i[1:].lower()]
                cate_validCount += 1
            else:
                cate_invalid_count+=1
        allSourceDict[asin].titleList = titleList
        allSourceDict[asin].cateList = cateList
        avgvec_title = avgvec_title / (title_validCount if title_validCount > 0 else 1)
        avgvec_cate = avgvec_title / (cate_validCount if cate_validCount > 0 else 1)

        allSourceDict[asin].topic_vec = (avgvec_title+avgvec_cate)/ \
                                        ( 2 if title_invalid_count==0 and cate_invalid_count==0
                                          else 1)
    del model
    cPickle.dump(allSourceDict, file(instance_addr + '_allSourceDict', 'w'))  # 保存 allSourceDict

    print 'gen_feature finished'

if __name__ =='__main__':

    fileSizeList = ['all','200000']
    for fileSize in fileSizeList:
        domain_addr = '../data/out_domain/{}_meta_no3.txt.gz'.format(fileSize)
        instance_addr = '../data/out_domain/{}_review_no3.txt.gz'.format(fileSize)
        if fileSize=='all':
            loadAndToAllSource(instance_addr)
        else:
            gen_feature(domain_addr, instance_addr)


    # fileSize = '50000'
    # domain_addr = '../data/out_domain/{}_meta_no3.txt.gz'.format(fileSize)
    # instance_addr = '../data/out_domain/{}_review_no3.txt.gz'.format(fileSize)
    # gen_feature(domain_addr,instance_addr)
    #
    # fileSize = '100000'
    # domain_addr = '../data/out_domain/{}_meta_no3.txt.gz'.format(fileSize)
    # instance_addr = '../data/out_domain/{}_review_no3.txt.gz'.format(fileSize)
    # gen_feature(domain_addr,instance_addr)

    # fileSize = 'all'
    # domain_addr = '../data/out_domain/{}_meta_no3.txt.gz'.format(fileSize)
    # instance_addr = '../data/out_domain/{}_review_no3.txt.gz'.format(fileSize)
    #loadAndToAllSource(instance_addr)
    # gen_feature(domain_addr,instance_addr)
    #allTermDict, gensimDict, allSourceDict = load_feature(instance_addr)
    #
    # fileSize = '50000'
    # domain_addr = '../data/out_domain/{}_meta_no3.txt.gz'.format(fileSize)
    # instance_addr = '../data/out_domain/{}_review_no3.txt.gz'.format(fileSize)
    # gen_feature(domain_addr,instance_addr)
    # allTermDict, gensimDict, allSourceDict = load_feature(instance_addr)


