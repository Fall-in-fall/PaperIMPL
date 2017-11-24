# -*- encoding:utf-8 -*-
import re
import gzip
import random
import collections


# 蓄水池算法采样meta及其评论文本 https://www.iteblog.com/archives/1525.html
def reservoirSampling(sampleSize ):

    metapath='../data/out_domain/all_meta_no3.txt.gz'
    reviewpath = '../data/out_domain/all_review_no3.txt.gz'

    rf_meta = gzip.open('../data/out_domain/{}_meta_no3.txt.gz'.format(str(sampleSize)), 'w')
    rf_review = gzip.open( '../data/out_domain/{}_review_no3.txt.gz'.format(str(sampleSize)), 'w')

    reservoir = ['']*sampleSize

    g = gzip.open(metapath, 'r')
    count = 0
    for line in g:
        if count<sampleSize:
            reservoir[count] = line.strip()+'\n'
        else:
            randNum = random.randint(0,count-1) #randInt是闭区间
            if randNum<sampleSize:
                reservoir[randNum] = line.strip()+'\n'
        count+=1
    rf_meta.writelines(reservoir)
    rf_meta.close()
    reservoir = set([i.split('\t')[0] for i in reservoir])
    #-------------
    g2 = gzip.open(reviewpath, 'r')
    count = 0
    # 如果再次取出掉thescore!=3的，存在可能有些域的实例全部被筛掉了，这些域保存在meta里面就没有意义，
    # 后续提取的时候也容易出现问题，需要进行判断处理，写完实例后，再写入meta，把不包含实例的meta去掉
    reviewCount = 0
    for line in g2:
        theasin = line.strip().split('\t')[0]
        if theasin in reservoir:
            rf_review.write( line.strip()+'\n' )
            reviewCount+=1
        count += 1
        if count % 100000 == 0:
            print str(count * 100 / 9307540.0) + '%'
    rf_review.close()
    print 'actually meta num and review num: ',len(reservoir),reviewCount

# 去除只有三分的meta,整理meta中按'\t'分割含有超过3个的，和评论
def genDeleteNo3():
    metapath='../data/out_domain/exceptSome2_pruned_metadata.txt.gz'
    reviewpath = '../data/out_domain/exceptSome2_pruned_kcore_5.txt.gz'

    rf_meta = gzip.open('../data/out_domain/all_meta_no3.txt.gz', 'w')
    rf_review = gzip.open( '../data/out_domain/all_review_no3.txt.gz', 'w')

    g = gzip.open(metapath, 'r')
    sampleAsin = {}
    countMeta = 0
    sep = 1
    countWrite = 0
    for line in g:
        countMeta += 1
        linesplit = line.strip().split('\t')
        if len(linesplit)<3: continue
        if len(linesplit) > 3:
            write_linesplit = [linesplit[0],' '.join( linesplit[1:-1] ),linesplit[-1] ]
        else:
            write_linesplit = linesplit
        if countMeta%sep==0:
            countWrite+=1
            sampleAsin[linesplit[0]]= [0,'\t'.join( write_linesplit )+'\n']

    g2 = gzip.open(reviewpath, 'r')
    count = 0
    # 如果再次取出掉thescore!=3的，存在可能有些域的实例全部被筛掉了，这些域保存在meta里面就没有意义，
    # 后续提取的时候也容易出现问题，需要进行判断处理，写完实例后，再写入meta，把不包含实例的meta去掉
    sampleAsinKeySet = set(sampleAsin.keys())
    reviewCount = 0
    for line in g2:
        linesplit = line.strip().split('\t')
        theasin = linesplit[0]
        thescore = int(float(linesplit[1]))
        if theasin in sampleAsinKeySet:
            if thescore!=3:
                label = '1' if thescore>3 else '0'
                rf_review.write( '\t'.join([theasin,label,linesplit[2].replace('*',' ') ])+'\n' )
                sampleAsin[theasin][0] += 1
                reviewCount+=1
        count += 1
        if count % 100000 == 0:
            print str(count * 100 / 19961351.0) + '%'
    metaCount = 0
    for k,v in sampleAsin.iteritems():
        if v[0]>0:
            rf_meta.write(v[1])
            metaCount+=1
    rf_meta.close()
    print 'actually meta num and review num: ',metaCount,reviewCount

# 排除类别的meta和实例数 2067174，10219594
# 排除类别并去掉只有3分实例的meta后  331537,9307540
# 采样10w  100000 2796705
# 采样5w   50000 1398921
# 采样1w   10000 274827
#          200000 5640493
# 服务器：



if __name__ == '__main__':
    #genDeleteNo3()
    # reservoirSampling(10000)
    reservoirSampling(200000)
    pass



# meta数据格式 ['0001042335', 'Hamlet: Complete &amp; Unabridged', 'Books\r\r\n']
# review 数据格式 ['0000031887','3.0', '... .\r\n']