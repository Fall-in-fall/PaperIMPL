# -*- encoding:utf-8 -*-
import re
import gzip

# 旧-
# # 除去那些信息不全的商品（信息不全无法使用），只保留编码，标题和类别并重组格式
def cleanMetaData():
    path='C:\\Users\\tyz\\Desktop\\TTSC-paper data\\Amazon Review Data\\metadata.json.gz'
    rf = file('C:\\Users\\tyz\\Desktop\\TTSC-paper data\\Amazon Review Data\\pruned_metadata.txt','w')

    g = gzip.open(path, 'r')
    count=0
    for line in g:
        # if raw_input('go on?')!='':
        #     break
        all=eval(line)
        if all.has_key('asin') and all.has_key('title') and all.has_key('categories'):
            cate=','.join\
                (  '-'.join(i)
                   for i in all['categories']   )
            rf.write(all['asin']+'\t'+all['title']+'\t'+cate+'\n')
        count+=1
        if count%100000==0:
            print str(count*100/9430088.0)+'%'
    rf.close()

# 除去那些信息不全的评论文本（信息不全无法使用）并重组格式
def cleanReviewData():
    # 这里的path可能应该是kcore_5这个文件，待验证
	# path = 'C:\\Users\\tyz\\Desktop\\TTSC-paper data\\Amazon Review Data\\pruned_metadata.txt.gz'
	# rf = file('C:\\Users\\tyz\\Desktop\\TTSC-paper data\\Amazon Review Data\\pruned_kcore_5.txt','a')

	g = gzip.open(path, 'r')
	count=0
	for line in g:
	    all=eval(line)
	    if all.has_key('asin') and all.has_key('overall') and all.has_key('summary') and all.has_key('reviewText'):
	    	rf.write(all['asin']+'\t'+
	                 str(all['overall'])+'\t'+
	                 all['summary']+'*'+
	                 all['reviewText']+'\n')
	    count+=1
	    if count%100000==0:
	    	print str(count*100/41135700.0)+'%'
	rf.close()

# 去除一些类别的meta
def exceptSomeCate():
	path = 'G:\\data collection\\TTSC\\TTSC-paper data\\processed Amazon Review Data\\pruned_metadata.txt.gz'
	# exceptDict=['baby', 'Patio', 'Lawn', 'Garden', 'Health', 'Personal Care','Pet Supplies',
	# 			'Musical Instruments', 'Digital Music', 'Home','Kitchen','Video Games', 'Toys','Games',
	# 			'Clothing','Shoes','Jewelry','Grocery','Gourmet Food', 'Beauty', 'CDs','Vinyl']

	exceptDict=['baby', 'Patio', 'Lawn', 'Garden', 'Health', 'Personal Care','Pet Supplies',
				'Musical Instruments', 'Digital Music', 'Home','Kitchen','Video Games', 'Toys','Games',
				'Books','Jewelry','Grocery','Gourmet Food', 'Beauty', 'CDs','Vinyl']

	rf = file('G:\\data collection\\TTSC\\TTSC-paper data\\processed Amazon Review Data\\exceptSome2_pruned_metadata.txt','w')
	g = gzip.open(path, 'r')
	isFind=False
	count=0
	for line in g:
		count+=1
		if count%100000==0:print str(count*100/9430088.0)+'%'
		isFind=False
		if len(line.split('\t'))<3:
			print 'bad'
			continue
		cate=line.split('\t')[2]
		for x in exceptDict:
			if x in cate:
				isFind=True
				break
		if isFind==False:
			rf.write(line)
	rf.close()

# 除去某些类别的评论
def exceptSomeReview():
    metapath='G:\\data collection\\TTSC\\TTSC-paper data\\processed Amazon Review Data\\exceptSome2_pruned_metadata.txt.gz'
    reviewpath = 'G:\\data collection\\TTSC\\TTSC-paper data\\processed Amazon Review Data\\pruned_kcore_5.txt.gz'
    g = gzip.open(metapath, 'r')
    allAsin = set()
    countMeta = 0
    for line in g:
        countMeta+=1
        if countMeta % 100000 == 0: print str(countMeta * 100 / 2067174.0) + '%'
        allAsin.add(line.strip().split('\t')[0])

    rf = file('G:\\data collection\\TTSC\\TTSC-paper data\\processed Amazon Review Data\\exceptSome2_pruned_kcore_5.txt', 'w')
    g2 = gzip.open(reviewpath, 'r')
    countReview = 0
    countWrite = 0
    for line in g2:
        theasin = line.strip().split('\t')[0]
        countReview += 1
        if countReview % 100000 == 0: print str(countReview * 100 /  41135700.0) + '%'
        if theasin in allAsin:
            countWrite+=1
            rf.write(line)
    print 'countWrite: ',countWrite
    rf.close()

if __name__ == '__main__':
    #exceptSomeCate()
    #exceptSomeReview()
    pass



