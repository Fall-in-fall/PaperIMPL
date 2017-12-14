topic_addr = './data/topic/final'
topicData = readTopicData(topic_addr)

resDict_non2topic = {}
for k, v in topicData.iteritems():
    print 'test in topic "{}"'.format(k)
    test_set, test_label = v['text'], v['label']
    train_set, train_label = nonTopicData['text'], nonTopicData['label']
    res = classificationTest(train_set, train_label, test_set, test_label,classifier=MultinomialNB())
    resDict_non2topic[k] = res

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