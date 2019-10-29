
#Open text and doc
#test_2010 = open('zh-en/IWSLT15.TED.tst2010.tc.en')
#doc_2010 = open('zh-en/IWSLT15.TED.tst2010.zh-en.doc')

test_2011_zh = open('zh-en/IWSLT15.TED.tst2011.tc.zh')
test_2011_en = open('zh-en/IWSLT15.TED.tst2011.tc.en')
doc_2011 = open('zh-en/IWSLT15.TED.tst2011.zh-en.doc')

test_2012_zh = open('zh-en/IWSLT15.TED.tst2012.tc.zh')
test_2012_en = open('zh-en/IWSLT15.TED.tst2012.tc.en')
doc_2012 = open('zh-en/IWSLT15.TED.tst2012.zh-en.doc')

test_2013_zh = open('zh-en/IWSLT15.TED.tst2013.tc.zh')
test_2013_en = open('zh-en/IWSLT15.TED.tst2013.tc.en')
doc_2013 = open('zh-en/IWSLT15.TED.tst2013.zh-en.doc')

#counter_2010=0
#list_test_2010 = []
#for line in test_2010:
#    list_test_2010.append(line)
#    counter_2010+=1
#list_doc_2010 = []
#for line in doc_2010:
#    num = int(line.strip())
#    list_doc_2010.append(num)


counter_2011=0
list_test_2011_en = []
for line in test_2011_en:
    list_test_2011_en.append(line)
    counter_2011+=1
list_test_2011_zh = []
for line in test_2011_zh:
    list_test_2011_zh.append(line)
list_doc_2011 = []
for line in doc_2011:
    num = int(line.strip())
    #num += counter_2010
    list_doc_2011.append(num)

#counter_2010_2011 = counter_2010 + counter_2011
counter_2010_2011 = counter_2011

counter_2012 = 0
list_test_2012_en = []
for line in test_2012_en:
    list_test_2012_en.append(line)
    counter_2012 += 1
list_test_2012_zh = []
for line in test_2012_zh:
    list_test_2012_zh.append(line)
list_doc_2012 = []
for line in doc_2012:
    num = int(line.strip())
    num += counter_2010_2011
    list_doc_2012.append(num)

counter_2010_2011_2012 = counter_2010_2011 +counter_2012


counter_2013 = 0
list_test_2013_en = []
for line in test_2013_en:
    list_test_2013_en.append(line)
    counter_2013 += 1
list_test_2013_zh = []
for line in test_2013_zh:
    list_test_2013_zh.append(line)
list_doc_2013 = []
for line in doc_2013:
    num = int(line.strip())
    num += counter_2010_2011_2012
    list_doc_2013.append(num)


#Total
total_sents_zh =list_test_2011_zh + list_test_2012_zh + list_test_2013_zh
total_sents_en =list_test_2011_en + list_test_2012_en + list_test_2013_en
total_docs = list_doc_2011 + list_doc_2012 + list_doc_2013


file_text = open('zh-en/text_2011_2013.zh','w')
for elem in total_sents_zh:
    file_text.write(elem)

file_text = open('zh-en/text_2011_2013.en','w')
for elem in total_sents_en:
    file_text.write(elem)

file_doc = open('zh-en/doc_2011_2013.txt', 'w')
for elem in total_docs:
    file_doc.write(str(elem)+'\n')

