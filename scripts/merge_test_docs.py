
#Open text and doc
test_2010 = open('zh-en/IWSLT15.TED.tst2010.tc.en')
doc_2010 = open('zh-en/IWSLT15.TED.tst2010.zh-en.doc')

test_2011 = open('zh-en/IWSLT15.TED.tst2011.tc.en')
doc_2011 = open('zh-en/IWSLT15.TED.tst2011.zh-en.doc')

test_2012 = open('zh-en/IWSLT15.TED.tst2012.tc.en')
doc_2012 = open('zh-en/IWSLT15.TED.tst2012.zh-en.doc')

test_2013 = open('zh-en/IWSLT15.TED.tst2013.tc.en')
doc_2013 = open('zh-en/IWSLT15.TED.tst2013.zh-en.doc')

counter_2010=0
list_test_2010 = []
for line in test_2010:
    list_test_2010.append(line)
    counter_2010+=1
list_doc_2010 = []
for line in doc_2010:
    num = int(line.strip())
    list_doc_2010.append(num)


counter_2011=0
list_test_2011 = []
for line in test_2011:
    list_test_2011.append(line)
    counter_2011+=1
list_doc_2011 = []
for line in doc_2011:
    num = int(line.strip())
    num += counter_2010
    list_doc_2011.append(num)

counter_2010_2011 = counter_2010 + counter_2011

counter_2012 = 0
list_test_2012 = []
for line in test_2012:
    list_test_2012.append(line)
    counter_2012 += 1
list_doc_2012 = []
for line in doc_2012:
    num = int(line.strip())
    num += counter_2010_2011
    list_doc_2012.append(num)

counter_2010_2011_2012 = counter_2010_2011 +counter_2012


counter_2013 = 0
list_test_2013 = []
for line in test_2013:
    list_test_2013.append(line)
    counter_2013 += 1
list_doc_2013 = []
for line in doc_2013:
    num = int(line.strip())
    num += counter_2010_2011_2012
    list_doc_2013.append(num)


#Total
total_sents = list_test_2010 + list_test_2011 + list_test_2012 + list_test_2013
total_docs = list_doc_2010 + list_doc_2011 + list_doc_2012 + list_doc_2013


file_text = open('text.txt','w')
for elem in total_sents:
    file_text.write(elem)

file_doc = open('doc.txt', 'w')
for elem in total_docs:
    file_doc.write(str(elem)+'\n')

