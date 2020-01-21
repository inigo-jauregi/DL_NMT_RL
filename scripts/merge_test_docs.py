
#Open text and doc
test_2010_cs = open('cs-en/IWSLT16.TED.tst2010.tc.cs')
test_2010_en = open('cs-en/IWSLT16.TED.tst2010.tc.en')
doc_2010 = open('cs-en/IWSLT16.TED.tst2010.en-cs.doc')

test_2011_cs = open('cs-en/IWSLT16.TED.tst2011.tc.cs')
test_2011_en = open('cs-en/IWSLT16.TED.tst2011.tc.en')
doc_2011 = open('cs-en/IWSLT16.TED.tst2011.en-cs.doc')

test_2012_cs = open('cs-en/IWSLT16.TED.tst2012.tc.cs')
test_2012_en = open('cs-en/IWSLT16.TED.tst2012.tc.en')
doc_2012 = open('cs-en/IWSLT16.TED.tst2012.en-cs.doc')

test_2013_cs = open('cs-en/IWSLT16.TED.tst2013.tc.cs')
test_2013_en = open('cs-en/IWSLT16.TED.tst2013.tc.en')
doc_2013 = open('cs-en/IWSLT16.TED.tst2013.en-cs.doc')

counter_2010=0
list_test_2010_en = []
for line in test_2010_en:
    list_test_2010_en.append(line)
    counter_2010+=1
list_test_2010_cs = []
for line in test_2010_cs:
    list_test_2010_cs.append(line)
list_doc_2010 = []
for line in doc_2010:
    num = int(line.strip())
    list_doc_2010.append(num)


counter_2011=0
list_test_2011_en = []
for line in test_2011_en:
    list_test_2011_en.append(line)
    counter_2011+=1
list_test_2011_cs = []
for line in test_2011_cs:
    list_test_2011_cs.append(line)
list_doc_2011 = []
for line in doc_2011:
    num = int(line.strip())
    num += counter_2010
    list_doc_2011.append(num)

counter_2010_2011 = counter_2010 + counter_2011
#counter_2010_2011 = counter_2011

counter_2012 = 0
list_test_2012_en = []
for line in test_2012_en:
    list_test_2012_en.append(line)
    counter_2012 += 1
list_test_2012_cs = []
for line in test_2012_cs:
    list_test_2012_cs.append(line)
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
list_test_2013_cs = []
for line in test_2013_cs:
    list_test_2013_cs.append(line)
list_doc_2013 = []
for line in doc_2013:
    num = int(line.strip())
    num += counter_2010_2011_2012
    list_doc_2013.append(num)


#Total
total_sents_cs =list_test_2010_cs + list_test_2011_cs + list_test_2012_cs + list_test_2013_cs
total_sents_en =list_test_2010_en + list_test_2011_en + list_test_2012_en + list_test_2013_en
total_docs = list_doc_2010 + list_doc_2011 + list_doc_2012 + list_doc_2013


file_text = open('cs-en/text_2010_2013.cs','w')
for elem in total_sents_cs:
    file_text.write(elem)

file_text = open('cs-en/text_2010_2013.en','w')
for elem in total_sents_en:
    file_text.write(elem)

file_doc = open('cs-en/doc_2010_2013.txt', 'w')
for elem in total_docs:
    file_doc.write(str(elem)+'\n')

