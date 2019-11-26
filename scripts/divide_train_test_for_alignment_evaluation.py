
train_doc = open('../zh-en/corpus.tok.zh')
alignment_doc = open('../full_source/zh-en_models/HAN_join_newCode/ppl/seed_1/train_plus_test_2011_2013_align.txt')

align_test_file = open('../full_source/zh-en_models/HAN_join_newCode/ppl/seed_1/align_2011_2013.txt','w')

list_train = []
for line in train_doc:
    list_train.append(line)

num_train_sentences = len(list_train)


list_align = []
for line in alignment_doc:
    list_align.append(line)


list_align = list_align[num_train_sentences:]

print (len(list_align))

for ele in list_align:
    align_test_file.write(ele)

align_test_file.close()