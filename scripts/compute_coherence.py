import codecs
import sys
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity



#Load LSA model
print ('Loading LSA model...')
vect_dic = {}
with codecs.open('../scripts/coherence_model/Wiki_6/voc', encoding='utf-8') as f:
    words = [l.strip() for l in f]
with codecs.open('../scripts/coherence_model/Wiki_6/lsaModel', encoding='utf-8') as f:
    vec_list = []
    for l in f:
        l=l.split()
        # l=[float(n) for n in l]
        l=np.asarray(l,dtype=float)
        vec_list.append(l)

# Create dictionary
for i in range(len(words)):
    # print (words[i])
    vect_dic[words[i]] = vec_list[i]

print ('Number of words: ', len(vect_dic))


def get_sen_embedding(sentence):

    #Split
    sentence = sentence.replace('\n','')
    tokenized_sen = sentence.split()
    sen_vec = np.zeros(300)
    count_words = 0
    for word in tokenized_sen:
        theWord = word.lower()
        if theWord in vect_dic:
            count_words +=1
            word_vec = vect_dic[theWord]
            sen_vec+=word_vec

    if count_words!=0:
        sen_vec = sen_vec / count_words

    return sen_vec

def compute_coherence(document_vectors):

    n_sentences = len(document_vectors)
    scores = []
    for i in range(n_sentences-1):
        # print (document_vectors[i])
        # print (document_vectors[i+1])
        cos_sim = cosine_similarity([document_vectors[i]],[document_vectors[i+1]])
        scores.append(cos_sim)

    # Average similarity
    avg_cos_sim = sum(scores)/float(len(scores))
    return avg_cos_sim


a = sys.argv[1:]
n=int(a[0])
f_doc=a[n+1:]
f_names=a[1:n+1]
scores = []

#Load doc file and predictions
for j in range(n):
    with codecs.open(f_doc[j], encoding='utf-8') as f:
        docs=[int(l.strip()) for l in f]

    with codecs.open(f_names[j], encoding='utf-8') as f:
        i = 1
        d = []
        for l in f:
            if i in docs:
                scores.append(compute_coherence(d))
                d = []
            i += 1
            d.append(get_sen_embedding(l))
        scores.append(compute_coherence(d))
        print(sum(scores) / len(scores))

print (sum(scores)/len(scores))
