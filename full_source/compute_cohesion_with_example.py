import nltk
from nltk.tokenize import word_tokenize, RegexpTokenizer
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
from nltk.stem import PorterStemmer
from collections import Counter


tokenizer = RegexpTokenizer(r'\w+')
stemmer = PorterStemmer()

stop_words = stopwords.words('english')

document_1 = ['Chine scrambled research on 16 key technical These techniques are from within headline everyones boosting'
              ' science and technology achieving goals and contend of delivered on time bound through achieving breakthroughs'
              'in essential technology and complimentarity resources . national']

document_2 = ['China is accelerating research 16 main technologies These technologies are within the important realm to '
              'promote sciences and technology and achieve national goals and must be completed in a timely manner through'
              ' achieving main discoveries in technology and integration resources .']


def cohesion_score_per_doc(doc):

    #lowercase
    doc = doc.lower()

    # Tokenization and remove punctuation
    doc_tok = tokenizer.tokenize(doc)

    # Remove Stop-words english
    doc_tok_clean = [w for w in doc_tok if w not in stop_words]

    # Stem
    doc_tok_stems = [stemmer.stem(w) for w in doc_tok_clean]

    # POS tags (only over unique words)
    doc_pos_tags = nltk.pos_tag(sorted(set(doc_tok_clean),key=doc_tok_clean.index))
    #map POS tags
    doc_pos_tags_mapped = []
    for pair in doc_pos_tags:
        if pair[1].startswith('N'):
            doc_pos_tags_mapped.append((pair[0],'n'))
        elif pair[1].startswith('V'):
            doc_pos_tags_mapped.append((pair[0],'v'))
        elif pair[1].startswith('J'):
            doc_pos_tags_mapped.append((pair[0],'a'))
        elif pair[1].startswith('R'):
            doc_pos_tags_mapped.append((pair[0],'r'))
        else:
            doc_pos_tags_mapped.append((pair[0],'none'))

    # synsets (if many select the first synset)
    synset_list = []
    for pair in doc_pos_tags_mapped:
        if pair[1]=='none':
            theSynsets = wn.synsets(pair[0])
            if len(theSynsets)>0:
                synset_list.append(theSynsets[0])
        else:
            theSynsets = wn.synsets(pair[0],pos=pair[1])
            if len(theSynsets)>0:
                synset_list.append(theSynsets[0])


    # Counts
    result_counts = Counter(doc_tok_stems)
    repeated_words = []
    repetition_counts = 0
    for elem in result_counts:
        if result_counts[elem]>1:
            repetition_counts+=result_counts[elem]
            repeated_words.append(elem)


    # Synonym, hypernim...
    count_occurences_synonyms=0
    synonym_list = []
    count_occurences_similarity = 0
    similar_list=[]
    for i in range(len(synset_list)-1):
        for j in range(i+1,len(synset_list)):
            dis = synset_list[i].path_similarity(synset_list[j])
            if dis == 1:
                count_occurences_synonyms+=1
                synonym_list.append((synset_list[i],synset_list[j]))
            else:
                wup_score = synset_list[i].wup_similarity(synset_list[j])
                if wup_score == None:
                    wup_score = 0
                if wup_score >= 0.96:
                    count_occurences_similarity+=1
                    similar_list.append((synset_list[i], synset_list[j]))




    # Semantic similarity
    lc = (repetition_counts + count_occurences_similarity + count_occurences_synonyms)/float(len(doc_tok))
    print ('Repetition: ',repetition_counts)
    print (repeated_words)
    print ('Superordinate and collocation: ',count_occurences_synonyms)
    print (synonym_list)
    print ('Synonym and near-synonyms: ',count_occurences_similarity)
    print (similar_list)

    return lc

print ('Score 1: ', cohesion_score_per_doc(document_1[0]))
print ('Score 2: ', cohesion_score_per_doc(document_2[0]))
