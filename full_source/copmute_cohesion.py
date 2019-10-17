import nltk
from nltk.tokenize import word_tokenize, RegexpTokenizer
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
from nltk.stem import PorterStemmer
from collections import Counter
from tqdm import tqdm


tokenizer = RegexpTokenizer(r'\w+')
stemmer = PorterStemmer()

stop_words = stopwords.words('english')


doc_lines_file = open('../zh-en/IWSLT15.TED.tst2010.zh-en.doc')

doc_lines = []
for line in doc_lines_file:
    num = int(line.replace('\n','').strip())
    doc_lines.append(num)


# Pred test
pred_file = open('../zh-en/IWSLT15.TED.tst2010.tc.en')
counter = 0
document_list = []
for line in pred_file:
    if counter == 0:
        buffer = []
    elif counter in doc_lines:
        document_list.append(" ".join(buffer))
        buffer = []

    #Add sentence to buffer
    line = line.replace('\n','')
    buffer.append(line)
    counter+=1
document_list.append(" ".join(buffer))


print (len(doc_lines))
print (len(document_list))


def cohesion_score_per_doc(doc):

    #lowercase
    # print (doc)
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
                if wup_score > 0.96:
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



#We have the documents now we start computing the cohesion score

scores = []
for document in tqdm(document_list):

    score = cohesion_score_per_doc(document)
    scores.append(score)

print ('Coherence score: ', sum(scores)/float(len(scores)))