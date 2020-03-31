import nltk
import sys
from nltk.corpus import stopwords
import codecs, string
from nltk.corpus import wordnet as wn

ps = nltk.stem.PorterStemmer()
stopW = stopWords = set(stopwords.words('english'))
remove_punctuation_map = dict((ord(char), None) for char in string.punctuation)

def penn_to_wn(tag):
    """ Convert between a Penn Treebank tag to a simplified Wordnet tag """
    if tag.startswith('N'):
        return 'n'
 
    if tag.startswith('V'):
        return 'v'
 
    if tag.startswith('J'):
        return 'a'
 
    if tag.startswith('R'):
        return 'r'
 
    return None



def get_score(f):

	swords = dict()
	CW=0

	
	for l in f:
		l = l.strip().split()
		pos=nltk.pos_tag(l)
		for w, p in pos:
			if w not in stopW:
				p=penn_to_wn(p)
				s=wn.synsets(w, p)
				if w not in string.punctuation: CW+=1	
				if len(s)>0:					
					sw = ps.stem(w)
					wid=w+"_"+(p if p else "")
					if wid in swords: swords[wid][1]+=1
					else: swords[wid] = [s[0],1,p,sw,False]
	RC=0
	LC1=0
	LC2=0
	list_words=list(swords)
	for i,sn in enumerate(list_words):	
		if swords[sn][1]>1 and not swords[sn][4]:
			RC+=swords[sn][1]
			swords[sn][4]=True

		for sn2 in list_words[i+1:]:
			if not swords[sn][4] or not swords[sn2][4]:
				if swords[sn][3] == swords[sn2][3] and swords[sn][2] == swords[sn2][2]:
					if not swords[sn][4]: 
						RC+=swords[sn][1]
						swords[sn][4]=True
					if not swords[sn2][4]: 
						RC+=swords[sn2][1]
						swords[sn2][4]=True
				else:
					fs1, fs2 = swords[sn][0], swords[sn2][0]

					if fs1.path_similarity(fs2) == 1:
						if not swords[sn][4]: 
							LC1+=swords[sn][1]
							swords[sn][4]=True
						if not swords[sn2][4]: 
							LC1+=swords[sn2][1]
							swords[sn2][4]=True

					elif swords[sn][2]==swords[sn2][2]:	
						min_d = fs1.min_depth()	
						min_d2 = fs2.min_depth()
						s3 = fs1.lowest_common_hypernyms(fs2)
						if len(s3)>0:
							s3=s3[0]
							min_d3 = s3.min_depth()
							sim = 2*min_d3/	(min_d+min_d2)
							if sim>=0.96:
								if not swords[sn][4]: 
									LC2+=swords[sn][1]
									swords[sn][4]=True
								if not swords[sn2][4]: 
									LC2+=swords[sn2][1]
									swords[sn2][4]=True
	if CW > 0.0:
		return (RC+LC1+LC2)/float(CW)
	else:
		return 0.0

a = sys.argv[1:]
n=int(a[0])
f_doc=a[n+1:]
f_names=a[1:n+1]
scores=[]

for j in range(n):
	with codecs.open(f_doc[j], encoding='utf-8') as f:
		docs=[int(l.strip()) for l in f]

	with codecs.open(f_names[j], encoding='utf-8') as f:
		i=1
		d=[]
		for l in f:
			if i in docs:
				scores.append(get_score(d))
				d=[]
			i+=1	
			d.append(l.strip())
		scores.append(get_score(d))
		print (sum(scores)/len(scores))
print (sum(scores)/len(scores))
	
				
