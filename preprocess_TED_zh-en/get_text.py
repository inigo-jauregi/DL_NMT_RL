import sys, re

a=sys.argv

folder = '../en-es_2013/'

f_s = open(folder+"train.tags.en-es.es")
f_t = open(folder+"train.tags.en-es.en")
f_s_o = open(folder+"corpus.es", "w")
f_t_o = open(folder+"corpus.en", "w")
f_doc = open(folder+"corpus.doc", "w")
f_s_doc = open(folder+"corpus.doc.es", "w")
f_t_doc = open(folder+"corpus.doc.en", "w")

count = 0
for ls, lt in zip(f_s, f_t):
	if ls.startswith("<title>"):
		if not lt.startswith("<title>"): 
			print ("katxuli")
			print ("error "+str(count))
			break
		ls = re.sub("(^\<title\>)(.*)(\</title\>)","\g<2>", ls).strip()
		lt = re.sub("(^\<title\>)(.*)(\</title\>)","\g<2>", lt).strip()

		f_doc.write(str(count)+"\n")
		f_s_doc.write(ls + "\n")
		f_t_doc.write(lt + "\n")

		

	elif ls.startswith("<transcript>"):
		if not lt.startswith("<transcript>"): 
			print ("error "+str(count))
			break
		ls = re.sub("(^\<transcript\>)(.*)(\</transcript\>)","\g<2>", ls).strip()
		lt = re.sub("(^\<transcript\>)(.*)(\</transcript\>)","\g<2>", lt).strip()
		f_s_doc.write(ls + "\n")
		f_t_doc.write(lt + "\n")

	elif not ls.startswith("<"):
		if ls.strip()!= "" and lt.strip()!= "":
			f_s_o.write(ls.strip() + "\n")
			f_t_o.write(lt.strip() + "\n")
			count +=1

f_s.close()
f_t.close()
f_s_o.close()
f_t_o.close()
f_doc.close()
f_s_doc.close()
f_t_doc.close()


for test in ["dev2010", "tst2010", "tst2011", "tst2012"]:
	f_s = open(folder+"IWSLT14.TED." + test +".en-es.es.xml")
	f_t = open(folder+"IWSLT14.TED." + test +".en-es.en.xml")

	count = 0

	f_s_o = open(folder+"IWSLT14.TED." + test +".en-es.es", "w")
	f_t_o = open(folder+"IWSLT14.TED." + test +".en-es.en", "w")
	f_doc = open(folder+"IWSLT14.TED." + test +".en-es.doc", "w")
	f_s_doc = open(folder+"IWSLT14.TED." + test +".en-es.doc.es", "w")
	f_t_doc = open(folder+"IWSLT14.TED." + test +".en-es.doc.en", "w")

	for ls, lt in zip(f_s, f_t):
		if ls.startswith("<talkid>"):
			if not lt.startswith("<talkid>"): 
				print ("katxuli")
				print ("error "+str(count))
				break
			s = re.sub("(^\<talkid\>)(.*)(\</talkid\>)","\g<2>", ls).strip()
			t = re.sub("(^\<talkid\>)(.*)(\</talkid\>)","\g<2>", lt).strip()

			if s!=t:
				print ("katxuli")
				print ("error "+str(count)+" "+test)
				break

			f_s_doc.write(ls.strip() + "\n")
			f_t_doc.write(lt.strip() + "\n")
			f_doc.write(str(count) + "\n")
			

		elif ls.startswith("<seg"): 
			if not lt.startswith("<seg"): 
				print ("error "+str(count)+" "+test)
				break

			ls = re.sub("(^\<seg.*\>)(.*)(\</seg\>)","\g<2>", ls).strip()
			lt = re.sub("(^\<seg.*\>)(.*)(\</seg\>)","\g<2>", lt).strip()

			if ls.strip()!= "" and lt.strip()!= "":
				f_s_o.write(ls + "\n")
				f_t_o.write(lt + "\n")
				count += 1
		else:

			f_s_doc.write(ls.strip() + "\n")
			f_t_doc.write(lt.strip() + "\n")

	f_s.close()
	f_t.close()
	f_s_o.close()
	f_t_o.close()
	f_doc.close()
	f_s_doc.close()
	f_t_doc.close()		
