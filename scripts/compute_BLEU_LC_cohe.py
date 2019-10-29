import os
from tqdm import tqdm

model_path = '../full_source/zh-en_models/HAN_join_newCode_plus_RL_0.33'
seed='/seed_1'
dev = '/dev'
devs = ['/dev1', '/dev2', '/dev3', '/dev4', '/dev5']



# Datos
f = open(model_path+seed+dev+'_analysis.txt','w')
for d in tqdm(devs):
    path = model_path+seed+dev+d+'.txt'
    print (d)
    # First BLEU
    BLEU_TEXT = os.system('perl ../mosesdecoder/scripts/generic/multi-bleu.perl '
                          '../zh-en/IWSLT15.TED.dev2010.tc.en '
                          '< '+path)

    # LC
    LC_text = os.system('python LC_RC.py 1 '+path+' '
                        '../zh-en/IWSLT15.TED.dev2010.zh-en.doc')

    # print (LC_text)
    # print (type(LC_text))

    # Coherence
    Coherence_text = os.system('python compute_coherence.py 1 ' + path + ' '
                                '../zh-en/IWSLT15.TED.dev2010.zh-en.doc')

    # print (Coherence_text)
    # print (type(Coherence_text))

    #Write results
    f.write('********** '+d+' ************\n')
    f.write('BLEU score\n')
    f.write(str(BLEU_TEXT)+'\n')
    f.write('LC score\n')
    f.write(str(LC_text)+'\n')
    f.write('Coherence score\n')
    f.write(str(Coherence_text)+'\n')