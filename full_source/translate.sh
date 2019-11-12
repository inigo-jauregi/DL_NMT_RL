#BASELINE

python translate.py -model zh-en_models/BASELINE_IWSLT2015_newCode/seed_2/EPOCH_acc_42.18_ppl_22.93_e14.pt -src ../zh-en/text_2011_2013.zh -doc ../zh-en/doc_2011_2013.txt -output zh-en_models/BASELINE_IWSLT2015_newCode/seed_2/test_2011_2013.txt -translate_part sentences -batch_size 1000 -gpu 0

python translate.py -model zh-en_models/BASELINE_IWSLT2015_newCode/seed_3/EPOCH_acc_42.68_ppl_22.67_e14.pt -src ../zh-en/text_2011_2013.zh -doc ../zh-en/doc_2011_2013.txt -output zh-en_models/BASELINE_IWSLT2015_newCode/seed_3/test_2011_2013.txt -translate_part sentences -batch_size 1000 -gpu 0

#HAN

python translate.py -model zh-en_models/HAN_join_newCode/ppl/seed_2/EPOCH_acc_42.98_ppl_22.59_e4.pt -src ../zh-en/text_2011_2013.zh -doc ../zh-en/doc_2011_2013.txt -output zh-en_models/HAN_join_newCode/ppl/seed_2/test_2011_2013.txt -translate_part all -batch_size 1000 -gpu 0

python translate.py -model zh-en_models/HAN_join_newCode/ppl/seed_3/EPOCH_acc_43.54_ppl_22.36_e4.pt -src ../zh-en/text_2011_2013.zh -doc ../zh-en/doc_2011_2013.txt -output zh-en_models/HAN_join_newCode/ppl/seed_3/test_2011_2013.txt -translate_part all -batch_size 1000 -gpu 0

#RL

python translate.py -model zh-en_models/HAN_join_newCode_plus_RL/seed_2/EPOCH_acc_42.92_ppl_22.62_e1_num1.pt -src ../zh-en/IWSLT15.TED.dev2010.tc.zh -doc ../zh-en/IWSLT15.TED.dev2010.zh-en.doc -output zh-en_models/HAN_join_newCode_plus_RL/seed_2/dev/dev_1.txt -translate_part all -batch_size 1000 -gpu 0

python translate.py -model zh-en_models/HAN_join_newCode_plus_RL/seed_2/EPOCH_acc_42.74_ppl_23.74_e1_num2.pt -src ../zh-en/IWSLT15.TED.dev2010.tc.zh -doc ../zh-en/IWSLT15.TED.dev2010.zh-en.doc -output zh-en_models/HAN_join_newCode_plus_RL/seed_2/dev/dev_2.txt -translate_part all -batch_size 1000 -gpu 0

python translate.py -model zh-en_models/HAN_join_newCode_plus_RL/seed_2/EPOCH_acc_42.75_ppl_23.76_e1_num3.pt -src ../zh-en/IWSLT15.TED.dev2010.tc.zh -doc ../zh-en/IWSLT15.TED.dev2010.zh-en.doc -output zh-en_models/HAN_join_newCode_plus_RL/seed_2/dev/dev_3.txt -translate_part all -batch_size 1000 -gpu 0

python translate.py -model zh-en_models/HAN_join_newCode_plus_RL/seed_2/EPOCH_acc_42.78_ppl_23.58_e1_num4.pt -src ../zh-en/IWSLT15.TED.dev2010.tc.zh -doc ../zh-en/IWSLT15.TED.dev2010.zh-en.doc -output zh-en_models/HAN_join_newCode_plus_RL/seed_2/dev/dev_4.txt -translate_part all -batch_size 1000 -gpu 0

#


python translate.py -model zh-en_models/HAN_join_newCode_plus_RL/seed_3/EPOCH_acc_43.53_ppl_22.41_e1_num1.pt -src ../zh-en/IWSLT15.TED.dev2010.tc.zh -doc ../zh-en/IWSLT15.TED.dev2010.zh-en.doc -output zh-en_models/HAN_join_newCode_plus_RL/seed_3/dev/dev_1.txt -translate_part all -batch_size 1000 -gpu 0

python translate.py -model zh-en_models/HAN_join_newCode_plus_RL/seed_3/EPOCH_acc_43.30_ppl_23.49_e1_num2.pt -src ../zh-en/IWSLT15.TED.dev2010.tc.zh -doc ../zh-en/IWSLT15.TED.dev2010.zh-en.doc -output zh-en_models/HAN_join_newCode_plus_RL/seed_3/dev/dev_2.txt -translate_part all -batch_size 1000 -gpu 0

python translate.py -model zh-en_models/HAN_join_newCode_plus_RL/seed_3/EPOCH_acc_43.30_ppl_23.29_e1_num3.pt -src ../zh-en/IWSLT15.TED.dev2010.tc.zh -doc ../zh-en/IWSLT15.TED.dev2010.zh-en.doc -output zh-en_models/HAN_join_newCode_plus_RL/seed_3/dev/dev_3.txt -translate_part all -batch_size 1000 -gpu 0

python translate.py -model zh-en_models/HAN_join_newCode_plus_RL/seed_3/EPOCH_acc_43.11_ppl_23.52_e1_num4.pt -src ../zh-en/IWSLT15.TED.dev2010.tc.zh -doc ../zh-en/IWSLT15.TED.dev2010.zh-en.doc -output zh-en_models/HAN_join_newCode_plus_RL/seed_3/dev/dev_4.txt -translate_part all -batch_size 1000 -gpu 0


