#python translate.py -model zh-en_models/HAN_join_newCode_plus_RL_0.33/seed_1/EPOCH_acc_43.38_ppl_22.65_e1_num2.pt -src ../zh-en/text_2011_2013.zh -doc ../zh-en/doc_2011_2013.txt -output zh-en_models/HAN_join_newCode_plus_RL_0.33/seed_1/test_2011_2013_ppl.txt -translate_part all -batch_size 1000 -gpu 1

#python translate.py -model zh-en_models/BASELINE_IWSLT2015_newCode/seed_1/EPOCH_acc_42.66_ppl_23.58_e16.pt -src ../zh-en/text_2011_2013.zh -doc ../zh-en/doc_2011_2013.txt -output zh-en_models/BASELINE_IWSLT2015_newCode/seed_1/test_2011_2013_acc.txt -translate_part all -batch_size 1000 -gpu 1

#python translate.py -model zh-en_models/HAN_join_newCode/acc/seed_1/EPOCH_acc_43.37_ppl_22.51_e1.pt -src ../zh-en/text_2011_2013.zh -doc ../zh-en/doc_2011_2013.txt -output zh-en_models/HAN_join_newCode/acc/seed_1/test_2011_2013.txt -translate_part all -batch_size 1000 -gpu 1

#python translate.py -model zh-en_models/HAN_join_newCode/ppl/seed_1/EPOCH_acc_43.17_ppl_22.94_e1.pt -src ../zh-en/text_2011_2013.zh -doc ../zh-en/doc_2011_2013.txt -output zh-en_models/HAN_join_newCode/ppl/seed_1/test_2011_2013.txt -translate_part all -batch_size 1000 -gpu 1

#LC
#python ../scripts/LC_RC.py 1 zh-en_models/HAN_join_newCode_plus_RL_0.33/seed_1/test_2011_2013_ppl.txt ../zh-en/doc_2011_2013.txt > zh-en_models/HAN_join_newCode_plus_RL_0.33/seed_1/LC_2011_2013_ppl.txt

#python ../scripts/LC_RC.py 1 zh-en_models/BASELINE_IWSLT2015_newCode/seed_1/test_2011_2013_ppl.txt ../zh-en/doc_2011_2013.txt > zh-en_models/BASELINE_IWSLT2015_newCode/seed_1/LC_2011_2013_ppl.txt

#python ../scripts/LC_RC.py 1 zh-en_models/HAN_join_newCode/ppl/seed_1/test_2011_2013.txt ../zh-en/doc_2011_2013.txt > zh-en_models/HAN_join_newCode/ppl/seed_1/LC_2011_2013.txt

#python ../scripts/LC_RC.py 1 zh-en_models/HAN_join_newCode/acc/seed_1/test_2011_2013.txt ../zh-en/doc_2011_2013.txt > zh-en_models/HAN_join_newCode/acc/seed_1/LC_2011_2013.txt

#Coherence
python ../scripts/compute_coherence.py 1 zh-en_models/HAN_join_newCode_plus_RL_0.33/seed_1/test_2011_2013_ppl.txt ../zh-en/doc_2011_2013.txt > zh-en_models/HAN_join_newCode_plus_RL_0.33/seed_1/coherence_2011_2013_ppl.txt


#BASELINE

#Seed 1
#python translate.py -model zh-en_models/BASELINE_IWSLT2015/seed_1/EPOCH_acc_42.39_ppl_24.27_e14.pt -src ../zh-en/IWSLT15.TED.tst2010.tc.zh -doc ../zh-en/IWSLT15.TED.tst2010.zh-en.doc -output zh-en_models/BASELINE_IWSLT2015/seed_1/test_2010.txt -translate_part all -batch_size 1000 -gpu 1

#python translate.py -model zh-en_models/BASELINE_IWSLT2015/seed_1/EPOCH_acc_42.39_ppl_24.27_e14.pt -src ../zh-en/IWSLT15.TED.tst2011.tc.zh -doc ../zh-en/IWSLT15.TED.tst2011.zh-en.doc -output zh-en_models/BASELINE_IWSLT2015/seed_1/test_2011.txt -translate_part all -batch_size 1000 -gpu 1

#python translate.py -model zh-en_models/BASELINE_IWSLT2015/seed_1/EPOCH_acc_42.39_ppl_24.27_e14.pt -src ../zh-en/IWSLT15.TED.tst2012.tc.zh -doc ../zh-en/IWSLT15.TED.tst2012.zh-en.doc -output zh-en_models/BASELINE_IWSLT2015/seed_1/test_2012.txt -translate_part all -batch_size 1000 -gpu 1

#python translate.py -model zh-en_models/BASELINE_IWSLT2015/seed_1/EPOCH_acc_42.39_ppl_24.27_e14.pt -src ../zh-en/IWSLT15.TED.tst2013.tc.zh -doc ../zh-en/IWSLT15.TED.tst2013.zh-en.doc -output zh-en_models/BASELINE_IWSLT2015/seed_1/test_2013.txt -translate_part all -batch_size 1000 -gpu 1

#python translate.py -model zh-en_models/BASELINE/seed_1/chosen_EPOCH_acc_41.75_ppl_25.49_e16.pt -src ../../HAN_NMT/zh-en/IWSLT15.TED.tst2013.tc.zh -doc ../../HAN_NMT/zh-en/IWSLT15.TED.tst2013.zh-en.doc -output zh-en_models/BASELINE/seed_1/chosen_all_test_2013.txt -translate_part all -batch_size 1000 -gpu 0

#Seed 2
#python translate.py -model zh-en_models/BASELINE/seed_2/chosen_EPOCH_acc_41.69_ppl_25.11_e14.pt -src ../../HAN_NMT/zh-en/IWSLT15.TED.tst2011.tc.zh -doc ../../HAN_NMT/zh-en/IWSLT15.TED.tst2011.zh-en.doc -output zh-en_models/BASELINE/seed_2/chosen_all_test_2011.txt -translate_part all -batch_size 1000 -gpu 0

#python translate.py -model zh-en_models/BASELINE/seed_2/chosen_EPOCH_acc_41.69_ppl_25.11_e14.pt -src ../../HAN_NMT/zh-en/IWSLT15.TED.tst2012.tc.zh -doc ../../HAN_NMT/zh-en/IWSLT15.TED.tst2012.zh-en.doc -output zh-en_models/BASELINE/seed_2/chosen_all_test_2012.txt -translate_part all -batch_size 1000 -gpu 0

#python translate.py -model zh-en_models/BASELINE/seed_2/chosen_EPOCH_acc_41.69_ppl_25.11_e14.pt -src ../../HAN_NMT/zh-en/IWSLT15.TED.tst2013.tc.zh -doc ../../HAN_NMT/zh-en/IWSLT15.TED.tst2013.zh-en.doc -output zh-en_models/BASELINE/seed_2/chosen_all_test_2013.txt -translate_part all -batch_size 1000 -gpu 0

#Seed 3
#python translate.py -model zh-en_models/BASELINE/seed_3/chosen_EPOCH_acc_41.92_ppl_24.82_e13.pt -src ../../HAN_NMT/zh-en/IWSLT15.TED.tst2011.tc.zh -doc ../../HAN_NMT/zh-en/IWSLT15.TED.tst2011.zh-en.doc -output zh-en_models/BASELINE/seed_3/chosen_all_test_2011.txt -translate_part all -batch_size 1000 -gpu 0

#python translate.py -model zh-en_models/BASELINE/seed_3/chosen_EPOCH_acc_41.92_ppl_24.82_e13.pt -src ../../HAN_NMT/zh-en/IWSLT15.TED.tst2012.tc.zh -doc ../../HAN_NMT/zh-en/IWSLT15.TED.tst2012.zh-en.doc -output zh-en_models/BASELINE/seed_3/chosen_all_test_2012.txt -translate_part all -batch_size 1000 -gpu 0

#python translate.py -model zh-en_models/BASELINE/seed_3/chosen_EPOCH_acc_41.92_ppl_24.82_e13.pt -src ../../HAN_NMT/zh-en/IWSLT15.TED.tst2013.tc.zh -doc ../../HAN_NMT/zh-en/IWSLT15.TED.tst2013.zh-en.doc -output zh-en_models/BASELINE/seed_3/chosen_all_test_2013.txt -translate_part all -batch_size 1000 -gpu 0


#HAN_join


#Seed 1
#python translate.py -model zh-en_models/HAN_join/seed_1/EPOCH_acc_42.93_ppl_23.09_e1.pt -src ../zh-en/IWSLT15.TED.tst2010.tc.zh -doc ../zh-en/IWSLT15.TED.tst2010.zh-en.doc -output zh-en_models/HAN_join/seed_1/test_2010.txt -translate_part all -batch_size 1000 -gpu 1

#python translate.py -model zh-en_models/HAN_join/seed_1/EPOCH_acc_42.93_ppl_23.09_e1.pt -src ../zh-en/IWSLT15.TED.tst2011.tc.zh -doc ../zh-en/IWSLT15.TED.tst2011.zh-en.doc -output zh-en_models/HAN_join/seed_1/test_2011.txt -translate_part all -batch_size 1000 -gpu 1

#python translate.py -model zh-en_models/HAN_join/seed_1/EPOCH_acc_42.93_ppl_23.09_e1.pt -src ../zh-en/IWSLT15.TED.tst2012.tc.zh -doc ../zh-en/IWSLT15.TED.tst2012.zh-en.doc -output zh-en_models/HAN_join/seed_1/test_2012.txt -translate_part all -batch_size 1000 -gpu 1

#python translate.py -model zh-en_models/HAN_join/seed_1/EPOCH_acc_42.93_ppl_23.09_e1.pt -src ../zh-en/IWSLT15.TED.tst2013.tc.zh -doc ../zh-en/IWSLT15.TED.tst2013.zh-en.doc -output zh-en_models/HAN_join/seed_1/test_2013.txt -translate_part all -batch_size 1000 -gpu 1

#python translate.py -model zh-en_models/HAN_join/seed_1/EPOCH_acc_41.70_ppl_25.08_e1.pt -src ../../HAN_NMT/zh-en/IWSLT15.TED.tst2012.tc.zh -doc ../../HAN_NMT/zh-en/IWSLT15.TED.tst2012.zh-en.doc -output zh-en_models/HAN_join/seed_1/e1_all_test_2012.txt -translate_part all -batch_size 1000 -gpu 0

#python translate.py -model zh-en_models/HAN_join/seed_1/EPOCH_acc_41.70_ppl_25.08_e1.pt -src ../../HAN_NMT/zh-en/IWSLT15.TED.tst2013.tc.zh -doc ../../HAN_NMT/zh-en/IWSLT15.TED.tst2013.zh-en.doc -output zh-en_models/HAN_join/seed_1/e1_all_test_2013.txt -translate_part all -batch_size 1000 -gpu 0


#Seed 2
#python translate.py -model zh-en_models/HAN_join/seed_2/EPOCH_acc_41.35_ppl_25.40_e1.pt -src ../../HAN_NMT/zh-en/IWSLT15.TED.tst2011.tc.zh -doc ../../HAN_NMT/zh-en/IWSLT15.TED.tst2011.zh-en.doc -output zh-en_models/HAN_join/seed_2/e1_all_test_2011.txt -translate_part all -batch_size 1000 -gpu 0

#python translate.py -model zh-en_models/HAN_join/seed_2/EPOCH_acc_41.35_ppl_25.40_e1.pt -src ../../HAN_NMT/zh-en/IWSLT15.TED.tst2012.tc.zh -doc ../../HAN_NMT/zh-en/IWSLT15.TED.tst2012.zh-en.doc -output zh-en_models/HAN_join/seed_2/e1_all_test_2012.txt -translate_part all -batch_size 1000 -gpu 0

#python translate.py -model zh-en_models/HAN_join/seed_2/EPOCH_acc_41.35_ppl_25.40_e1.pt -src ../../HAN_NMT/zh-en/IWSLT15.TED.tst2013.tc.zh -doc ../../HAN_NMT/zh-en/IWSLT15.TED.tst2013.zh-en.doc -output zh-en_models/HAN_join/seed_2/e1_all_test_2013.txt -translate_part all -batch_size 1000 -gpu 0


#Seed 3
#python translate.py -model zh-en_models/HAN_join/seed_3/EPOCH_acc_41.39_ppl_25.63_e1.pt -src ../../HAN_NMT/zh-en/IWSLT15.TED.tst2011.tc.zh -doc ../../HAN_NMT/zh-en/IWSLT15.TED.tst2011.zh-en.doc -output zh-en_models/HAN_join/seed_3/e1_all_test_2011.txt -translate_part all -batch_size 1000 -gpu 0

#python translate.py -model zh-en_models/HAN_join/seed_3/EPOCH_acc_41.39_ppl_25.63_e1.pt -src ../../HAN_NMT/zh-en/IWSLT15.TED.tst2012.tc.zh -doc ../../HAN_NMT/zh-en/IWSLT15.TED.tst2012.zh-en.doc -output zh-en_models/HAN_join/seed_3/e1_all_test_2012.txt -translate_part all -batch_size 1000 -gpu 0

#python translate.py -model zh-en_models/HAN_join/seed_3/EPOCH_acc_41.39_ppl_25.63_e1.pt -src ../../HAN_NMT/zh-en/IWSLT15.TED.tst2013.tc.zh -doc ../../HAN_NMT/zh-en/IWSLT15.TED.tst2013.zh-en.doc -output zh-en_models/HAN_join/seed_3/e1_all_test_2013.txt -translate_part all -batch_size 1000 -gpu 0
