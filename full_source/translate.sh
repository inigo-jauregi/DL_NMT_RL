python translate.py -model zh-en_models/BASELINE_IWSLT2015/seed_1/EPOCH_acc_42.39_ppl_24.27_e14.pt -src ../test_out/TED_zh-en/IWSLT15.TED.tst.tc.zh -doc ../doc.txt -output zh-en_models/BASELINE_IWSLT2015/seed_1/test_all.txt -translate_part all -batch_size 1000 -gpu 1

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
