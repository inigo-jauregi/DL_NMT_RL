

python translate.py -model zh-en_models/HAN_join_plus_RL_BLEU_LC_coher/seed_2/EPOCH_acc_43.26_ppl_23.14_e1_num3.pt -src ../zh-en/text_2011_2013.zh -doc ../zh-en/doc_2011_2013.txt -output zh-en_models/HAN_join_newCode_plus_DL_RL/seed_2/test_2011_2013_num3.txt -translate_part all -batch_size 1000 -gpu 0

python translate.py -model zh-en_models/HAN_join_newCode_plus_DL_RL/seed_3/EPOCH_acc_43.31_ppl_22.82_e1_num5.pt -src ../zh-en/text_2011_2013.zh -doc ../zh-en/doc_2011_2013.txt -output zh-en_models/HAN_join_newCode_plus_DL_RL/seed_3/test_2011_2013_num5.txt -translate_part all -batch_size 1000 -gpu 0




