
#BASELINE

#seed 1
#python train.py -data ../zh-en/IWSLT15.TED -save_model zh-en_models/BASELINE_IWSLT2015_newCode/seed_1/EPOCH -encoder_type transformer -decoder_type transformer -enc_layers 6 -dec_layers 6 -label_smoothing 0.1 -src_word_vec_size 512 -tgt_word_vec_size 512 -rnn_size 512 -position_encoding -dropout 0.1 -batch_size 4096 -start_decay_at 20 -report_every 500 -epochs 20 -gpuid 0 -max_generator_batches 16 -batch_type tokens -normalization tokens -accum_count 4 -optim adam -adam_beta2 0.998 -decay_method noam -warmup_steps 8000 -learning_rate 2 -max_grad_norm 0 -param_init 0 -param_init_glorot -train_part sentences -seed 1

#seed 2
#python train.py -data ../../HAN_NMT/zh-en/IWSLT15.TED -save_model zh-en_models/BASELINE_IWSLT2015/seed_2/EPOCH -encoder_type transformer -decoder_type transformer -enc_layers 6 -dec_layers 6 -label_smoothing 0.1 -src_word_vec_size 512 -tgt_word_vec_size 512 -rnn_size 512 -position_encoding -dropout 0.1 -batch_size 4096 -start_decay_at 20 -report_every 500 -epochs 20 -gpuid 0 -max_generator_batches 16 -batch_type tokens -normalization tokens -accum_count 4 -optim adam -adam_beta2 0.998 -decay_method noam -warmup_steps 8000 -learning_rate 2 -max_grad_norm 0 -param_init 0 -param_init_glorot -train_part sentences -seed 2

#seed 3
#python train.py -data ../../HAN_NMT/zh-en/IWSLT15.TED -save_model zh-en_models/BASELINE_IWSLT2015/seed_3/EPOCH -encoder_type transformer -decoder_type transformer -enc_layers 6 -dec_layers 6 -label_smoothing 0.1 -src_word_vec_size 512 -tgt_word_vec_size 512 -rnn_size 512 -position_encoding -dropout 0.1 -batch_size 4096 -start_decay_at 20 -report_every 500 -epochs 20 -gpuid 0 -max_generator_batches 16 -batch_type tokens -normalization tokens -accum_count 4 -optim adam -adam_beta2 0.998 -decay_method noam -warmup_steps 8000 -learning_rate 2 -max_grad_norm 0 -param_init 0 -param_init_glorot -train_part sentences -seed 3


#HAN_encoder and HAN_decoder
#We need the contextual

#seed 1
python train.py -data ../zh-en/IWSLT15.TED -save_model zh-en_models/HAN_join_newCode_plus_RL_0.33/seed_1/EPOCH -encoder_type transformer -decoder_type transformer -enc_layers 6 -dec_layers 6 -label_smoothing 0.1 -src_word_vec_size 512 -tgt_word_vec_size 512 -rnn_size 512 -position_encoding -dropout 0.1 -batch_size 15 -start_decay_at 2 -report_every 500 -epochs 2 -max_generator_batches 32 -accum_count 4 -optim adam -adam_beta2 0.998 -decay_method noam -warmup_steps 8000 -learning_rate 0.2 -max_grad_norm 0 -param_init 0 -param_init_glorot -train_part all -context_type HAN_join -context_size 3 -train_from zh-en_models/HAN_join_newCode/ppl/seed_1/EPOCH_acc_43.17_ppl_22.94_e1.pt -seed 1 -train_validate True -RISK_ratio 0.33 -beam_size 2 -n_best 2 -gpuid 0

#python train.py -data ../zh-en/IWSLT15.TED -save_model zh-en_models/HAN_join_newCode/acc/seed_1/EPOCH -encoder_type transformer -decoder_type transformer -enc_layers 6 -dec_layers 6 -label_smoothing 0.1 -src_word_vec_size 512 -tgt_word_vec_size 512 -rnn_size 512 -position_encoding -dropout 0.1 -batch_size 4096 -start_decay_at 2 -report_every 500 -epochs 2 -gpuid 0 -max_generator_batches 32 -batch_type tokens -normalization tokens -accum_count 4 -optim adam -adam_beta2 0.998 -decay_method noam -warmup_steps 8000 -learning_rate 2 -max_grad_norm 0 -param_init 0 -param_init_glorot -train_part all -context_type HAN_join -context_size 3 -train_from zh-en_models/BASELINE_IWSLT2015_newCode/seed_1/EPOCH_acc_42.66_ppl_23.58_e16.pt -seed 1

#seed 2
#python train.py -data ../../HAN_NMT/TRY -save_model zh-en_models/HAN_join/seed_2/EPOCH -encoder_type transformer -decoder_type transformer -enc_layers 6 -dec_layers 6 -label_smoothing 0.1 -src_word_vec_size 512 -tgt_word_vec_size 512 -rnn_size 512 -position_encoding -dropout 0.1 -batch_size 1024 -start_decay_at 2 -report_every 500 -epochs 3 -gpuid 0 -max_generator_batches 32 -batch_type tokens -normalization tokens -accum_count 4 -optim adam -adam_beta2 0.998 -decay_method noam -warmup_steps 8000 -learning_rate 2 -max_grad_norm 0 -param_init 0 -param_init_glorot -train_part all -context_type HAN_join -context_size 3 -train_from zh-en_models/BASELINE/seed_2/chosen_EPOCH_acc_41.69_ppl_25.11_e14.pt -seed 2

#seed 3
#python train.py -data ../../HAN_NMT/TRY -save_model zh-en_models/HAN_join/seed_3/EPOCH -encoder_type transformer -decoder_type transformer -enc_layers 6 -dec_layers 6 -label_smoothing 0.1 -src_word_vec_size 512 -tgt_word_vec_size 512 -rnn_size 512 -position_encoding -dropout 0.1 -batch_size 1024 -start_decay_at 2 -report_every 500 -epochs 3 -gpuid 0 -max_generator_batches 32 -batch_type tokens -normalization tokens -accum_count 4 -optim adam -adam_beta2 0.998 -decay_method noam -warmup_steps 8000 -learning_rate 2 -max_grad_norm 0 -param_init 0 -param_init_glorot -train_part all -context_type HAN_join -context_size 3 -train_from zh-en_models/BASELINE/seed_3/chosen_EPOCH_acc_41.92_ppl_24.82_e13.pt -seed 3
