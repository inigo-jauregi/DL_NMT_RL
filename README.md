# DL_NMT_RL
Document-Level Neural Machine Translation with Reinforcement Learning, using discourse rewards.

Code used for the experiments in the paper titled: _Leveraging Discourse Rewards for Document-Level
Neural Machine Translation_.

This project has been developed by modifying the the source code released by Miculicich et al. (2018), which at the
same time has been based on OpenNMT-py (Klein et al., 2017)

## Requirements

The code requires the following packages:


```
nltk==3.4.5
scikit-learn==0.22.1
torch==1.1.0
torchtext==0.5.0
tqdm==4.42.1
bert-score==0.3.1
```

They can be installed running the following command with pip:

```python
pip install -r requirements.txt
```

## Quick tutorial for model training and inference

### Preprocessing

First, the training and validation data files need to be preprocessed for model training.
The file *scripts/preprocessing.sh* contains the necessary pre-processing steps, which include
tokenization and true-casign.

Then, we run the following command:

```python
python preprocess.py -train_src [source_file]
                     -train_tgt [target_file]
                     -train_doc [doc_file]
                     -valid_src [source_dev_file]
                     -valid_tgt [target_dev_file]
                     -valid_doc [doc_dev_file]
                     -save_data [out_file]
```

- **train.src**: Sentences of the documents in the source language (one sentence per line).
- **train.tgt**: Sentences of the documents in the target language (one sentence per line).
- **train.doc**: Document delimiter file.

(same for validation)

For all the available arguments that allow different preprocessing options check the
*full_source/onmt/opt.py* file.

### Training

These are the commands to reproduce the main models of the paper.discourse

**Sentence-level NMT**:

```python
python train.py -data [preprocessed_data]
                -save_model [model_path]
                -encoder_type transformer
                -decoder_type transformer
                -enc_layers 6
                -dec_layers 6
                -label_smoothing 0.1
                -src_word_vec_size 512
                -tgt_word_vec_size 512
                -rnn_size 512
                -position_encoding
                -dropout 0.1
                -batch_size 4096
                -start_decay_at 20
                -report_every 500
                -epochs 20
                -gpuid 0
                -max_generator_batches 16
                -batch_type tokens
                -normalization tokens
                -accum_count 4
                -optim adam
                -adam_beta2 0.998
                -decay_method noam
                -warmup_steps 8000
                -learning_rate 2
                -max_grad_norm 0
                -param_init 0
                -param_init_glorot
                -train_part sentences
                -seed 1
```

**HAN join**:

```python
python train.py -data [preprocessed_data]
                -save_model [model_path]
                -encoder_type transformer
                -decoder_type transformer
                -enc_layers 6
                -dec_layers 6
                -label_smoothing 0.1
                -src_word_vec_size 512
                -tgt_word_vec_size 512
                -rnn_size 512
                -position_encoding
                -dropout 0.1
                -batch_size 1024
                -start_decay_at 8
                -report_every 500
                -epochs 10
                -max_generator_batches 16
                -batch_type tokens
                -normalization tokens
                -accum_count 4
                -optim adam
                -adam_beta2 0.998
                -decay_method noam
                -warmup_steps 8000
                -learning_rate 0.2
                -max_grad_norm 0
                -param_init 0
                -param_init_glorot
                -gpuid 0
                -seed 1
                -train_part all
                -context_size 3
                -train_from [pretrained_sentence_level_model]
```

**RISK(1.0) (with BLEUdoc, LCdoc and COH doc as rewards)**:

```python
python train.py -data [preprocessed_data]
                -save_model [model_path]
                -encoder_type transformer
                -decoder_type transformer
                -enc_layers 6
                -dec_layers 6
                -label_smoothing 0.1
                -src_word_vec_size 512
                -tgt_word_vec_size 512
                -rnn_size 512
                -position_encoding
                -dropout 0.1
                -batch_size 15
                -start_decay_at 2
                -report_every 500
                -max_generator_batches 32
                -accum_count 4
                -optim adam
                -adam_beta2 0.998
                -decay_method noam
                -warmup_steps 8000
                -learning_rate 0.2
                -max_grad_norm 0
                -param_init 0
                -param_init_glorot
                -train_part all
                -context_type HAN_join
                -context_size 3
                -seed 0
                -train_validate True
                -RISK_ratio 1.0
                -beam_size 2
                -n_best 2
                -train_from [pretrained_document_level_model]
                -gpuid 0
                -doc_level_reward True
                -doc_COH True
                -doc_LC True
                -doc_bleu True
```

NOTE: The *training.py* python script requires having the LSA model (Wiki-6) downloaded
(Stefanescu et al., 2014), and saved in the *scripts/coherence_model* folder.

### Inference

This is the command to run the inference using a particular trained model:

```python
python translate.py -model [trained_model_path]
                    -src [test_src]
                    -doc [test_doc]
                    -output [predictions_file]
                    -translate_part all
                    -batch_size 1000
                    -gpu 0

```

### Evaluation

**BLEU**:

```python
perl mosesdecoder/scripts/generic/multi-bleu.perl [ref_file]  < [predictions_file] > [output_results_file]
```

**LC**:

```python
python scripts/LC_RC.py 1 [predicted_file] [test_doc] > [output_results]
```

**COH**:

```python
python scripts/compute_coherence.py 1 [predicted_file] [test_doc] > [output_results]
```

**F-bert**:

```python
bert-score -r [ref_file] -c [predicted_file] --lang en -m bert-base-uncased -v > [output_results_file]
```

## References

Guillaume Klein, Yoon Kim, Yuntian Deng, Jean Senellart, and AlexanderMRush. 2017. Opennmt: Open-source
toolkit for neural machine translation. arXiv preprint arXiv:1701.02810.

Lesly Miculicich, Dhananjay Ram, Nikolaos Pappas, and James Henderson. 2018. Document-level neural machine
translation with hierarchical attention networks. In Proceedings of the Conference on Empirical Methods
in Natural Language Processing (EMNLP).

Dan Stefanescu, Rajendra Banjade, and Vasile Rus. 2014. Latent semantic analysis models on wikipedia and tasa.
In Language resources evaluation conference (LREC).