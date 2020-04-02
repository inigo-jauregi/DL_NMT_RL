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
python preprocess.py -train_src [source_file] -train_tgt [target_file] -train_doc [doc_file]
-valid_src [source_dev_file] -valid_tgt [target_dev_file] -valid_doc [doc_dev_file] -save_data [out_file]
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
python train.py -data [preprocessed_data] \\
                -save_model [model_path] \\
                -encoder_type transformer \\
                -decoder_type transformer \\
                -enc_layers 6 -dec_layers 6 -label_smoothing 0.1 -src_word_vec_size 512 -tgt_word_vec_size 512 -rnn_size 512 -position_encoding -dropout 0.1 -batch_size 4096 -start_decay_at 20 -report_every 500 -epochs 20 -gpuid 0 -max_generator_batches 16 -batch_type tokens -normalization tokens -accum_count 4 -optim adam -adam_beta2 0.998 -decay_method noam -warmup_steps 8000 -learning_rate 2 -max_grad_norm 0 -param_init 0 -param_init_glorot -train_part sentences -seed 1
```

**HAN join**:

```python
```

**RISK(1.0)**:

```python
```

### Inference

### Evaluation

## References