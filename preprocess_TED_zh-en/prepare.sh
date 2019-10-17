#!/bin/bash
moses_scripts=../mosesdecoder/scripts

zh_segment_home=/idiap/home/lmiculicich/.cache/pip/wheels/ce/32/de/c2be1db5f30804bc7f146ff698c52963f8aa11ba5049811b0d
#kpu_preproc_dir=/fs/zisa0/bhaddow/code/preprocess/build/bin

max_len=200

export PYTHONPATH=$zh_segment_home

src=es
tgt=en
pair=$src-$tgt
data_path=../es-en_2013

# Tokenise the Spanish part
cat $data_path/corpus.$src | \
$moses_scripts/tokenizer/normalize-punctuation.perl -l $src | \
$moses_scripts/tokenizer/tokenizer.perl -a -l $src  \
> $data_path/corpus.tok.$src

# Tokenise the English part
cat $data_path/corpus.$tgt | \
$moses_scripts/tokenizer/normalize-punctuation.perl -l $tgt | \
$moses_scripts/tokenizer/tokenizer.perl -a -l $tgt  \
> $data_path/corpus.tok.$tgt

#Segment the Chinese part
#python -m jieba -d ' ' < $data_path/corpus.$src > $data_path/corpus.tok.$src 

#
###
#### Clean
#$moses_scripts/training/clean-corpus-n.perl corpus.tok $src $tgt corpus.clean 1 $max_len corpus.retained
###
#

#### Train truecaser and truecase
$moses_scripts/recaser/train-truecaser.perl -model $data_path/truecase-model.$src -corpus $data_path/corpus.tok.$src
$moses_scripts/recaser/truecase.perl < $data_path/corpus.tok.$src > $data_path/corpus.tc.$src -model $data_path/truecase-model.$src


$moses_scripts/recaser/train-truecaser.perl -model $data_path/truecase-model.$tgt -corpus $data_path/corpus.tok.$tgt
$moses_scripts/recaser/truecase.perl < $data_path/corpus.tok.$tgt > $data_path/corpus.tc.$tgt -model $data_path/truecase-model.$tgt

ln -s $data_path/corpus.tok.$src  $data_path/corpus.tc.$src
#
#  
# dev sets
for devset in dev2010 tst2010 tst2011 tst2012; do
  for lang  in $src $tgt; do
    if [ $lang = $tgt ]; then
      side="src"
      $moses_scripts/tokenizer/normalize-punctuation.perl -l $lang < $data_path/IWSLT14.TED.$devset.$tgt-$src.$lang | \
      $moses_scripts/tokenizer/tokenizer.perl -a -l $lang |  \
      $moses_scripts/recaser/truecase.perl -model $data_path/truecase-model.$lang \
      > $data_path/IWSLT14.TED.$devset.tc.$lang
    else
      side="ref"
      $moses_scripts/tokenizer/normalize-punctuation.perl -l $lang < $data_path/IWSLT14.TED.$devset.$tgt-$src.$lang | \
      $moses_scripts/tokenizer/tokenizer.perl -a -l $lang |  \
      $moses_scripts/recaser/truecase.perl -model $data_path/truecase-model.$lang \
      > $data_path/IWSLT14.TED.$devset.tc.$lang
      #python -m jieba -d ' '  < $data_path/IWSLT15.TED.$devset.$src-$tgt.$lang > $data_path/IWSLT15.TED.$devset.tc.$lang
      
    fi
    
  done

done

python ../full_source/preprocess.py -train_src $data_path/corpus.tc.es -train_tgt $data_path/corpus.tc.en -train_doc $data_path/corpus.doc -valid_src $data_path/IWSLT14.TED.dev2010.tc.es -valid_tgt $data_path/IWSLT14.TED.dev2010.tc.en -valid_doc $data_path/IWSLT14.TED.dev2010.en-es.doc -save_data ../IWSLT14.TED_ES_EN -src_vocab_size 30000 -tgt_vocab_size 30000 -src_seq_length 80 -tgt_seq_length 80

