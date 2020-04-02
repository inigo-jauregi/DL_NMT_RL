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

-valid_src [source_dev_file] -valid_tgt [target_dev_file] -valid_doc [doc_dev_file]
-save_data [out_file]
```

- **train.src**: Sentences of the documents in the source language (one sentence per line).
- **train.tgt**: Sentences of the documents in the target language (one sentence per line).
- **train.doc**: Document delimiter file.

(same for validation)

For all the available arguments that allow different preprocessing options check the
*full_source/onmt/opt.py* file.

### Training

### Inference

### Evaluation

## References