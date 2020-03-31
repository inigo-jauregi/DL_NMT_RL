### Copyright (c) 2018 Idiap Research Institute, http://www.idiap.ch/
### Modified by Lesly Miculicich <lmiculicich@idiap.ch>
# 
# This file is part of HAN-NMT.
# 
# HAN-NMT is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 3 as
# published by the Free Software Foundation.
# 
# HAN-NMT is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with HAN-NMT. If not, see <http://www.gnu.org/licenses/>.

"""
This file handles the details of the loss function during training.

This includes: LossComputeBase and the standard NMTLossCompute, and
               sharded loss compute stuff.
"""
from __future__ import division
import torch
import torch.nn as nn
from torch.autograd import Variable

import onmt
import onmt.io
from nltk.translate.bleu_score import sentence_bleu
from nltk.corpus import wordnet as wn
import nltk
from nltk.corpus import stopwords
import codecs, string


ps = nltk.stem.PorterStemmer()
stopW = stopWords = set(stopwords.words('english'))
remove_punctuation_map = dict((ord(char), None) for char in string.punctuation)

from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

#Load LSA model
print ('Loading LSA model...')
vect_dic = {}
with codecs.open('../scripts/coherence_model/Wiki_6/voc', encoding='utf-8') as f:
    words = [l.strip() for l in f]
with codecs.open('../scripts/coherence_model/Wiki_6/lsaModel', encoding='utf-8') as f:
    vec_list = []
    for l in f:
        l=l.split()
        # l=[float(n) for n in l]
        l=np.asarray(l,dtype=float)
        vec_list.append(l)

# Create dictionary
for i in range(len(words)):
    # print (words[i])
    vect_dic[words[i]] = vec_list[i]

class LossComputeBase(nn.Module):
    """
    Class for managing efficient loss computation. Handles
    sharding next step predictions and accumulating mutiple
    loss computations


    Users can implement their own loss computation strategy by making
    subclass of this one.  Users need to implement the _compute_loss()
    and make_shard_state() methods.

    Args:
        generator (:obj:`nn.Module`) :
             module that maps the output of the decoder to a
             distribution over the target vocabulary.
        tgt_vocab (:obj:`Vocab`) :
             torchtext vocab object representing the target output
        normalzation (str): normalize by "sents" or "tokens"
    """
    def __init__(self, generator, tgt_vocab):
        super(LossComputeBase, self).__init__()
        self.generator = generator
        self.tgt_vocab = tgt_vocab
        self.padding_idx = tgt_vocab.stoi[onmt.io.PAD_WORD]

    def _make_shard_state(self, batch, output, range_, attns=None):
        """
        Make shard state dictionary for shards() to return iterable
        shards for efficient loss computation. Subclass must define
        this method to match its own _compute_loss() interface.
        Args:
            batch: the current batch.
            output: the predict output from the model.
            range_: the range of examples for computing, the whole
                    batch or a trunc of it?
            attns: the attns dictionary returned from the model.
        """
        return NotImplementedError

    # def _compute_loss(self, batch, output, target, **kwargs):
    #     """
    #     Compute the loss. Subclass must define this method.
    #
    #     Args:
    #
    #         batch: the current batch.
    #         output: the predict output from the model.
    #         target: the validate target to compare output with.
    #         **kwargs(optional): additional info for computing loss.
    #     """
    #     return NotImplementedError

    def monolithic_compute_loss(self, batch, output, attns):
        """
        Compute the forward loss for the batch.

        Args:
          batch (batch): batch of labeled examples
          output (:obj:`FloatTensor`):
              output of decoder model `[tgt_len x batch x hidden]`
          attns (dict of :obj:`FloatTensor`) :
              dictionary of attention distributions
              `[tgt_len x batch x src_len]`
        Returns:
            :obj:`onmt.Statistics`: loss statistics
        """
        range_ = (0, batch.tgt.size(0))
        shard_state = self._make_shard_state(batch, output, range_, attns)
        _, batch_stats = self._compute_loss(batch, **shard_state)

        return batch_stats

    def sharded_compute_loss(self, batch, output=None, attns=None,
                             cur_trunc=None, trunc_size=None, shard_size=None,
                             normalization=None,ret=None, doc_index=None):
        """Compute the forward loss and backpropagate.  Computation is done
        with shards and optionally truncation for memory efficiency.

        Also supports truncated BPTT for long sequences by taking a
        range in the decoder output sequence to back propagate in.
        Range is from `(cur_trunc, cur_trunc + trunc_size)`.

        Note sharding is an exact efficiency trick to relieve memory
        required for the generation buffers. Truncation is an
        approximate efficiency trick to relieve the memory required
        in the RNN buffers.

        Args:
          batch (batch) : batch of labeled examples
          output (:obj:`FloatTensor`) :
              output of decoder model `[tgt_len x batch x hidden]`
          attns (dict) : dictionary of attention distributions
              `[tgt_len x batch x src_len]`
          cur_trunc (int) : starting position of truncation window
          trunc_size (int) : length of truncation window
          shard_size (int) : maximum number of examples in a shard
          normalization (int) : Loss is divided by this number

        Returns:
            :obj:`onmt.Statistics`: validation loss statistics

        """
        batch_stats = onmt.Statistics()
        if ret:
            loss, stats = self._compute_loss(batch,ret,doc_index)
            # print (loss.requires_grad)
            #Normalization over sentences
            loss.div(float(batch.tgt.size()[1])).backward(retain_graph=True)
            batch_stats.update(stats)

        else:

            range_ = (cur_trunc, cur_trunc + trunc_size)
            shard_state = self._make_shard_state(batch, output, range_, attns)

            for shard in shards(shard_state, shard_size):
                loss, stats = self._compute_loss(batch, **shard)
                loss.div(float(normalization)).backward(retain_graph=True)
                batch_stats.update(stats)

        #Maybe here we can detach the loss
        loss.detach()

        return batch_stats

    def _stats(self, loss, scores, target,n_sentences=0):
        """
        Args:
            loss (:obj:`FloatTensor`): the loss computed by the loss criterion.
            scores (:obj:`FloatTensor`): a score for each possible output
            target (:obj:`FloatTensor`): true targets

        Returns:
            :obj:`Statistics` : statistics for this batch.
        """
        pred = scores.max(1)[1]
        non_padding = target.ne(self.padding_idx)
        num_correct = pred.eq(target) \
                          .masked_select(non_padding) \
                          .sum()
        return onmt.Statistics(loss.item(), non_padding.sum(), num_correct,n_sentences=n_sentences)

    def _bottle(self, v):
        return v.view(-1, v.size(2))

    def _unbottle(self, v, batch_size):
        return v.view(-1, batch_size, v.size(1))


class NMTLossCompute(LossComputeBase):
    """
    Standard NMT Loss Computation.
    """
    def __init__(self, generator, tgt_vocab, normalization="sents",
                 label_smoothing=0.0):
        super(NMTLossCompute, self).__init__(generator, tgt_vocab)
        assert (label_smoothing >= 0.0 and label_smoothing <= 1.0)
        if label_smoothing > 0:
            # When label smoothing is turned on,
            # KL-divergence between q_{smoothed ground truth prob.}(w)
            # and p_{prob. computed by model}(w) is minimized.
            # If label smoothing value is set to zero, the loss
            # is equivalent to NLLLoss or CrossEntropyLoss.
            # All non-true labels are uniformly set to low-confidence.
            self.criterion = nn.KLDivLoss(size_average=False)
            one_hot = torch.randn(1, len(tgt_vocab))
            one_hot.fill_(label_smoothing / (len(tgt_vocab) - 2))
            one_hot[0][self.padding_idx] = 0
            self.register_buffer('one_hot', one_hot)
        else:
            weight = torch.ones(len(tgt_vocab))
            weight[self.padding_idx] = 0
            self.criterion = nn.NLLLoss(weight, size_average=False)
        self.confidence = 1.0 - label_smoothing

    def _make_shard_state(self, batch, output, range_, attns=None):
        return {
            "output": output,
            "target": batch.tgt[range_[0] + 1: range_[1]],
        }

    def _compute_loss(self, batch, output, target):
        scores = self.generator(self._bottle(output))

        gtruth = target.view(-1)
        if self.confidence < 1:
            tdata = gtruth.data
            mask = torch.nonzero(tdata.eq(self.padding_idx)).squeeze()
            log_likelihood = torch.gather(scores.data, 1, tdata.unsqueeze(1))
            tmp_ = self.one_hot.repeat(gtruth.size(0), 1)
            tmp_.scatter_(1, tdata.unsqueeze(1), self.confidence)
            if mask.nelement() > 0:
                log_likelihood.index_fill_(0, mask, 0.0)
                tmp_.index_fill_(0, mask, 0.0)
            gtruth = Variable(tmp_, requires_grad=False)
        loss = self.criterion(scores, gtruth)
        if self.confidence < 1:
            # Default: report smoothed ppl.
            # loss_data = -log_likelihood.sum(0)
            loss_data = loss.data.clone()
        else:
            loss_data = loss.data.clone()

        stats = self._stats(loss_data, scores.data, target.view(-1).data, n_sentences=target.size()[1])
        return loss, stats

class REINFORCELossCompute(LossComputeBase):
    """
    Standard REINFORCE Loss Computation.
    """
    def __init__(self, generator, tgt_vocab, n_best, doc_level=False, bleu_doc=False, LC_doc=False, COH_doc=False, bleu_sen=False):
        super(REINFORCELossCompute, self).__init__(generator, tgt_vocab)

        self.tgt_vocab = tgt_vocab
        weight = torch.ones(len(tgt_vocab))
        weight[self.padding_idx] = 0
        self.criterion = nn.NLLLoss(weight, size_average=False)
        self.n_best = n_best
        self.bleu_doc=bleu_doc
        self.doc_level = doc_level
        self.LC_doc = LC_doc
        self.COH_doc = COH_doc
        self.bleu_sen = bleu_sen


    def _make_shard_state(self, batch, output, range_, attns=None):
        return {
            "output": output,
            "target": batch.tgt[range_[0] + 1: range_[1]],
        }

    def _compute_loss(self, batch, ret, doc_index):

        #RET obtain the hypothesis
        top_hyp = ret['predictions']
        top_probabilities = ret['out_prob']
        #AND ALSO THE REFERENCE
        tgt = batch.tgt
        batch_num = tgt.size()[1]

        # print ('top_hyp ',len(top_hyp))
        # print ('top_probabilities ', len(top_probabilities))
        # print ('tgt ',tgt.size())
        if self.doc_level:
            sentences_pred, prob_per_word = self.words_probs_from_preds(top_hyp, top_probabilities, doc_index=doc_index,batch_num=batch_num)
            sentences_gt = self.obtain_words_from_tgt(tgt, doc_index=doc_index)
            # print (doc_index)
            # print (prob_per_word.size())
            # print (len(sentences_pred[0][0]))
            # print (len(sentences_pred[0][1]))
            # print (len(sentences_gt[0]))

            dl_rewards = 0

            if self.bleu_doc:
                bleu_doc_scores = self.BLEU_score(sentences_pred,sentences_gt)
                dl_rewards += bleu_doc_scores
            if self.LC_doc:
                LC_scores = self.LC_scores(sentences_pred)
                dl_rewards += LC_scores
            if self.COH_doc:
                coher_scores = self.coher_scores(sentences_pred)
                dl_rewards += coher_scores


            loss_doc = self.RISK_loss(prob_per_word, dl_rewards)
            loss_doc = loss_doc.sum(0)

        if self.bleu_sen:
            sentences_pred, prob_per_word = self.words_probs_from_preds(top_hyp,top_probabilities)
            # print (prob_per_word.size())
            # print ('i am printing')
            sentences_gt = self.obtain_words_from_tgt(tgt)
            sl_rewards = self.BLEU_score(sentences_pred, sentences_gt)
            loss_sen = self.RISK_loss(prob_per_word, sl_rewards)
            loss_sen = loss_sen.sum(0)

        if self.doc_level and self.bleu_sen:
            loss = loss_doc + loss_sen
        elif self.doc_level:
            loss = loss_doc
        else:
            loss = loss_sen

        # #NEED TO COMPUTE THE LOSS
        # loss = bleu_scores*prob_per_word
        # loss =loss.sum(0)
        loss = - loss  # I set negative because we don't have baseline for the moent
        # print ('The loss: ', loss)

        loss_data = loss.data.clone()

        #loss.detach()

        stats = onmt.Statistics(loss_data.item(), n_sentences=tgt.size()[1])
        return loss, stats

    def words_probs_from_preds(self, hyps, probs, doc_index=None, batch_num=None):
        # print (doc_index)
        if type(doc_index) is list:
            if len(doc_index)>=1:
                new_doc_lines = [el for el in doc_index]
            else:
                new_doc_lines = [0]

            # Treat separately each hypothesis
            sen_best = []
            prob_best = []
            for i in range(self.n_best):
                # Iterate over the sentences in the batch
                # But stop if a doc_index is added
                sentences_batch_in_beam = []
                probs_batch_in_beam = []
                count_doc_lines = 0
                current_doc_break= new_doc_lines[count_doc_lines]
                words = []
                probability_sen = []
                for j in range(batch_num):
                    # Then go thorugh the probabilities of the word.
                    # Compute a single document probability??? I would say yes.
                    if j == current_doc_break and current_doc_break!=0:
                        sentences_batch_in_beam.append(words)
                        # print (len(probability_sen))
                        # print ('maima')
                        prob_sentence = torch.exp(torch.stack(probability_sen, dim=0).sum(dim=0) / len(probability_sen))
                        # print ('maima2')
                        probs_batch_in_beam.append(prob_sentence)
                        words = []
                        probability_sen = []
                        count_doc_lines+=1
                        if len(new_doc_lines) > count_doc_lines:
                            current_doc_break = new_doc_lines[count_doc_lines]
                    elif j==current_doc_break and current_doc_break==0:
                        words = []
                        probability_sen = []
                        count_doc_lines += 1
                        if len(new_doc_lines) > count_doc_lines:
                            current_doc_break = new_doc_lines[count_doc_lines]
                    # Iterate over the sentenc
                    for k in range(len(hyps[j][i])):
                        word = self.tgt_vocab.itos[hyps[j][i][k]]
                        words.append(word)
                        prob = probs[j][i][k][0, hyps[j][i][k]]
                        probability_sen.append(prob)
                # print(len(probability_sen))
                # print ('ketxua')
                prob_sentence = torch.exp(torch.stack(probability_sen, dim=0).sum(dim=0) / len(probability_sen))
                # print ('ketxua2')
                sentences_batch_in_beam.append(words)
                probs_batch_in_beam.append(prob_sentence)

                sen_best.append(sentences_batch_in_beam)
                probs_batch_in_beam = torch.stack(probs_batch_in_beam,dim=0)
                prob_best.append(probs_batch_in_beam)

            # Print
            probs_tensor = torch.stack(prob_best,dim=0).t()
            # print (probs_tensor.size())
            # print (probs_tensor)

            # Change sentences
            sentences = []
            for i in range(len(sen_best[0])):
                both_beams = []
                for j in range(len(sen_best)):
                    both_beams.append(sen_best[j][i])
                sentences.append(both_beams)

            # print(len(sentences))
            # print(len(sentences[0]))

            return sentences, probs_tensor

        else:
            sentences = []
            probs_per_word = []
            for i in range(len(hyps)):
                sentences_in_beam = []
                probs_in_beam =[]
                for j in range(len(hyps[i])):
                    words = []
                    probability_sen = []
                    for k in range(len(hyps[i][j])):
                        word = self.tgt_vocab.itos[hyps[i][j][k]]
                        words.append(word)
                        prob = probs[i][j][k][0,hyps[i][j][k]]
                        # print (prob)
                        # print (prob.requires_grad)
                        #print (prob)
                        # prob = torch.log(prob)  THERE IS NO NEED FOR BECAUSE WE HAVE USED LOG-SOFTMAX BEFORE
                        probability_sen.append(prob)

                    sentences_in_beam.append(words)
                    # print (probability_sen[0])
                    # print (probability_sen)
                    # print ('haifa')
                    prob_sentence = torch.exp(torch.stack(probability_sen,dim=0).sum(dim=0)/len(probability_sen))
                    # print ('haifa 2')
                    # print (prob_sentence)

                    # prob_sentence[prob_sentence!=prob_sentence]=0
                    # print (type(prob_sentence))
                    # print (prob_sentence)
                    probs_in_beam.append(prob_sentence)

                sentences.append(sentences_in_beam)
                probs_in_beam = torch.stack(probs_in_beam,dim=0)
                probs_per_word.append(probs_in_beam)

            probs_tensor = torch.stack(probs_per_word,dim=0)



            return sentences, probs_tensor.squeeze()

    def obtain_words_from_tgt(self,tgt, doc_index=None):
        sentence_max_length, batch_size = tgt.size()
        padID = self.tgt_vocab.stoi['<blank>']

        if type(doc_index) is list:
            if len(doc_index)>=1:
                new_doc_lines = [el for el in doc_index]
            else:
                new_doc_lines = [0]

            sentences = []

            count_doc_lines = 0
            current_doc_break = new_doc_lines[count_doc_lines]
            words = []
            for i in range(batch_size):
                # Then go thorugh the probabilities of the word.
                # Compute a single document probability??? I would say yes.
                if i == current_doc_break and current_doc_break != 0:
                    sentences.append(words)
                    words = []
                    count_doc_lines += 1
                    if len(new_doc_lines) > count_doc_lines:
                        current_doc_break = new_doc_lines[count_doc_lines]
                elif i == current_doc_break and current_doc_break == 0:
                    words = []
                    count_doc_lines += 1
                    if len(new_doc_lines) > count_doc_lines:
                        current_doc_break = new_doc_lines[count_doc_lines]
                # words = []
                for j in range(sentence_max_length):
                    if tgt[j, i] == padID:
                        break
                    words.append(self.tgt_vocab.itos[tgt[j, i]])
            sentences.append(words)

        else:

            sentences = []
            for i in range(batch_size):
                words = []
                for j in range(sentence_max_length):
                    if tgt[j,i] == padID:
                        break
                    words.append(self.tgt_vocab.itos[tgt[j,i]])

                sentences.append(words)

        # print (len(sentences))
        return sentences

    def BLEU_score(self,predictions,ground_truth):

        batch_size = len(predictions)
        n_best = len(predictions[0])
        # print (n_best)
        # print (predictions[0][0])
        # print (predictions[0][1])
        bleu_scores = []
        for i in range(batch_size):
            bs_nBest = []
            for j in range(n_best):
                pred = predictions[i][j]
                # print (pred)
                gt = [ground_truth[i]]
                # print (gt)
                BLEU = sentence_bleu(gt,pred)*100
                bs_nBest.append(BLEU)

            bleu_scores.append(bs_nBest)

        bleu_scores = torch.FloatTensor(bleu_scores).cuda() #Cuda line

        # print (bleu_scores.size())

        return bleu_scores.squeeze()

    def coher_scores(self,predictions):
        batch_size = len(predictions)
        n_best = len(predictions[0])

        # print ('Batch_size: ',batch_size)

        coher_scores = []
        for i in range(batch_size):
            bs_nBest = []
            for j in range(n_best):
                pred = predictions[i][j]
                coher = self.get_coher_score(pred)*100
                bs_nBest.append(coher)

            coher_scores.append(bs_nBest)
        # print (coher_scores)
        coher_scores = torch.FloatTensor(coher_scores).cuda()  # Cuda line

        return coher_scores.squeeze()

    def get_coher_score(self,sen):
        # Convert input into sentences
        sen = " ".join(sen)
        sen = sen.split('</s>')

        d = []
        # print ('Num sen: ',len(sen))
        for l in sen:
            d.append(self.get_sen_embedding(l))

        avg_coherence = self.compute_coherence(d)

        return avg_coherence

    def compute_coherence(self,document_vectors):

        n_sentences = len(document_vectors)
        scores = []
        for i in range(n_sentences - 1):
            # print (document_vectors[i])
            # print (document_vectors[i+1])
            cos_sim = cosine_similarity([document_vectors[i]], [document_vectors[i + 1]])
            scores.append(cos_sim)

        # Average similarity
        if len(scores)>0:
            avg_cos_sim = sum(scores) / float(len(scores))
        else:
            avg_cos_sim = np.zeros((1,1))
        return avg_cos_sim


    def get_sen_embedding(self,sentence):

        # Split
        sentence = sentence.replace('\n', '')
        tokenized_sen = sentence.split()
        sen_vec = np.zeros(300)
        count_words = 0
        for word in tokenized_sen:
            theWord = word.lower()
            if theWord in vect_dic:
                count_words += 1
                word_vec = vect_dic[theWord]
                sen_vec += word_vec

        if count_words != 0:
            sen_vec = sen_vec / count_words

        return sen_vec

    def LC_scores(self,predictions):
        batch_size = len(predictions)
        n_best = len(predictions[0])

        lc_scores = []
        for i in range(batch_size):
            bs_nBest = []
            for j in range(n_best):
                pred = predictions[i][j]
                LC = self.get_score(pred)*100
                bs_nBest.append(LC)

            lc_scores.append(bs_nBest)

        lc_scores = torch.FloatTensor(lc_scores).cuda()  # Cuda line

        return lc_scores.squeeze()

    def penn_to_wn(self,tag):
        """ Convert between a Penn Treebank tag to a simplified Wordnet tag """
        if tag.startswith('N'):
            return 'n'

        if tag.startswith('V'):
            return 'v'

        if tag.startswith('J'):
            return 'a'

        if tag.startswith('R'):
            return 'r'

        return None

    def get_score(self,f):

        swords = dict()
        CW = 0

        f = " ".join(f)
        f = f.split('</s>')

        for l in f:
            l = l.strip().split()
            pos = nltk.pos_tag(l)
            for w, p in pos:
                if w not in stopW:
                    p = self.penn_to_wn(p)
                    s = wn.synsets(w, p)
                    if w not in string.punctuation: CW += 1
                    if len(s) > 0:
                        sw = ps.stem(w)
                        wid = w + "_" + (p if p else "")
                        if wid in swords:
                            swords[wid][1] += 1
                        else:
                            swords[wid] = [s[0], 1, p, sw, False]
        RC = 0
        LC1 = 0
        LC2 = 0
        list_words = list(swords)
        for i, sn in enumerate(list_words):
            if swords[sn][1] > 1 and not swords[sn][4]:
                RC += swords[sn][1]
                swords[sn][4] = True

            for sn2 in list_words[i + 1:]:
                if not swords[sn][4] or not swords[sn2][4]:
                    if swords[sn][3] == swords[sn2][3] and swords[sn][2] == swords[sn2][2]:
                        if not swords[sn][4]:
                            RC += swords[sn][1]
                            swords[sn][4] = True
                        if not swords[sn2][4]:
                            RC += swords[sn2][1]
                            swords[sn2][4] = True
                    else:
                        fs1, fs2 = swords[sn][0], swords[sn2][0]

                        if fs1.path_similarity(fs2) == 1:
                            if not swords[sn][4]:
                                LC1 += swords[sn][1]
                                swords[sn][4] = True
                            if not swords[sn2][4]:
                                LC1 += swords[sn2][1]
                                swords[sn2][4] = True

                        elif swords[sn][2] == swords[sn2][2]:
                            min_d = fs1.min_depth()
                            min_d2 = fs2.min_depth()
                            s3 = fs1.lowest_common_hypernyms(fs2)
                            if len(s3) > 0:
                                s3 = s3[0]
                                min_d3 = s3.min_depth()
                                sim = 2 * min_d3 / (min_d + min_d2)
                                if sim >= 0.96:
                                    if not swords[sn][4]:
                                        LC2 += swords[sn][1]
                                        swords[sn][4] = True
                                    if not swords[sn2][4]:
                                        LC2 += swords[sn2][1]
                                        swords[sn2][4] = True

        if float(CW) == 0.0:
            return 0.0

        return (RC + LC1 + LC2) / float(CW)

    def RISK_loss(self,probs, bleu):


        # print (probs[0])
        # print (probs)
        sum_probs = probs.sum(1).unsqueeze(1)
        # print (sum_probs.size())
        sum_probs2 = sum_probs.clone()
        #print (sum_probs2.size())
        sum_probs = torch.cat((sum_probs,sum_probs2),1)
        #print (sum_probs.size())
        probabilities = probs/sum_probs
        #print (probabilities[0])
        prob_with_bleu = bleu * probabilities

        loss_per_sentence = prob_with_bleu.sum(1)

        # loss_per_sentence=loss_per_sentence[loss_per_sentence==loss_per_sentence]

        # print (loss_per_sentence)

        return loss_per_sentence

def filter_shard_state(state, requires_grad=True, volatile=False):
    for k, v in state.items():
        if v is not None:
            if isinstance(v, Variable) and v.requires_grad:
                v = Variable(v.data, requires_grad=requires_grad,
                             volatile=volatile)
            yield k, v


def shards(state, shard_size, eval=False):
    """
    Args:
        state: A dictionary which corresponds to the output of
               *LossCompute._make_shard_state(). The values for
               those keys are Tensor-like or None.
        shard_size: The maximum size of the shards yielded by the model.
        eval: If True, only yield the state, nothing else.
              Otherwise, yield shards.

    Yields:
        Each yielded shard is a dict.

    Side effect:
        After the last shard, this function does back-propagation.
    """
    if eval:
        yield filter_shard_state(state, False, True)
    else:
        # non_none: the subdict of the state dictionary where the values
        # are not None.
        non_none = dict(filter_shard_state(state))

        # Now, the iteration:
        # state is a dictionary of sequences of tensor-like but we
        # want a sequence of dictionaries of tensors.
        # First, unzip the dictionary into a sequence of keys and a
        # sequence of tensor-like sequences.
        keys, values = zip(*((k, torch.split(v, shard_size))
                             for k, v in non_none.items()))

        # Now, yield a dictionary for each shard. The keys are always
        # the same. values is a sequence of length #keys where each
        # element is a sequence of length #shards. We want to iterate
        # over the shards, not over the keys: therefore, the values need
        # to be re-zipped by shard and then each shard can be paired
        # with the keys.
        for shard_tensors in zip(*values):
            yield dict(zip(keys, shard_tensors))

        # Assumed backprop'd
        variables = ((state[k], v.grad.data) for k, v in non_none.items()
                     if isinstance(v, Variable) and v.grad is not None)
        inputs, grads = zip(*variables)
        torch.autograd.backward(inputs, grads)
