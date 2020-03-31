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

from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F


from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from torch.distributions import Categorical

import onmt
from onmt.Utils import aeq

#My added libraries
import numpy as np

def rnn_factory(rnn_type, **kwargs):
	# Use pytorch version when available.
	no_pack_padded_seq = False
	if rnn_type == "SRU":
		# SRU doesn't support PackedSequence.
		no_pack_padded_seq = True
		rnn = onmt.modules.SRU(**kwargs)
	else:
		rnn = getattr(nn, rnn_type)(**kwargs)
	return rnn, no_pack_padded_seq


class EncoderBase(nn.Module):
	"""
	Base encoder class. Specifies the interface used by different encoder types
	and required by :obj:`onmt.Models.NMTModel`.

	.. mermaid::

	   graph BT
		  A[Input]
		  subgraph RNN
			C[Pos 1]
			D[Pos 2]
			E[Pos N]
		  end
		  F[Memory_Bank]
		  G[Final]
		  A-->C
		  A-->D
		  A-->E
		  C-->F
		  D-->F
		  E-->F
		  E-->G
	"""
	def _check_args(self, input, lengths=None, hidden=None):
		s_len, n_batch, n_feats = input.size()
		if lengths is not None:
			n_batch_, = lengths.size()
			aeq(n_batch, n_batch_)

	def forward(self, src, lengths=None, encoder_state=None):
		"""
		Args:
			src (:obj:`LongTensor`):
			   padded sequences of sparse indices `[src_len x batch x nfeat]`
			lengths (:obj:`LongTensor`): length of each sequence `[batch]`
			encoder_state (rnn-class specific):
			   initial encoder_state state.

		Returns:
			(tuple of :obj:`FloatTensor`, :obj:`FloatTensor`):
				* final encoder state, used to initialize decoder
				* memory bank for attention, `[src_len x batch x hidden]`
		"""
		raise NotImplementedError


class MeanEncoder(EncoderBase):
	"""A trivial non-recurrent encoder. Simply applies mean pooling.

	Args:
	   num_layers (int): number of replicated layers
	   embeddings (:obj:`onmt.modules.Embeddings`): embedding module to use
	"""
	def __init__(self, num_layers, embeddings):
		super(MeanEncoder, self).__init__()
		self.num_layers = num_layers
		self.embeddings = embeddings

	def forward(self, src, lengths=None, encoder_state=None):
		"See :obj:`EncoderBase.forward()`"
		self._check_args(src, lengths, encoder_state)

		emb = self.embeddings(src)
		s_len, batch, emb_dim = emb.size()
		mean = emb.mean(0).expand(self.num_layers, batch, emb_dim)
		memory_bank = emb
		encoder_final = (mean, mean)
		return encoder_final, memory_bank


class RNNEncoder(EncoderBase):
	""" A generic recurrent neural network encoder.

	Args:
	   rnn_type (:obj:`str`):
		  style of recurrent unit to use, one of [RNN, LSTM, GRU, SRU]
	   bidirectional (bool) : use a bidirectional RNN
	   num_layers (int) : number of stacked layers
	   hidden_size (int) : hidden size of each layer
	   dropout (float) : dropout value for :obj:`nn.Dropout`
	   embeddings (:obj:`onmt.modules.Embeddings`): embedding module to use
	"""
	def __init__(self, rnn_type, bidirectional, num_layers,
				 hidden_size, dropout=0.0, embeddings=None,
				 use_bridge=False):
		super(RNNEncoder, self).__init__()
		assert embeddings is not None

		num_directions = 2 if bidirectional else 1
		assert hidden_size % num_directions == 0
		hidden_size = hidden_size // num_directions
		self.embeddings = embeddings

		self.rnn, self.no_pack_padded_seq = \
			rnn_factory(rnn_type,
						input_size=embeddings.embedding_size,
						hidden_size=hidden_size,
						num_layers=num_layers,
						dropout=dropout,
						bidirectional=bidirectional)

		# Initialize the bridge layer
		self.use_bridge = use_bridge
		if self.use_bridge:
			self._initialize_bridge(rnn_type,
									hidden_size,
									num_layers)

	def forward(self, src, lengths=None, encoder_state=None):
		"See :obj:`EncoderBase.forward()`"
		self._check_args(src, lengths, encoder_state)

		emb = self.embeddings(src)
		s_len, batch, emb_dim = emb.size()

		packed_emb = emb
		if lengths is not None and not self.no_pack_padded_seq:
			# Lengths data is wrapped inside a Variable.
			l = lengths.view(-1).tolist()
			index_sorted = sorted(range(len(l)), key=lambda k: -l[k])
			re_sorted = sorted(range(len(index_sorted)), key=lambda k: index_sorted[k])
			l = sorted(l, reverse=True)
			emb = emb.index_select(1, Variable(torch.LongTensor(index_sorted).type_as(lengths)))

			packed_emb = pack(emb, l)

		memory_bank, encoder_final = self.rnn(packed_emb, encoder_state)

		if lengths is not None and not self.no_pack_padded_seq:
			memory_bank = unpack(memory_bank)[0]
			re_sorted = Variable(torch.LongTensor(re_sorted).type_as(lengths))
			memory_bank = memory_bank.index_select(1, re_sorted)
			encoder_final = (encoder_final[0].index_select(1, re_sorted), encoder_final[1].index_select(1, re_sorted))

		if self.use_bridge:
			encoder_final = self._bridge(encoder_final)
		return encoder_final, memory_bank

	def _initialize_bridge(self, rnn_type,
						   hidden_size,
						   num_layers):

		# LSTM has hidden and cell state, other only one
		number_of_states = 2 if rnn_type == "LSTM" else 1
		# Total number of states
		self.total_hidden_dim = hidden_size * num_layers

		# Build a linear layer for each
		self.bridge = nn.ModuleList([nn.Linear(self.total_hidden_dim,
											   self.total_hidden_dim,
											   bias=True)
									 for i in range(number_of_states)])

	def _bridge(self, hidden):
		"""
		Forward hidden state through bridge
		"""
		def bottle_hidden(linear, states):
			"""
			Transform from 3D to 2D, apply linear and return initial size
			"""
			size = states.size()
			result = linear(states.view(-1, self.total_hidden_dim))
			return F.relu(result).view(size)

		if isinstance(hidden, tuple):  # LSTM
			outs = tuple([bottle_hidden(layer, hidden[ix])
						  for ix, layer in enumerate(self.bridge)])
		else:
			outs = bottle_hidden(self.bridge[0], hidden)
		return outs


class RNNDecoderBase(nn.Module):
	"""
	Base recurrent attention-based decoder class.
	Specifies the interface used by different decoder types
	and required by :obj:`onmt.Models.NMTModel`.


	.. mermaid::

	   graph BT
		  A[Input]
		  subgraph RNN
			 C[Pos 1]
			 D[Pos 2]
			 E[Pos N]
		  end
		  G[Decoder State]
		  H[Decoder State]
		  I[Outputs]
		  F[Memory_Bank]
		  A--emb-->C
		  A--emb-->D
		  A--emb-->E
		  H-->C
		  C-- attn --- F
		  D-- attn --- F
		  E-- attn --- F
		  C-->I
		  D-->I
		  E-->I
		  E-->G
		  F---I

	Args:
	   rnn_type (:obj:`str`):
		  style of recurrent unit to use, one of [RNN, LSTM, GRU, SRU]
	   bidirectional_encoder (bool) : use with a bidirectional encoder
	   num_layers (int) : number of stacked layers
	   hidden_size (int) : hidden size of each layer
	   attn_type (str) : see :obj:`onmt.modules.GlobalAttention`
	   coverage_attn (str): see :obj:`onmt.modules.GlobalAttention`
	   context_gate (str): see :obj:`onmt.modules.ContextGate`
	   copy_attn (bool): setup a separate copy attention mechanism
	   dropout (float) : dropout value for :obj:`nn.Dropout`
	   embeddings (:obj:`onmt.modules.Embeddings`): embedding module to use
	"""
	def __init__(self, rnn_type, bidirectional_encoder, num_layers,
				 hidden_size, attn_type="general",
				 coverage_attn=False, context_gate=None,
				 copy_attn=False, dropout=0.0, embeddings=None,
				 reuse_copy_attn=False):
		super(RNNDecoderBase, self).__init__()

		# Basic attributes.
		self.decoder_type = 'rnn'
		self.bidirectional_encoder = bidirectional_encoder
		self.num_layers = num_layers
		self.hidden_size = hidden_size
		self.embeddings = embeddings
		self.dropout = nn.Dropout(dropout)

		# Build the RNN.
		self.rnn = self._build_rnn(rnn_type,
								   input_size=self._input_size,
								   hidden_size=hidden_size,
								   num_layers=num_layers,
								   dropout=dropout)

		# Set up the context gate.
		self.context_gate = None
		if context_gate is not None:
			self.context_gate = onmt.modules.context_gate_factory(
				context_gate, self._input_size,
				hidden_size, hidden_size, hidden_size
			)

		# Set up the standard attention.
		self._coverage = coverage_attn
		self.attn = onmt.modules.GlobalAttention(
			hidden_size, coverage=coverage_attn,
			attn_type=attn_type
		)

		# Set up a separated copy attention layer, if needed.
		self._copy = False
		if copy_attn and not reuse_copy_attn:
			self.copy_attn = onmt.modules.GlobalAttention(
				hidden_size, attn_type=attn_type
			)
		if copy_attn:
			self._copy = True
		self._reuse_copy_attn = reuse_copy_attn

	def forward(self, tgt, memory_bank, state, memory_lengths=None):
		"""
		Args:
			tgt (`LongTensor`): sequences of padded tokens
								`[tgt_len x batch x nfeats]`.
			memory_bank (`FloatTensor`): vectors from the encoder
				 `[src_len x batch x hidden]`.
			state (:obj:`onmt.Models.DecoderState`):
				 decoder state object to initialize the decoder
			memory_lengths (`LongTensor`): the padded source lengths
				`[batch]`.
		Returns:
			(`FloatTensor`,:obj:`onmt.Models.DecoderState`,`FloatTensor`):
				* decoder_outputs: output from the decoder (after attn)
						 `[tgt_len x batch x hidden]`.
				* decoder_state: final hidden state from the decoder
				* attns: distribution over src at each tgt
						`[tgt_len x batch x src_len]`.
		"""
		# Check
		assert isinstance(state, RNNDecoderState)
		tgt_len, tgt_batch, _ = tgt.size()
		_, memory_batch, _ = memory_bank.size()
		aeq(tgt_batch, memory_batch)
		# END

		# Run the forward pass of the RNN.
		decoder_final, decoder_outputs, attns, c = self._run_forward_pass(
			tgt, memory_bank, state, memory_lengths=memory_lengths)

		# Update the state with the result.
		final_output = decoder_outputs[-1]
		coverage = None
		if "coverage" in attns:
			coverage = attns["coverage"][-1].unsqueeze(0)
		state.update_state(decoder_final, final_output.unsqueeze(0), coverage)

		# Concatenates sequence of tensors along a new dimension.
		decoder_outputs = torch.stack(decoder_outputs)
		for k in attns:
			attns[k] = torch.stack(attns[k])

		return decoder_outputs, state, attns, c

	def init_decoder_state(self, src, memory_bank, encoder_final):
		def _fix_enc_hidden(h):
			# The encoder hidden is  (layers*directions) x batch x dim.
			# We need to convert it to layers x batch x (directions*dim).
			if self.bidirectional_encoder:
				h = torch.cat([h[0:h.size(0):2], h[1:h.size(0):2]], 2)
			return h

		if isinstance(encoder_final, tuple):  # LSTM
			return RNNDecoderState(self.hidden_size,
								   tuple([_fix_enc_hidden(enc_hid)
										 for enc_hid in encoder_final]))
		else:  # GRU
			return RNNDecoderState(self.hidden_size,
								   _fix_enc_hidden(encoder_final))


class StdRNNDecoder(RNNDecoderBase):
	"""
	Standard fully batched RNN decoder with attention.
	Faster implementation, uses CuDNN for implementation.
	See :obj:`RNNDecoderBase` for options.


	Based around the approach from
	"Neural Machine Translation By Jointly Learning To Align and Translate"
	:cite:`Bahdanau2015`


	Implemented without input_feeding and currently with no `coverage_attn`
	or `copy_attn` support.
	"""
	def _run_forward_pass(self, tgt, memory_bank, state, memory_lengths=None):
		"""
		Private helper for running the specific RNN forward pass.
		Must be overriden by all subclasses.
		Args:
			tgt (LongTensor): a sequence of input tokens tensors
								 [len x batch x nfeats].
			memory_bank (FloatTensor): output(tensor sequence) from the encoder
						RNN of size (src_len x batch x hidden_size).
			state (FloatTensor): hidden state from the encoder RNN for
								 initializing the decoder.
			memory_lengths (LongTensor): the source memory_bank lengths.
		Returns:
			decoder_final (Variable): final hidden state from the decoder.
			decoder_outputs ([FloatTensor]): an array of output of every time
									 step from the decoder.
			attns (dict of (str, [FloatTensor]): a dictionary of different
							type of attention Tensor array of every time
							step from the decoder.
		"""
		assert not self._copy  # TODO, no support yet.
		assert not self._coverage  # TODO, no support yet.

		# Initialize local and return variables.
		attns = {}
		emb = self.embeddings(tgt)

		# Run the forward pass of the RNN.
		if isinstance(self.rnn, nn.GRU):
			rnn_output, decoder_final = self.rnn(emb, state.hidden[0])
		else:
			rnn_output, decoder_final = self.rnn(emb, state.hidden)

		# Check
		tgt_len, tgt_batch, _ = tgt.size()
		output_len, output_batch, _ = rnn_output.size()
		aeq(tgt_len, output_len)
		aeq(tgt_batch, output_batch)
		# END

		# Calculate the attention.
		decoder_outputs, p_attn, c = self.attn(
			rnn_output.transpose(0, 1).contiguous(),
			memory_bank.transpose(0, 1),
			memory_lengths=memory_lengths
		)
		attns["std"] = p_attn

		# Calculate the context gate.
		if self.context_gate is not None:
			decoder_outputs = self.context_gate(
				emb.view(-1, emb.size(2)),
				rnn_output.view(-1, rnn_output.size(2)),
				decoder_outputs.view(-1, decoder_outputs.size(2))
			)
			decoder_outputs = \
				decoder_outputs.view(tgt_len, tgt_batch, self.hidden_size)

		decoder_outputs = self.dropout(decoder_outputs)
		return decoder_final, decoder_outputs, attns, c

	def _build_rnn(self, rnn_type, **kwargs):
		rnn, _ = rnn_factory(rnn_type, **kwargs)
		return rnn

	@property
	def _input_size(self):
		"""
		Private helper returning the number of expected features.
		"""
		return self.embeddings.embedding_size


class InputFeedRNNDecoder(RNNDecoderBase):
	"""
	Input feeding based decoder. See :obj:`RNNDecoderBase` for options.

	Based around the input feeding approach from
	"Effective Approaches to Attention-based Neural Machine Translation"
	:cite:`Luong2015`


	.. mermaid::

	   graph BT
		  A[Input n-1]
		  AB[Input n]
		  subgraph RNN
			E[Pos n-1]
			F[Pos n]
			E --> F
		  end
		  G[Encoder]
		  H[Memory_Bank n-1]
		  A --> E
		  AB --> F
		  E --> H
		  G --> H
	"""

	def _run_forward_pass(self, tgt, memory_bank, state, memory_lengths=None):
		"""
		See StdRNNDecoder._run_forward_pass() for description
		of arguments and return values.
		"""
		# Additional args check.
		input_feed = state.input_feed.squeeze(0)
		input_feed_batch, _ = input_feed.size()
		tgt_len, tgt_batch, _ = tgt.size()
		aeq(tgt_batch, input_feed_batch)
		# END Additional args check.

		# Initialize local and return variables.
		decoder_outputs, cs = [], []
		attns = {"std": []}
		if self._copy:
			attns["copy"] = []
		if self._coverage:
			attns["coverage"] = []

		emb = self.embeddings(tgt)
		assert emb.dim() == 3  # len x batch x embedding_dim

		hidden = state.hidden
		coverage = state.coverage.squeeze(0) \
			if state.coverage is not None else None

		# Input feed concatenates hidden state with
		# input at every time step.
		for i, emb_t in enumerate(emb.split(1)):
			emb_t = emb_t.squeeze(0)
			decoder_input = torch.cat([emb_t, input_feed], 1)

			rnn_output, hidden = self.rnn(decoder_input, hidden)
			decoder_output, p_attn, c = self.attn(
				rnn_output,
				memory_bank.transpose(0, 1),
				memory_lengths=memory_lengths)
			if self.context_gate is not None:
				# TODO: context gate should be employed
				# instead of second RNN transform.
				decoder_output = self.context_gate(
					decoder_input, rnn_output, decoder_output
				)
			decoder_output = self.dropout(decoder_output)
			input_feed = decoder_output

			decoder_outputs += [decoder_output]
			attns["std"] += [p_attn]
			cs += [c]

			# Update the coverage attention.
			if self._coverage:
				coverage = coverage + p_attn \
					if coverage is not None else p_attn
				attns["coverage"] += [coverage]

			# Run the forward pass of the copy attention layer.
			if self._copy and not self._reuse_copy_attn:
				_, copy_attn = self.copy_attn(decoder_output,
											  memory_bank.transpose(0, 1))
				attns["copy"] += [copy_attn]
			elif self._copy:
				attns["copy"] = attns["std"]
		# Return result.
		return hidden, decoder_outputs, attns, cs

	def _build_rnn(self, rnn_type, input_size,
				   hidden_size, num_layers, dropout):
		assert not rnn_type == "SRU", "SRU doesn't support input feed! " \
				"Please set -input_feed 0!"
		if rnn_type == "LSTM":
			stacked_cell = onmt.modules.StackedLSTM
		else:
			stacked_cell = onmt.modules.StackedGRU
		return stacked_cell(num_layers, input_size,
							hidden_size, dropout)

	@property
	def _input_size(self):
		"""
		Using input feed by concatenating input with attention vectors.
		"""
		return self.embeddings.embedding_size + self.hidden_size


class NMTModel(nn.Module):
	"""
	Core trainable object in OpenNMT. Implements a trainable interface
	for a simple, generic encoder + decoder model.

	Args:
	  encoder (:obj:`EncoderBase`): an encoder object
	  decoder (:obj:`RNNDecoderBase`): a decoder object
	  multi<gpu (bool): setup for multigpu support
	"""
	def __init__(self, encoder, decoder, context, multigpu=False, context_type="",tgt_vocab=None,beam_size=None,
				 n_best=None,min_length=None, max_length=None,stepwise_penalty=None,scorer=None,block_ngram_repeat=None,
				 ignore_when_blocking=None,gpu=None,copy_attn=None,context_size=None):
		self.multigpu = multigpu
		super(NMTModel, self).__init__()
		self.encoder = encoder
		self.decoder = decoder
		self.doc_context = context
		self.context_type = context_type
		self.context_size = context_size
		#Generator ??
		#beam search
		self.vocab = tgt_vocab
		self.beam_size = beam_size
		self.n_best = n_best
		self.min_length = min_length
		self.max_length = max_length
		self.stepwise_penalty = stepwise_penalty
		self.global_scorer = scorer
		self.block_ngram_repeat = block_ngram_repeat
		self.ignore_when_blocking = ignore_when_blocking
		self.my_gpu = gpu
		self.copy_attn =copy_attn


	def forward(self, src, tgt, lengths, dec_state=None, context_index=None, part="all", cache=None, batch_i=None,REINFORCE=False,batch = None, data=None):
		"""Forward propagate a `src` and `tgt` pair for training.
		Possible initialized with a beginning decoder state.

		Args:
			src (:obj:`Tensor`):
				a source sequence passed to encoder.
				typically for inputs this will be a padded :obj:`LongTensor`
				of size `[len x batch x features]`. however, may be an
				image or other generic input depending on encoder.
			tgt (:obj:`LongTensor`):
				 a target sequence of size `[tgt_len x batch]`.
			lengths(:obj:`LongTensor`): the src lengths, pre-padding `[batch]`.
			dec_state (:obj:`DecoderState`, optional): initial decoder state
		Returns:
			(:obj:`FloatTensor`, `dict`, :obj:`onmt.Models.DecoderState`):

				 * decoder output `[tgt_len x batch x hidden]`
				 * dictionary attention dists of `[tgt_len x batch x src_len]`
				 * final decoder state
		"""
		# print (context_index)
		# #Visualize sentences
		# for i in range(tgt.size()[1]):
		# 	list_words = []
		# 	for j in range(tgt.size()[0]):
		# 		list_words.append(self.vocab.itos[tgt[j,i,0]])
		# 	print (" ".join(list_words))
		tgt = tgt[:-1]  # exclude last target from inputs WHYYYYYY???? <eos>?
		attn_word_enc, attn_sent_enc, attn_word_dec = None, None, None

		enc_final, memory_bank = self.encoder(src, lengths)
		if self.doc_context and part in ["all", "context"]:
			if self.context_type == "HAN_join":
				memory_bank, attn_word_enc, attn_sent_enc = self.doc_context[0](src, memory_bank, memory_bank, context_index)
			elif self.context_type == "HAN_enc" or self.context_type == "HAN_dec_source":
				memory_bank, attn_word_enc, attn_sent_enc = self.doc_context(src, memory_bank, memory_bank, context_index, batch_i=batch_i)

		if REINFORCE == False:
			# print ('YAHOOO')

			enc_state = \
				self.decoder.init_decoder_state(src, memory_bank, enc_final)
			decoder_outputs, dec_state, attns, mid = \
				self.decoder(tgt, memory_bank,
							 enc_state if dec_state is None
							 else dec_state,
							 memory_lengths=lengths)


			if self.doc_context and part in ["all", "context"]:
				if self.context_type == "HAN_join":
					decoder_outputs, attn_word_dec, attn_sent_dec = self.doc_context[1](tgt, decoder_outputs, decoder_outputs, context_index, batch_i=batch_i)
				elif "HAN_dec" in self.context_type:
					query = decoder_outputs
					if self.context_type == "HAN_dec_source":
						ctxt = memory_bank
						inp = src
					elif self.context_type == "HAN_dec_context":
						ctxt = mid
						inp = tgt
					else:
						ctxt = decoder_outputs
						inp = tgt
					decoder_outputs, attn_word_dec, attn_sent_dec = self.doc_context(inp, query, ctxt, context_index, batch_i=batch_i)

		elif REINFORCE == True:

			#Not finished of implementing
			# self.RISK_sampling(batch,data,context_index,src,lengths,enc_final,memory_bank,part)

			ret = self.REINFORCE_step(batch,data,context_index,src,lengths,enc_final,memory_bank,part)
			return ret



		if self.multigpu:
			# Not yet supported on multi-gpu
			dec_state = None
			attns = None
		return decoder_outputs, attns, dec_state


	def RISK_sampling(self,batch, data, context, src, src_lengths,enc_states, memory_bank,translate_part):
		batch_size = batch.batch_size
		data_type = 'text'
		sampling_num = 3
		vocab = self.vocab

		store_sampling_id = torch.zeros(self.max_length+1, sampling_num,batch_size, dtype=torch.long)
		store_sampling_id[0, :] = torch.ones([1, sampling_num,batch_size], dtype=torch.long) * 2
		store_sampling_prob = torch.zeros(self.max_length, sampling_num,batch_size,dtype=torch.float)

		cache, ind_cache = [src[:, 0:1, :], enc_states[:, 0:1, :], enc_states[:, 0:1, :]], []

		for batch_i in range(batch_size):


			if isinstance(enc_states, tuple):
				enc_states_i = (enc_states[0][:, batch_i:batch_i + 1, :], enc_states[1][:, batch_i:batch_i + 1, :])
			else:
				enc_states_i = enc_states[:, batch_i:batch_i + 1, :]
			print (enc_states_i.size())
			dec_states = self.decoder.init_decoder_state(
				src[:, batch_i:batch_i + 1, :], memory_bank[:, batch_i:batch_i + 1, :], enc_states_i)

			# (2) Repeat src objects `beam_size` times.
			src_map = batch.src_map[batch_i:batch_i + 1].data \
				if data_type == 'text' and self.copy_attn else None
			# PROBLEM: memory_bank[:, batch_i:batch_i+1,:].data
			memory_bank_i = self.rvar_sam(memory_bank[:, batch_i:batch_i + 1, :].data)
			print (memory_bank_i.size())
			memory_lengths = src_lengths[batch_i:batch_i + 1].repeat(sampling_num)
			dec_states.repeat_beam_size_times(sampling_num)

			for i in range(self.max_length):
				inp = store_sampling_id[i,:,batch_i]
				inp = inp.unsqueeze(0)
				print (inp.size())
				print (inp)
				print ('yay')
				# Temporary kludge solution to handle changed dim expectation
				# in the decoder
				inp = inp.unsqueeze(2)
				print (type(inp))

				# Run one step.
				dec_out, dec_states, attn, mid = self.decoder(
					inp, memory_bank_i, dec_states, memory_lengths=memory_lengths)

				# dec_out: beam x rnn_size

				if self.doc_context and translate_part in ["all", "context"]:
					if self.context_type == "HAN_join":
						dec_out, _, _ = self.doc_context[1](cache[0], dec_out, cache[1], context, batch_i=batch_i)
					elif "HAN_dec" in self.context_type:
						dec_out, _, _ = self.doc_context(cache[0], dec_out, cache[1], context, batch_i=batch_i)

				dec_out = dec_out.squeeze(0)

				# (b) Compute a vector of batch x beam word scores.
				if not self.copy_attn:
					out = self.generator.forward(dec_out)
					out = out.view(sampling_num, 1, -1)
				else:
					out = self.generator.forward(dec_out,
												 attn["copy"].squeeze(0), src_map)
					# beam x (tgt_vocab + extra_vocab)
					out = data.collapse_copy_scores(
						out.view(sampling_num, 1, -1),
						batch[batch_i], vocab, data.src_vocabs)

				#Obtain probability and value
				id_list = []
				prob_list = []
				for sam_i in range(sampling_num):
					sampler = Categorical(out[sam_i,0,:].squeeze())
					wordID = sampler.sample()
					word_prob = out[sam_i,0,wordID]
					print (vocab.itos[wordID])
					print (word_prob)
					id_list.append(wordID)
					prob_list.append(word_prob)

				stack_id_list = torch.stack(id_list)
				stack_prob_list = torch.stack(prob_list)
				print (stack_id_list.size())
				print (stack_prob_list.size())
				print ('yupii')
				store_sampling_id[i+1,:,batch_i] = stack_id_list
				print (store_sampling_id[i+1,:,batch_i])
				store_sampling_prob[i,:,batch_i] = stack_prob_list

			#Update cache
			# cache, ind_cache = self.update_context(store_sampling_id[], cache, ind_cache,
			# 									   enc_states, src, memory_bank, src_lengths, batch_i, translate_part,
			# 									   vocab.stoi[onmt.io.PAD_WORD])




	def REINFORCE_step(self,batch, data, context, src, src_lengths,enc_states, memory_bank,translate_part):
		beam_size = self.beam_size
		batch_size = batch.batch_size
		data_type = 'text'
		vocab = self.vocab

		# Define a list of tokens to exclude from ngram-blocking
		# exclusion_list = ["<t>", "</t>", "."]
		exclusion_tokens = set([vocab.stoi[t]
								for t in self.ignore_when_blocking])

		beam = [onmt.translate.Beam(beam_size, n_best=self.n_best,
									cuda=self.my_gpu,
									global_scorer=self.global_scorer,
									pad=vocab.stoi[onmt.io.PAD_WORD],
									eos=vocab.stoi[onmt.io.EOS_WORD],
									bos=vocab.stoi[onmt.io.BOS_WORD],
									min_length=self.min_length,
									stepwise_penalty=self.stepwise_penalty,
									block_ngram_repeat=self.block_ngram_repeat,
									exclusion_tokens=exclusion_tokens,
									for_training=True)
				for __ in range(batch_size)]

		if src_lengths is None:
			src_lengths = torch.Tensor(batch_size).type_as(memory_bank.data)\
												  .long()\
												  .fill_(memory_bank.size(0))


		ret = {"predictions": [],
			   "scores": [],
			   "attention": [],
			   "out_prob": []}

		cache, ind_cache = [src[:,0:1,:], enc_states[:,0:1,:], enc_states[:,0:1,:]], []


		#print (batch_size)
		#print (memory_bank.size())
		for batch_i in range(batch_size):
			#print (batch_i)

			if isinstance(enc_states, tuple):
				enc_states_i = (enc_states[0][:,batch_i:batch_i+1,:], enc_states[1][:,batch_i:batch_i+1,:])
			else:
				enc_states_i = enc_states[:,batch_i:batch_i+1,:]

			dec_states = self.decoder.init_decoder_state(
										src[:,batch_i:batch_i+1,:], memory_bank[:,batch_i:batch_i+1,:], enc_states_i)

			# (2) Repeat src objects `beam_size` times.
			src_map = batch.src_map[batch_i:batch_i+1].data \
				 if data_type == 'text' and self.copy_attn else None
			#PROBLEM: memory_bank[:, batch_i:batch_i+1,:].data
			memory_bank_i = self.rvar(memory_bank[:, batch_i:batch_i+1, :].data)
			memory_lengths = src_lengths[batch_i:batch_i+1].repeat(beam_size)
			dec_states.repeat_beam_size_times(beam_size)

			# (3) run the decoder to generate sentences, using beam search.
			for i in range(self.max_length):
				if beam[batch_i].done():
					break

				# Construct batch x beam_size nxt words.
				# Get all the pending current beam words and arrange for forward.
				inp = self.var(beam[batch_i].get_current_state().view(1, -1))
				# print (inp.size())
				# print (type(inp))

				# Turn any copied words to UNKs
				# 0 is unk
				if self.copy_attn:
					inp = inp.masked_fill(
						inp.gt(len(vocab) - 1), 0)

				# Temporary kludge solution to handle changed dim expectation
				# in the decoder
				inp = inp.unsqueeze(2)
				# print(type(inp))

				# Run one step.
				dec_out, dec_states, attn, mid = self.decoder(
					inp, memory_bank_i, dec_states, memory_lengths=memory_lengths)

				# dec_out: beam x rnn_size

				if self.doc_context and translate_part in ["all", "context"]:
					if self.context_type == "HAN_join":
						dec_out, _,_ = self.doc_context[1](cache[0], dec_out, cache[1], context, batch_i=batch_i)
					elif "HAN_dec" in self.context_type:
						dec_out, _,_  = self.doc_context(cache[0], dec_out, cache[1], context, batch_i=batch_i)

				dec_out = dec_out.squeeze(0)

				# (b) Compute a vector of batch x beam word scores.
				if not self.copy_attn:
					out = self.generator.forward(dec_out)
					out = out.view(beam_size, 1, -1)
					# print (out.requires_grad)
					# print (out.size())
					# beam x tgt_vocab
					beam_attn = attn["std"].view(beam_size, 1, -1)
				else:
					out = self.generator.forward(dec_out,
						attn["copy"].squeeze(0), src_map)
					# beam x (tgt_vocab + extra_vocab)
					out = data.collapse_copy_scores(
						out.view(beam_size, 1, -1),
						batch[batch_i], vocab, data.src_vocabs)
					# beam x tgt_vocab
					out = out.log()
					beam_attn = attn["copy"].view(beam_size, 1, -1)
				# (c) Advance each beam.
				beam[batch_i].advance(out[:, 0], beam_attn.data[:, 0, :memory_lengths[0]],out_prob = out)
				dec_states.beam_update(0, beam[batch_i].get_current_origin(), beam_size)

			self._from_beam(beam[batch_i], ret)
			choose=0
			if len(ret["predictions"][-1][choose]) == 1:
				choose=1
			cache, ind_cache = self.update_context(ret["predictions"][-1][choose], cache, ind_cache,
									enc_states, src, memory_bank, src_lengths, batch_i, translate_part, vocab.stoi[onmt.io.PAD_WORD])
		del beam
		# (4) Extract sentences from beam.
		#ret = self._from_beam(beam)
		# ret["gold_score"] = [0] * batch_size
		# ret["ctx_attn"] = None
		# if "tgt" in batch.__dict__:
		# 	ret["gold_score"], ret["ctx_attn"] = self._run_target(batch, data, context, translate_part)
		# ret["batch"] = batch
		#print (ret)
		return ret

	# Help functions for working with beams and batches
	def var(self,a):
		with torch.no_grad():
			theOut =Variable(a)
		return theOut

	def rvar(self,a):
		#print (a.size())
		return self.var(a.repeat(1, self.beam_size, 1))

	def var_sam(self,a):
		with torch.no_grad():
			theOut =Variable(a)
		return theOut

	def rvar_sam(self,a):
		#print (a.size())
		return self.var(a.repeat(1, 3, 1))

	def update_context_sam(self,latest_sampled_ids, cache, ind_cache, enc_states, src, memory_bank, src_lengths, batch_i, translate_part,
					   pad):

		if self.context_type == "HAN_enc" or self.context_type == "HAN_dec_source":
			b_len = min(self.context_size - 1, batch_i)
			cache = [src[:, batch_i - b_len:batch_i + 1, :], memory_bank[:, batch_i - b_len:batch_i + 1, :]]

		elif self.context_type in {"HAN_dec", "HAN_dec_context", "HAN_join"}:
			ind_cache.append(pred)
			if len(ind_cache) > self.context_size:
				del ind_cache[0]
			s_len = max([len(i) for i in ind_cache])
			b_len = len(ind_cache)
			pred = np.empty([s_len, b_len])
			pred.fill(pad)
			for i in range(b_len):
				pred[:len(ind_cache[i]), i] = ind_cache[i]

			prev_context = Variable(torch.Tensor(pred).type_as(memory_bank.data).long().unsqueeze(2))
			prev_context = prev_context[:-1]
			prev_out, prev_memory_bank = self.run_decoder(prev_context, enc_states, src, memory_bank, src_lengths,
														  batch_i, b_len - 1)

			if self.context_type == "HAN_dec_context":
				cache = [prev_context, prev_memory_bank, None]
			else:
				cache = [prev_context, prev_out, None]

		return cache, ind_cache

	def update_context(self, pred, cache, ind_cache, enc_states, src, memory_bank, src_lengths, batch_i, translate_part,
					   pad):

		if self.context_type == "HAN_enc" or self.context_type == "HAN_dec_source":
			b_len = min(self.context_size - 1, batch_i)
			cache = [src[:, batch_i - b_len:batch_i + 1, :], memory_bank[:, batch_i - b_len:batch_i + 1, :]]

		elif self.context_type in {"HAN_dec", "HAN_dec_context", "HAN_join"}:
			ind_cache.append(pred)
			if len(ind_cache) > self.context_size:
				del ind_cache[0]
			s_len = max([len(i) for i in ind_cache])
			b_len = len(ind_cache)
			pred = np.empty([s_len, b_len])
			pred.fill(pad)
			for i in range(b_len):
				pred[:len(ind_cache[i]), i] = ind_cache[i]

			prev_context = Variable(torch.Tensor(pred).type_as(memory_bank.data).long().unsqueeze(2))
			prev_context = prev_context[:-1]
			prev_out, prev_memory_bank = self.run_decoder(prev_context, enc_states, src, memory_bank, src_lengths,
														  batch_i, b_len - 1)

			if self.context_type == "HAN_dec_context":
				cache = [prev_context, prev_memory_bank, None]
			else:
				cache = [prev_context, prev_out, None]

		return cache, ind_cache

	def run_decoder(self, prev_context, enc_states, src, memory_bank, src_lengths, batch_i, b_len=0):
		if isinstance(enc_states, tuple):
			enc_states_i = (
			enc_states[0][:, batch_i - b_len:batch_i + 1, :], enc_states[1][:, batch_i - b_len:batch_i + 1, :])
		else:
			enc_states_i = enc_states[:, batch_i - b_len:batch_i + 1, :]

		dec_states = self.decoder.init_decoder_state(src[:, batch_i - b_len:batch_i + 1, :],
														   memory_bank[:, batch_i - b_len:batch_i + 1, :],
														   enc_states_i)
		out, _, _, mid = self.decoder(prev_context,
											memory_bank[:, batch_i - b_len:batch_i + 1, :],
											dec_states,
											memory_lengths=src_lengths[batch_i - b_len:batch_i + 1])

		return out, mid

	def _from_beam(self, b, ret):
		n_best = self.n_best
		scores, ks = b.sort_finished(minimum=n_best)
		hyps, attn, out_prob_from_beam = [], [], []
		for i, (times, k) in enumerate(ks[:n_best]):
			hyp, att, out_prob = b.get_hyp(times, k)
			hyps.append(hyp)
			attn.append(att)
			out_prob_from_beam.append(out_prob)
		ret["predictions"].append(hyps)
		ret["scores"].append(scores)
		ret["attention"].append(attn)
		ret["out_prob"].append(out_prob_from_beam)


class DecoderState(object):
	"""Interface for grouping together the current state of a recurrent
	decoder. In the simplest case just represents the hidden state of
	the model.  But can also be used for implementing various forms of
	input_feeding and non-recurrent models.

	Modules need to implement this to utilize beam search decoding.
	"""
	def detach(self):
		for h in self._all:
			if h is not None:
				h = h.detach()

	def beam_update(self, idx, positions, beam_size):
		for e in self._all:
			sizes = e.size()
			br = sizes[1]
			if len(sizes) == 3:
				sent_states = e.view(sizes[0], beam_size, br // beam_size,
									 sizes[2])[:, :, idx]
			else:
				sent_states = e.view(sizes[0], beam_size,
									 br // beam_size,
									 sizes[2],
									 sizes[3])[:, :, idx]

			sent_states.data.copy_(
				sent_states.data.index_select(1, positions))


class RNNDecoderState(DecoderState):
	def __init__(self, hidden_size, rnnstate):
		"""
		Args:
			hidden_size (int): the size of hidden layer of the decoder.
			rnnstate: final hidden state from the encoder.
				transformed to shape: layers x batch x (directions*dim).
		"""
		if not isinstance(rnnstate, tuple):
			self.hidden = (rnnstate,)
		else:
			self.hidden = rnnstate
		self.coverage = None

		# Init the input feed.
		batch_size = self.hidden[0].size(1)
		h_size = (batch_size, hidden_size)
		self.input_feed = Variable(self.hidden[0].data.new(*h_size).zero_(),
								   requires_grad=False).unsqueeze(0)

	@property
	def _all(self):
		return self.hidden + (self.input_feed,)

	def update_state(self, rnnstate, input_feed, coverage):
		if not isinstance(rnnstate, tuple):
			self.hidden = (rnnstate,)
		else:
			self.hidden = rnnstate
		self.input_feed = input_feed
		self.coverage = coverage

	def repeat_beam_size_times(self, beam_size):
		""" Repeat beam_size times along batch dimension. """
		vars = [Variable(e.data.repeat(1, beam_size, 1), volatile=True)
				for e in self._all]
		self.hidden = tuple(vars[:-1])
		self.input_feed = vars[-1]
