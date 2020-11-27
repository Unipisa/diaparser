# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModel, AutoConfig

from .scalar_mix import ScalarMix
from .dropout import TokenDropout
# from torch.cuda import memory_allocated

class BertEmbedding(nn.Module):
    r"""
    A module that directly utilizes the pretrained models in `transformers`_ to produce BERT representations.
    While mainly tailored to provide input preparation and post-processing for the BERT model,
    it is also compatible with other pretrained language models like XLNet, RoBERTa and ELECTRA, etc.

    Args:
        model (str):
            Path or name of the pretrained models registered in `transformers`_, e.g., ``'bert-base-cased'``.
        n_layers (int):
            The number of layers from the model to use.
            If 0, uses all layers.
        n_out (int):
            The requested size of the embeddings.
            If 0, uses the size of the pretrained embedding model.
        stride (int):
            A sequence longer than the limited max length will be splitted into several small pieces
            with a window size of ``stride``. Default: 5.
        pad_index (int):
            The index of the padding token in the BERT vocabulary. Default: 0.
        dropout (float):
            The dropout ratio of BERT layers. Default: 0.
            This value will be passed into the :class:`ScalarMix` layer.
        requires_grad (bool):
            If ``True``, the model parameters will be updated together with the downstream task.
            Default: ``False``.

    .. _transformers:
        https://github.com/huggingface/transformers
    """

    def __init__(self, model, n_layers, n_out, stride=5, pad_index=0, dropout=0, requires_grad=False,
                 mask_token_id=0, token_dropout=0.0, mix_dropout=0.0,
                 use_hidden_states=True, use_attentions=False,
                 attention_head=0, attention_layer=8):
        r"""
        :param model (str): path or name of the pretrained model.
        :param n_layers (int): number of layers from the model to use.
            If 0, use all layers.
        :param n_out (int): the requested size of the embeddings.
            If 0, use the size of the pretrained embedding model
        :param requires_grad (bool): whether to fine tune the embeddings.
        :param mask_token_id (int): the value of the [MASK] token to use for dropped tokens.
        :param dropout (float): drop layers with this probability when comuting their
            weighted average with ScalarMix.
        :param use_hidden_states (bool): use the output hidden states from bert if True, or else
            the outputs.
        :param use_attentions (bool): wheth4er to use attention weights.
        :param attention_head (int): which attention head to use.
        :param attention_layer (int): which attention layer to use.
        """

        super().__init__()

        config = AutoConfig.from_pretrained(model, output_hidden_states=True,
                                            output_attentions=use_attentions)
        self.bert = AutoModel.from_pretrained(model, config=config)
        self.bert.requires_grad_(requires_grad)

        self.model = model
        self.n_layers = n_layers or self.bert.config.num_hidden_layers
        self.hidden_size = self.bert.config.hidden_size
        self.n_out = n_out or self.hidden_size
        self.stride = stride
        self.pad_index = self.bert.config.pad_token_id
        self.mix_dropout = mix_dropout
        self.token_dropout = token_dropout
        self.requires_grad = requires_grad
        self.max_len = self.bert.config.max_position_embeddings
        self.use_hidden_states = use_hidden_states
        self.mask_token_id = mask_token_id
        self.use_attentions = use_attentions
        self.head = attention_head
        self.attention_layer = attention_layer

        self.token_dropout = TokenDropout(token_dropout, mask_token_id) if token_dropout else None
        self.scalar_mix = ScalarMix(self.n_layers, mix_dropout)
        if self.hidden_size != self.n_out:
            self.projection = nn.Linear(self.hidden_size, self.n_out, False)

    def __repr__(self):
        s = f"{self.model}, n_layers={self.n_layers}, n_out={self.n_out}"
        s += f", pad_index={self.pad_index}"
        s += f", max_len={self.max_len}"
        if self.mix_dropout > 0:
            s += f", mix_dropout={self.mix_dropout}"
        if self.use_attentions:
            s += f", use_attentions={self.use_attentions}"
        if self.hidden_size != self.n_out:
            s += f", projection=({self.hidden_size} x {self.n_out})"
        s += f", mask_token_id={self.mask_token_id}"
        if self.requires_grad:
            s += f", requires_grad={self.requires_grad}"

        return f"{self.__class__.__name__}({s})"

    def forward(self, subwords):
        r"""
        Args:
            subwords (~torch.Tensor): ``[batch_size, seq_len, fix_len]``.
        Returns:
            ~torch.Tensor:
                BERT embeddings of shape ``[batch_size, seq_len, n_out]``.
        """
        batch_size, seq_len, fix_len = subwords.shape
        if self.token_dropout:
            subwords = self.token_dropout(subwords)

        mask = subwords.ne(self.pad_index)
        lens = mask.sum((1, 2))

        if not self.requires_grad:
            self.bert.eval()    # CHECK_ME (supar does not do)
        # [batch_size, n_subwords]
        subwords = pad_sequence(subwords[mask].split(lens.tolist()), True)
        bert_mask = pad_sequence(mask[mask].split(lens.tolist()), True)
        # Outputs from the transformer:
        # - last_hidden_state: [batch, seq_len, hidden_size]
        # - pooler_output: [batch, hidden_size],
        # - hidden_states (optional): [[batch_size, seq_length, hidden_size]] * (1 + layers)
        # - attentions (optional): [[batch_size, num_heads, seq_length, seq_length]] * layers
        # print('<BERT, GPU MiB:', memory_allocated() // (1024*1024)) # DEBUG
        outputs = self.bert(subwords[:, :self.max_len], attention_mask=bert_mask[:, :self.max_len].float())
        # print('BERT>, GPU MiB:', memory_allocated() // (1024*1024)) # DEBUG
        if self.use_hidden_states:
            bert_idx = -2 if self.use_attentions else -1
            bert = outputs[bert_idx]
            # [n_layers, batch_size, n_subwords, hidden_size]
            bert = bert[-self.n_layers:]
            # [batch_size, n_subwords, hidden_size]
            bert = self.scalar_mix(bert)
            for i in range(self.stride, (subwords.shape[1]-self.max_len+self.stride-1)//self.stride*self.stride+1, self.stride):
                part = self.bert(subwords[:, i:i+self.max_len], attention_mask=bert_mask[:, i:i+self.max_len].float())[bert_idx]
                bert = torch.cat((bert, self.scalar_mix(part[-self.n_layers:])[:, self.max_len-self.stride:]), 1)
        else:
            bert = outputs[0]
        
        # [batch_size, n_subwords]
        bert_lens = mask.sum(-1)
        bert_lens = bert_lens.masked_fill_(bert_lens.eq(0), 1)
        # [batch_size, seq_len, fix_len, hidden_size]
        embed = bert.new_zeros(*mask.shape, self.hidden_size)
        embed = embed.masked_scatter_(mask.unsqueeze(-1), bert[bert_mask])
        # [batch_size, seq_len, hidden_size]
        embed = embed.sum(2) / bert_lens.unsqueeze(-1)  # sum wordpieces
        seq_attn = None
        if self.use_attentions:
            # (a list of layers) = [ [batch, num_heads, sent_len, sent_len] ]
            attns = outputs[-1]
            # [batch, n_subwords, n_subwords]
            attn = attns[self.attention_layer][:,self.head,:,:]  # layer 9 represents syntax
            # squeeze out multiword tokens
            mask2 = ~mask
            mask2[:,:,0] = True  # keep first column
            sub_masks = pad_sequence(mask2[mask].split(lens.tolist()), True)
            seq_mask = torch.einsum('bi,bj->bij', sub_masks, sub_masks)  # outer product
            seq_lens = seq_mask.sum((1,2))
            # [batch_size, seq_len, seq_len]
            sub_attn = attn[seq_mask].split(seq_lens.tolist())
            # fill a tensor [batch_size, seq_len, seq_len]
            seq_attn = attn.new_zeros(batch_size, seq_len, seq_len)
            for i, attn_i in enumerate(sub_attn):
                size = sub_masks[i].sum(0)
                attn_i = attn_i.view(size, size)
                seq_attn[i,:size,:size] = attn_i
        if hasattr(self, 'projection'):
            embed = self.projection(embed)

        return embed, seq_attn
