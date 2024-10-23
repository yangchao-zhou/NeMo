# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy

import torch

from nemo.collections.common.tokenizers.tokenizer_spec import TokenizerSpec
from nemo.collections.nlp.data.language_modeling.megatron.gpt_sft_dataset import GPTSFTDataset
from nemo.utils import logging

__all__ = ['GPTSFTChatDataset', 'get_prompt_template_example']


IGNORE_INDEX = -100


def _get_header_conversation_type_mask_role(source, special_tokens):
    system_fmt = special_tokens['system_fmt']
    conversation = source['system']
    mask_role = source.get('mask', 'User')
    header = system_fmt.format(conversation)
    conversation = _add_speaker_and_signal(header, source['conversations'], special_tokens)
    return header, conversation, mask_role


def get_prompt_template_example(special_tokens):
    source = {
        'system': '{system message}',
        'conversations': [
            {'from': 'User', 'value': '{turn 1 user message}'},
            {'from': 'Assistant', 'value': '{turn 1 assistant message}'},
            {'from': 'User', 'value': '{turn 2 user message}'},
            {'from': 'Assistant', 'value': '{turn 2 assistant message}'},
        ],
        "mask": "User",
        "type": "VALUE_TO_TEXT",
    }
    _, conversation, _ = _get_header_conversation_type_mask_role(source, special_tokens)
    return conversation


def identify_start_index_of_subsequence(subsequence, sequence):
    """find the location of the small tensor in the large tensor.
        e.g.  small = [1,3], large = [2,3,1,3], returns 2
              small = [3,2], large = [2,3,1,3], returns -1
    Args:
        small (tensor): small tensor
        large (tensor): large tensor
    """
    for i in range(sequence.size(0) - subsequence.size(0) + 1):
        if torch.equal(sequence[i : i + subsequence.size(0)], subsequence):
            return i
    return -1


def _mask_targets(
    target,
    tokenized_lens,
    speakers,
    header_len,
    s_ids,
    tokenizer,
    mask_role,
    special_tokens,
    is_mask_list,
):
    """This function masks the tokens so the loss is computed only on the non-masked role's responses.
    For 'TEXT_TO_VALUE' type, the loss is computed on the value attributes.

    Args:
        target (Tensor): input ids
        tokenized_lens (List[int]): array of lengths of each turns
        speakers (List[str]): array of speakers of each turns
        header_len (int): the system prompt length
        s_ids (List[Tensor]): array of tokenized ids of each turns
        tokenizer (TokenizerSpec): tokenizer object
        mask_role (str): the speaker id to be masked from loss computation
        special_tokens (dict): special tokens used for the chat prompt. It has the keys: system_turn_start, turn_start, label_start, end_of_turn
    """
    ai_fmt = special_tokens['ai_fmt']
    ai_fmt_start = ai_fmt.split("{}")[0]
    if len(ai_fmt_start) > 0:
        ai_start_len = len(tokenizer.text_to_ids(ai_fmt_start))
    else:
        ai_start_len = 0

    cur_idx = header_len
    tgt_len = target.shape[0]
    for i, (tokenized_len, speaker, s_id, is_mask) in enumerate(zip(tokenized_lens, speakers, s_ids, is_mask_list)):
        if cur_idx >= tgt_len:
            break

        if speaker == mask_role:
            target[cur_idx: cur_idx + tokenized_len] = IGNORE_INDEX
        else:
            target[cur_idx: cur_idx + ai_start_len] = IGNORE_INDEX
        if is_mask and speaker != mask_role:
            target[cur_idx: cur_idx + tokenized_len] = IGNORE_INDEX

        cur_idx += tokenized_len


def response_value_formater(label, label_start, end_signal):
    if isinstance(label, str):
        return label_start + label + end_signal
    elif label is None:
        return ''
    else:
        raise ValueError(f'Unknown label type {type(label)}, only str type is supported')



def _add_speaker_and_signal(header, source, special_tokens):
    user_fmt = special_tokens['user_fmt']
    ai_fmt = special_tokens['ai_fmt']

    conversation = header
    for i, sentence in enumerate(source):
        sentence_from = sentence["from"]

        if sentence_from == "User":
            sentence["value"] = user_fmt.format(sentence["value"])
        else:  # Assistant
            sentence["value"] = ai_fmt.format(sentence["value"])

        conversation += sentence["value"]
    return conversation


def preprocess(
    source: dict,
    tokenizer: TokenizerSpec,
    special_tokens: dict,
):
    """
    Given a conversation list. This transform:
    1. Add signal '### ' at the beginning each sentence, with end signal '\n';
    2. Concatenate conversations together;
    3. Tokenize the concatenated conversation;
    4. Make a deepcopy as the target. Mask human words with IGNORE_INDEX.
    """
    # header, conversation, data_type, mask_role = _get_header_conversation_type_mask_role(source, special_tokens)
    header, conversation, mask_role = _get_header_conversation_type_mask_role(source, special_tokens)
    # tokenize conversations
    input_ids = tokenizer.text_to_ids(conversation)
    target = copy.deepcopy(input_ids)
    header_tokens = tokenizer.text_to_ids(header)
    header_len = len(header_tokens)

    ids = []
    tokenized_lens = []
    assert torch.equal(torch.tensor(target[:header_len]), torch.tensor(header_tokens))
    for s in source['conversations']:
        if s["from"] == "User":
            tokenized_sentence = tokenizer.text_to_ids(s["value"])
        else:
            tokenized_sentence = tokenizer.text_to_ids(s["value"])
        ids.append(torch.tensor(tokenized_sentence))
        tokenized_lens.append(len(tokenized_sentence))
    speakers = [sentence["from"] for sentence in source['conversations']]
    assert mask_role in speakers, "mask role not in the conversation"
    is_mask_list = [sentence["is_mask"] for sentence in source['conversations']]
    target = torch.LongTensor(target)
    # not going to train on the header
    target[:header_len] = IGNORE_INDEX
    input_ids = torch.LongTensor(input_ids)
    _mask_targets(
        target,
        tokenized_lens,
        speakers,
        header_len,
        ids,
        tokenizer,
        mask_role,
        special_tokens,
        is_mask_list,
    )
    mask = (target != IGNORE_INDEX).bool()
    assert mask.sum().item() != 0, "mask is empty"
    # Choose the last conversation as answer other history are context
    last_ignore_index_pos = torch.nonzero(target == IGNORE_INDEX)[-1].item() + 1
    context_ids = input_ids[:last_ignore_index_pos]
    answer_ids = input_ids[last_ignore_index_pos:]
    return dict(input_ids=input_ids, mask=mask, context_ids=context_ids, answer_ids=answer_ids)


class GPTSFTChatDataset(GPTSFTDataset):
    def _maybe_validate_prompt_template(self):
        pass

    def _build_samples_mapping(self):
        super()._build_samples_mapping()
        assert hasattr(self.tokenizer, "vocab"), "tokenizer should have vocab property, not supported"

    def _process_example(self, example):
        """
        Create an example by concatenating text and answer.
        Truncation is carried out when needed, but it is performed only on the prompt side.
        BOS, EOS, and SEP, are added if specified.
        """
        result = preprocess(
            example,
            self.tokenizer,
            self.special_tokens,
        )

        # store metadata in dataset, in case user may have keys required in the prediction json files
        metadata = {k: v for k, v in example.items() if k not in ['conversations']}
        result['metadata'] = metadata
        if self.output_original_text:
            result['metadata']['conversations'] = example['conversations']

        return result

    def collate_fn(self, batch):
        input_ids = [item['input_ids'][:-1].tolist() for item in batch]
        labels = [item['input_ids'][1:].tolist() for item in batch]
        contexts = [item['context_ids'].tolist() for item in batch]
        answers = [item['answer_ids'].tolist() for item in batch]
        loss_mask = [item['mask'][1:].tolist() for item in batch]
        metadata = [item['metadata'] for item in batch]

        max_length = max(max([len(x) for x in input_ids]), max([len(x) for x in contexts]) + self.tokens_to_generate)
        if max_length > self.max_seq_length:
            # truncate the sequences if it is longer than max_seq_length
            input_ids = [x[: self.max_seq_length] for x in input_ids]
            labels = [x[: self.max_seq_length] for x in labels]
            loss_mask = [x[: self.max_seq_length] for x in loss_mask]
            contexts = [x[: self.max_seq_length] for x in contexts]
            answers = [x[: self.max_seq_length] for x in answers]

        # increase max length to nearest multiple of 4 or 8
        if self.pad_to_max_length:
            max_length = self.max_seq_length
        else:
            max_length = min(self.max_seq_length, self._ceil_to_nearest(max_length, 8))
        assert max_length <= self.max_seq_length

        if not self.get_attention_mask_from_fusion:
            attention_mask = [self._create_attention_mask(max_length) for _ in batch]
            attention_mask = torch.stack(attention_mask)
        position_ids = [list(range(max_length)) for _ in batch]
        position_ids = torch.LongTensor(position_ids)
        input_ids = torch.LongTensor(
            self._collate_item(input_ids, max_length=max_length, pad_id=self.tokenizer.eos_id)
        )
        labels = torch.LongTensor(self._collate_item(labels, max_length=max_length, pad_id=self.tokenizer.eos_id))
        loss_mask = torch.LongTensor(self._collate_item(loss_mask, max_length=max_length, pad_id=0))
        context_lengths = torch.LongTensor([len(x) for x in contexts])
        contexts = torch.LongTensor(self._collate_item(contexts, max_length=max_length, pad_id=self.tokenizer.eos_id))
        answers = torch.LongTensor(self._collate_item(answers, max_length=max_length, pad_id=self.tokenizer.eos_id))

        processed_batch = {
            'tokens': input_ids,
            'labels': labels,
            'loss_mask': loss_mask,
            'position_ids': position_ids,
            'contexts': contexts,
            'context_lengths': context_lengths,
            'answers': answers,
            'metadata': metadata,
        }

        if not self.get_attention_mask_from_fusion:
            processed_batch['attention_mask'] = attention_mask

        return processed_batch
