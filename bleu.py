"""Perform BLEU scoring for given model as described in the paper"""

from typing import Dict, List

import torch
from fairseq import utils
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.translate.bleu_score import sentence_bleu
from tqdm import tqdm

from anp4nlg.models.np.neural_process import NeuralProcess


def load_data(path: str='wikitext-103/wiki.test.tokens') -> str:
    txt = ''
    with open(path, 'r') as datafile:
        line = datafile.readline()
        while line:
            txt += line.replace('\n', '')
            line = datafile.readline()

    return txt


def load_model(
    model_dir: str='checkpoints/transformer_wikitext-103',
    checkpoint_file: str='checkpoint_sent_len_32.pt',
    data_dir: str='data-bin/wikitext-103'
) -> NeuralProcess:
    lm = NeuralProcess.from_pretrained(
        model_dir,
        checkpoint_file=checkpoint_file,
        data_name_or_path=data_dir
    )

    lm.eval()

    return lm


def data_to_tokens(
    dataset: str, model: NeuralProcess, sentence_length: int=64
) -> torch.Tensor:
    dictionary = model.models[0].decoder.dictionary
    tokens = dictionary.encode_line(
        dataset, append_eos=False, add_if_not_exist=False)
    num_sentences = len(tokens) // sentence_length
    tokens = tokens[:num_sentences * sentence_length]

    return tokens.reshape(num_sentences, sentence_length, 1)


def genereate_sentence_pairs(
    dataset: torch.Tensor, model: NeuralProcess, context_size: int=21,
    sampling_topk: int=20, no_repeat_ngram_size: int=3, tempearture: int=1.5,
    min_len: int=64
) -> List[Dict[str, str]]:
    dictionary = model.models[0].decoder.dictionary

    sentences = []

    def to_string(tokens: torch.Tensor) -> str:
        return dictionary.string(utils.strip_pad(tokens, dictionary.pad()),
                                 bpe_symbol="@@ ")

    for sentence_tokens in tqdm(dataset):
        context_tokens = sentence_tokens[:context_size]
        target_tokens = sentence_tokens[context_size:]

        num_successful = 0
        num_failed = 0

        try:
            pred = model.sample(
                dictionary.string(context_tokens),
                sampling=True, sampling_topk=sampling_topk,
                no_repeat_ngram_size=no_repeat_ngram_size, tempearture=tempearture,
                min_len=min_len)

            sentences.append({
                "ctx": to_string(context_tokens),
                "tgt": to_string(target_tokens),
                "pred": pred[min_len:],
            })

            num_successful += 1
        except RuntimeError:
            num_failed += 1

    print(f'Successful: {num_successful}, failed: {num_failed}, total: {num_successful + num_failed}')

    return sentences


def score(sentences: List[Dict[str, str]]) -> float:
    stop_words = set(stopwords.words('english'))
    stop_words.update(['.', ',', '"', "'", '?', '!', ':', ';', '(', ')',
                       '[', ']', '{', '}', '<', 'unk', '>', "''"])

    def preprocess(sentence):
        tokens = word_tokenize(sentence)
        tokens = [t for t in tokens if t not in stop_words]
        return tokens

    scores = []
    highscore_sentence = ''
    highscore = 0.0

    for sentence in tqdm(sentences):
        reference, hypothesis = preprocess(sentence["tgt"]), preprocess(sentence["pred"])
        score = sentence_bleu([reference], hypothesis, weights=(1, 0))
        scores.append(score)

        if score > highscore:
            highscore = score
            highscore_sentence = sentence

    print('Highest scoring sentence has score', highscore)
    print('Highscore sentence context:', highscore_sentence["ctx"])
    print('Highscore sentence reference:', highscore_sentence["tgt"])
    print('Highscore sentence hypothesis:', highscore_sentence["pred"])

    return sum(scores) / len(scores)


model = load_model()
dataset = data_to_tokens(load_data(), model)
sentence_pairs = genereate_sentence_pairs(dataset, model)
bleu_score = score(sentence_pairs)

print('BLEU:', bleu_score)
