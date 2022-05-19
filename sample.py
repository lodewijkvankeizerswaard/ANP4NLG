import torch
from anp4nlg.models.np.neural_process import NeuralProcess

en_lm = NeuralProcess.from_pretrained(
    'checkpoints/transformer_wikitext-103',
    checkpoint_file='checkpoint_best.pt',
    data_name_or_path='data-bin/wikitext-103'
)

en_lm.eval()

en_lm.sample('Barack Obama', beam=1, sampling=True, sampling_topk=10, temperature=0.8)