from anp4nlg.models.np.neural_process import NeuralProcess

en_lm = NeuralProcess.from_pretrained(
    'checkpoints/transformer_wikitext-103',
    checkpoint_file='checkpoint_last.pt',
    data_name_or_path='data-bin/wikitext-103'
)

en_lm.eval()

sampled_text = en_lm.sample('The building and the surrounding park', beam=1, sampling=True, sampling_topk=10)

print(sampled_text)