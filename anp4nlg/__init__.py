from anp4nlg.models.np.neural_process import NeuralProcess

from fairseq.models import register_model_architecture


@register_model_architecture('neural_process', 'neural_process_lm')
def nerual_process_arch(args):
    pass
