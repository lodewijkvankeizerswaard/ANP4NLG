from anp4nlg.models.np.neural_process import NeuralProcess
from anp4nlg.criterions.neural_process import NeuralProcessCriterion

from fairseq.models import register_model_architecture


@register_model_architecture('neural_process', 'neural_process_lm')
def neural_process_arch(args):
    pass

