# from anp4nlg.models.np.neural_process import NeuralProcess, AttentiveNeuralProcess
# from anp4nlg.models.np.criterion import NeuralProcessCriterion
from anp4nlg.models.np_test.neural_process import NeuralProcess
from anp4nlg.criterions.neural_process import NeuralProcessCriterion

from fairseq.models import register_model_architecture


@register_model_architecture('neural_process', 'neural_process_lm')
def neural_process_arch(args):
    pass

# @register_model_architecture('attentive_neural_process', 'attentive_neural_process_lm')
# def attentive_neural_process_arch(args):
#     pass
