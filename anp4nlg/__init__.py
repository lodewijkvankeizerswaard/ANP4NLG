from anp4nlg.models.np.neural_process import NeuralProcess, AttentiveNeuralProcess
from anp4nlg.models.np.criterion import NeuralProcessCriterion

from fairseq.models import register_model_architecture
from fairseq.criterions import register_criterion


@register_model_architecture('neural_process', 'neural_process_lm')
def neural_process_arch(args):
    pass

@register_model_architecture('attentive_neural_process', 'attentive_neural_process_lm')
def attentive_neural_process_arch(args):
    pass
