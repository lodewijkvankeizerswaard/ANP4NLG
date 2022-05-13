from anp4nlg.models.np.neural_process import NeuralProcess
from anp4nlg.models.np.criterion import NeuralProcessCriterion

from fairseq.models import register_model_architecture
from fairseq.criterions import register_criterion


@register_model_architecture('neural_process', 'neural_process_lm')
def nerual_process_arch(args):
    pass


