import torch

from neural_process import NeuralProcess
from encoder import MLPEncoder
from aggregator import MeanAggregator
from latent_distribution import NormalLatentDistribution
from decoder import MLPDecoder

from datasets import SineData

BATCH_SIZE = 16
X_DIM = 1
Y_DIM = 1
N = 4
M = 2
R_DIM = 20
S_DIM = (20, 2)
H_DIM = 20
Z_DIM = 20

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = SineData(amplitude_range=(-1., 1.),
                   shift_range=(-.5, .5),
                   num_samples=2000)

    x_context = torch.randn((BATCH_SIZE, N, X_DIM))
    y_context = torch.randn((BATCH_SIZE, N, Y_DIM))

    x_target = torch.randn((BATCH_SIZE, M, X_DIM))
    y_target = torch.randn((BATCH_SIZE, M, Y_DIM))

    np = NeuralProcess(MLPEncoder(X_DIM, Y_DIM, R_DIM, H_DIM),
                        MeanAggregator(X_DIM, R_DIM),
                        MLPEncoder(X_DIM, Y_DIM, S_DIM, H_DIM),
                        MeanAggregator(X_DIM, S_DIM),
                        NormalLatentDistribution(Z_DIM, S_DIM),
                        MLPDecoder(X_DIM, R_DIM, Z_DIM, Y_DIM, H_DIM)
    ).to(device)

    np(x_context, y_context, x_target, y_target)

