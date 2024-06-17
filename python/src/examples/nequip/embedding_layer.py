import torch
import torch.nn as nn

from gelib import SO3partArr, SO3vecArr

from src.examples.common.point_cloud_factory import PointCloudFactory

class AtomicNumberEmbedding(nn.Module):
    """Embeds atomic numbers into initial scalar features (l = 0)."""

    def __init__(self, num_species: int, embedding_dim: int, max_l):
        super().__init__()

        self.embedding_dim_ = embedding_dim
        self.embedding_ = nn.Embedding(num_embeddings = num_species,
                                      embedding_dim = embedding_dim) 

    def forward(self, atomic_numbers : torch.Tensor) -> SO3vecArr:
        """Embeds atomic numbers.

        Args:
            atomic_numbers: Tensor of shape [B,...,N] (batch, ..., atoms)
            with atoms containing atomic numbers.

        Returns:
            PointCloud of shape [B, ..., embedding_dim, 2, 1, 1, N]
            (batch, ..., embedding length, parity, channels, l = 0, atoms)
        """
        assert atomic_numbers.dim() >= 2, atomic_numbers.size()

        # Embed atomic numbers into a feature vector. Results in shape:
        # [(B * ...), N, embedding_dim]
        old_shape = atomic_numbers.size()
        atomic_numbers.reshape(-1, old_shape[-1])
        features = self.embedding_.forward(atomic_numbers)

        # Reshape for SO3partArr (l = 0, even parity). New shape will be:
        # [B, ..., embedding_dim, 1, 1, 1, N]
        new_shape = old_shape + (features.size()[-1],)
        features = features.reshape(new_shape).transpose(-2, -1)
        features = features.unsqueeze(-2).unsqueeze(-2).unsqueeze(-2)

        # Stack with a zero vector for odd parity.
        features = torch.stack(features, torch.zeros(features.size()), dim = -4)
        features = SO3vecArr.from_part(SO3partArr(features))

        # Throw it all into a point cloud
        #
        # TODO: Where do positions come from?
        point_cloud = PointCloudFactory.CreatePointCloud(positions, features)
        
        return point_cloud