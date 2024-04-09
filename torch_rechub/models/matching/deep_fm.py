import torch
import numpy as np

class DeepFactorizationMachineModel(torch.nn.Module):
    """
    A PyTorch implementation of DeepFM.
    Reference: H Guo, et al. DeepFM: A Factorization-Machine based Neural Network for CTR Prediction, 2017.
    """

    def __init__(self, sparse_id_dims, sparse_side_dims, dense_dim, embed_dim, embed_dim_side, mlp_dims, dropout):
        super().__init__()
        # Initialize the linear part of the model
        self.linear = FeaturesLinear(sparse_id_dims + sparse_side_dims)
        # Initialize embeddings for ID features
        self.embedding = FeaturesEmbedding(sparse_id_dims, embed_dim)
        # Calculate the output dimension of ID embeddings
        self.embed_output_dim = len(sparse_id_dims) * embed_dim
        # Count the number of ID fields and side information fields
        self.num_fields = len(sparse_id_dims)
        self.num_fields_side = len(sparse_side_dims)
        # Set the dimension for dense features
        self.dense_dim = dense_dim

        # If there are side information fields, initialize embeddings for them
        if self.num_fields_side > 0:
            self.embedding_side = FeaturesEmbedding(sparse_side_dims, embed_dim_side)
            self.embed_output_dim_side = len(sparse_side_dims) * embed_dim_side

        # If there are dense features, initialize an MLP for them
        if self.dense_dim > 0:
            self.mlp_dense = MultiLayerPerceptron(dense_dim, mlp_dims, dropout)
        # Initialize the Factorization Machine part for capturing feature interactions
        self.fm = FactorizationMachine(reduce_sum=True)

    def forward(self, xx):
        """
        Forward pass for the DeepFM model
        :param xx: Long tensor of size ``(batch_size, num_sparse_id_fields + num_sparse_side_fields + dense_dim)``
        """
        # Split the input into sparse ID, sparse side information, and dense features
        x_sparse_id, x_sparse_side, x_dense = xx[:, :self.num_fields], \
            xx[:, self.num_fields:self.num_fields + self.num_fields_side], \
            xx[:, self.num_fields + self.num_fields_side:]
        # Convert the sparse ID features to integers and get embeddings
        x_sparse_id = x_sparse_id.to(torch.int32)
        embed_x_id = self.embedding(x_sparse_id)

        # If there are side information fields, convert them to integers and get embeddings
        if self.num_fields_side > 0:
            x_sparse_side = x_sparse_side.to(torch.int32)
            embed_x_side = self.embedding_side(x_sparse_side)
            # Concatenate the sparse ID and side information embeddings
            x_sparse = torch.cat([x_sparse_id, x_sparse_side], dim=1)
            embed_x = [embed_x_id, embed_x_side]
        else:
            x_sparse = x_sparse_id
            embed_x = embed_x_id

        # Convert dense features to float and compute the linear, FM, and MLP parts
        x_dense = x_dense.to(torch.float32)
        y = self.linear(x_sparse) + self.fm(embed_x)
        if self.dense_dim > 0:
            y += self.mlp_dense(x_dense)
        # Apply sigmoid to the output and squeeze to remove extra dimensions
        return torch.sigmoid(y.squeeze(1))