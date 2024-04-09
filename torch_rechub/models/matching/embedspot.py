import torch
import torch.nn.functional as F
from ...basic.layers import MLP, EmbeddingLayer


class EmbedSpot(torch.nn.Module):
    """EmbedSpot

    Arg
        user_features (list[Feature Class]): training by the user encoder module.
        item_features (list[Feature Class]): training by the item encoder module.
        temperature (float): temperature factor for similarity score, default to 1.0.
        user_params (dict): the params of the User Tower module, keys include:`{"dims":list, "activation":str, "dropout":float, "output_layer":bool`}.
        item_params (dict): the params of the Item Tower module, keys include:`{"dims":list, "activation":str, "dropout":float, "output_layer":bool`}.
    """

    def __init__(self, user_features, item_features, user_params, item_params, temperature=1.0):
        super().__init__()
        self.user_features = user_features
        self.item_features = item_features
        self.temperature = temperature
        self.user_dims = sum([fea.embed_dim for fea in user_features])
        self.item_dims = sum([fea.embed_dim for fea in item_features])

        self.embedding = EmbeddingLayer(user_features + item_features)
        self.user_mlp = MLP(self.user_dims, output_layer=False, **user_params)
        self.item_mlp = MLP(self.item_dims, output_layer=False, **item_params)
        self.mode = None

    def forward(self, x):
        user_embedding = self.user_encoder(x)
        item_embedding = self.item_encoder(x)
        if self.mode == "user":
            return user_embedding
        if self.mode == "item":
            return item_embedding

        # calculate cosine score
        y = torch.mul(user_embedding, item_embedding).sum(dim=1)
        # y = y / self.temperature
        return torch.sigmoid(y)

    def user_encoder(self, x):
        if self.mode == "item":
            return None
        input_user = self.embedding(x, self.user_features, squeeze_dim=True)  #[batch_size, num_features*deep_dims]
        user_embedding = self.user_mlp(input_user)  #[batch_size, user_params["dims"][-1]]
        user_embedding = F.normalize(user_embedding, p=2, dim=1)  # L2 normalize
        return user_embedding

    def item_encoder(self, x):
        if self.mode == "user":
            return None
        input_item = self.embedding(x, self.item_features, squeeze_dim=True)  #[batch_size, num_features*embed_dim]
        item_embedding = self.item_mlp(input_item)  #[batch_size, item_params["dims"][-1]]
        item_embedding = F.normalize(item_embedding, p=2, dim=1)
        return item_embedding