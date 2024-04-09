import os
import numpy as np
import torch
import pandas as pd
import pickle
from collections import defaultdict
from sklearn.preprocessing import LabelEncoder
from flask import Flask, request, jsonify
from torch_rechub.models.matching import EmbedSpot
from torch_rechub.utils.match import Annoy

app = Flask(__name__)

class Config:
    model_save_path = "trained_models/embedspot/embedspot_weights.pth"
    save_dir = 'examples/examples/ranking/data/ml-1m/saved/'
    user_cols = ["user_id", "gender", "age", "occupation", "zip"]
    user_embedding_path = 'temp_user/user_embedding.pkl' # for testing purposes
    item_embedding_path = 'item_embeddings/item_embedding_10April_12_20.pkl'  # Assuming .npy extension for numpy array
    user_features_path = 'features/user_features.pkl'
    item_features_path = 'features/item_features.pkl'
    embedding_size = 64
    n_trees = 10  # Number of trees for Annoy index

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

ensure_dir(Config.save_dir)

# Load the user and item features
with open(Config.user_features_path, 'rb') as f:
    user_features = pickle.load(f)

with open(Config.item_features_path, 'rb') as f:
    item_features = pickle.load(f)

with open(Config.item_embedding_path, 'rb') as f:
    item_embeddings = pickle.load(f)

annoy = Annoy(n_trees=100)
annoy.fit(item_embeddings)

# load the trained model
model = EmbedSpot(
    user_features,
    item_features,
    temperature=0.02,
    user_params={
        "dims": [256, 128, 64],
        "activation": 'relu',
    },
    item_params={
        "dims": [256, 128, 64],
        "activation": 'relu',
    }
)

model.load_state_dict(torch.load(Config.model_save_path))
model.eval()


def encode_user_features(data, sparse_features=['user_id', 'gender', 'age', 'occupation', 'zip']):
    feature_max_idx, user_map = {}, {}
    for feature in sparse_features:
        lbe = LabelEncoder()
        data[feature] = lbe.fit_transform(data[feature]) + 1
        feature_max_idx[feature] = data[feature].max() + 1
        user_map = {encode_id + 1: raw_id for encode_id, raw_id in enumerate(lbe.classes_)}
    return data, feature_max_idx, user_map

@app.route('/predict', methods=['POST'])
def retrieval():
    try:
        data = request.json
        k = int(data.get('k', 10))
        user_profile = data['user_profile']  # Expect a list of user features

        user_features_tensor = torch.tensor(user_profile, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            user_embedding = model.user_encoder(user_features_tensor).numpy().flatten()

        # nearest neighbors from Annoy index
        items_idx, items_scores = annoy.query(user_embedding, k)

        # map indices to item IDs (not provided in original code)
        predicted_item_ids = [item_map[idx] for idx in items_idx]

        return jsonify({'recommended_items': predicted_item_ids}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def retrieval_test(user_features_tensor, k=10):

    user_embedding = model.user_encoder(user_features_tensor.unsqueeze(0))

    # convert to numpy and flatten the array
    user_emb_np = user_embedding.detach().numpy().flatten()


    items_idx = annoy_index.get_nns_by_vector(user_emb_np, k, include_distances=False)
    predicted_item_ids = [item_map[idx] for idx in items_idx]
    print({'recommended_items': predicted_item_ids})


with open(Config.user_embedding_path, 'rb') as f:
    user_embeddings = pickle.load(f)

# test code
# print(item_embeddings[0])


# Run the server
if __name__ == '__main__':

    # production mode：can be integrated with more model.
    # TODO: The frond-end API should be better developed to address the user id and historical movie interaction issue.
    # app.run(debug=True)

    print(type(user_embeddings))

    topk = 1000
    # for demo purpose: choose a stochastic user embedding as input
    items_idx, items_scores = annoy.query(v=user_embeddings[19], n=topk)  #the index of topk match items

    print(" the user embedding is: ", user_embeddings[19])
    print("---EmbedSpot Retrieval based on the user embedding, result: ")

    # a dictionary to group item indices by score
    score_to_item_indices = defaultdict(list)
    for idx, score in zip(items_idx, items_scores):
        score_to_item_indices[score].append(idx)

    # print out the item indices grouped by score
    for score, indices in score_to_item_indices.items():
        print(f"User_Item Embedding Similarity Score: {score}")
        print(f"Item Indices with this score: {indices}")
        print()


