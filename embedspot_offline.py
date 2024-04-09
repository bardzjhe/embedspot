import os
import collections
import numpy as np
import pickle
import pandas as pd
import torch
import time
from scipy.spatial.distance import cdist
from sklearn.preprocessing import LabelEncoder
from torch_rechub.models.matching import EmbedSpot
from torch_rechub.utils.match import Annoy, Milvus
from torch_rechub.basic.metric import topk_metrics
from sklearn.metrics.pairwise import cosine_similarity
from torch_rechub.trainers import MatchTrainer
from torch_rechub.utils.match import generate_seq_feature_match, gen_model_input
from torch_rechub.basic.features import SparseFeature, SequenceFeature
from torch_rechub.utils.data import MatchDataGenerator, df_to_dict

# Configurations
class Config:
    user_train = "data/train_user.pkl"
    user_test = "data/test_user.pkl"
    all_item = "data/all_item.pkl"
    file_path = "examples/matching/data/ml-1m/ml-1m.csv"
    save_dir = 'examples/examples/ranking/data/ml-1m/saved/'
    model_save_path = "trained_models/embedspot/embedspot_weights.pth"
    user_embedding_path = 'temp_user/user_embedding.pkl'
    item_embedding_path = 'item_embeddings/item_embedding_10April_12_20.pkl'
    user_features_path = 'features/user_features.pkl'
    item_features_path = 'features/item_features.pkl'
    user_cols = ["user_id", "gender", "age", "occupation", "zip"]
    item_cols = ['movie_id', "cate_id"]
    sparse_features = ['user_id', 'movie_id', 'gender', 'age', 'occupation', 'zip', "cate_id"]
    user_col = 'user_id'
    item_col = 'movie_id'
    seq_max_len = 50
    batch_size = 256
    embedding_dim = 16
    lr = 1e-4
    weight_decay = 1e-6

# Ensure save directory exists
def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

# Load and preprocess data
def load_data(file_path):
    data = pd.read_csv(file_path)
    data["cate_id"] = data["genres"].apply(lambda x: x.split("|")[0])
    data = data[data["rating"] >= 3]  # This line removes rows with ratings below 3
    # Reset the index of the filtered DataFrame
    data.reset_index(drop=True, inplace=True)
    return data

# Label encoding
def encode_features(data, sparse_features, user_col, item_col):
    feature_max_idx, user_map, item_map = {}, {}, {}
    for feature in sparse_features:
        lbe = LabelEncoder()
        data[feature] = lbe.fit_transform(data[feature]) + 1
        feature_max_idx[feature] = data[feature].max() + 1
        if feature == user_col:
            user_map = {encode_id + 1: raw_id for encode_id, raw_id in enumerate(lbe.classes_)}
        if feature == item_col:
            item_map = {encode_id + 1: raw_id for encode_id, raw_id in enumerate(lbe.classes_)}
    return data, feature_max_idx, user_map, item_map


# Linear scan
def match_evaluation_linear(user_embedding, item_embedding, test_user, all_item, user_col='user_id', item_col='movie_id',
                     raw_id_maps="./data/ml-1m/saved/raw_id_maps.npy", topk=[10, 50]):
    start_time = time.time()  # Start timing
    print("evaluate embedding matching on test data")

    user_map, item_map = np.load(raw_id_maps, allow_pickle=True)
    match_res = collections.defaultdict(dict)  # user id -> predicted item ids

    print("matching for topk")

    print("user shape", user_embedding.detach().cpu().numpy().shape)

    user_emb_np = user_embedding.detach().cpu().numpy()[0,:]  # Call .detach() if it requires grad
    print("user_emb_np.shape", user_emb_np.shape)
    # Assuming `item_embedding` is the PyTorch tensor you've shown,
    # convert the item_embedding tensor to a numpy array
    item_embedding_np = item_embedding.detach().cpu().numpy()
    print("item_embedding_np.shape", item_embedding_np.shape)
    for i in topk:
        for user_id, user_emb in zip(test_user[user_col], user_embedding):
            # Calculate cosine similarity between user and all items
            scores = cosine_similarity([user_emb_np], item_embedding_np)[0]
            # Get the indices of items with highest similarity scores
            items_idx = np.argsort(scores)[-i:]
            # Map the indices to actual item ids
            match_res[user_map[user_id]] = np.vectorize(item_map.get)(all_item[item_col][items_idx])

    # get ground truth
    print("generate ground truth")

    data = pd.DataFrame({user_col: test_user[user_col], item_col: test_user[item_col]})
    data[user_col] = data[user_col].map(user_map)
    data[item_col] = data[item_col].map(item_map)
    user_pos_item = data.groupby(user_col).agg(list).reset_index()
    ground_truth = dict(zip(user_pos_item[user_col], user_pos_item[item_col]))  # user id -> ground truth items

    print("compute topk metrics")
    # Assuming topk_metrics is a function that computes the evaluation metrics
    out = topk_metrics(y_true=ground_truth, y_pred=match_res, topKs=topk)
    end_time = time.time()  # End timing
    time_elapsed = end_time - start_time
    print("Linear scan time elapsed: {:.2f} seconds".format(time_elapsed))
    return out

# annoy
def match_evaluation_annoy(user_embedding, item_embedding, test_user, all_item, user_col='user_id', item_col='movie_id',
                     raw_id_maps="./data/ml-1m/saved/raw_id_maps.npy", topk=[10, 50]):
    start_time = time.time()  # Start timing
    print("evaluate embedding matching on test data")
    annoy = Annoy(n_trees=10)
    annoy.fit(item_embedding)

    #for each user of test dataset, get ann search topk result
    print("matching for topk")
    user_map, item_map = np.load(raw_id_maps, allow_pickle=True)
    match_res = collections.defaultdict(dict)  # user id -> predicted item ids
    for i in topk:
        for user_id, user_emb in zip(test_user[user_col], user_embedding):
            items_idx, items_scores = annoy.query(v=user_emb, n=i)  #the index of topk match items
            match_res[user_map[user_id]] = np.vectorize(item_map.get)(all_item[item_col][items_idx])

    #get ground truth
    print("generate ground truth")

    data = pd.DataFrame({user_col: test_user[user_col], item_col: test_user[item_col]})
    data[user_col] = data[user_col].map(user_map)
    data[item_col] = data[item_col].map(item_map)
    user_pos_item = data.groupby(user_col).agg(list).reset_index()
    ground_truth = dict(zip(user_pos_item[user_col], user_pos_item[item_col]))  # user id -> ground truth

    print("compute topk metrics")
    out = topk_metrics(y_true=ground_truth, y_pred=match_res, topKs=topk)
    end_time = time.time()  # End timing
    time_elapsed = end_time - start_time
    print("Annoy time elapsed: {:.2f} seconds".format(time_elapsed))
    return out


# milvus vector database
def match_evaluation_milvus(user_embedding, item_embedding, test_user, all_item, user_col='user_id', item_col='movie_id',
                     raw_id_maps="./data/ml-1m/saved/raw_id_maps.npy", topk=[10, 50]):
    print("evaluate embedding matching on test data")

    milvus = Milvus(dim=64)
    milvus.fit(item_embedding)

    # for each user of test dataset, get ann search topk result
    print("matching for topk")
    user_map, item_map = np.load(raw_id_maps, allow_pickle=True)
    match_res = collections.defaultdict(dict)  # user id -> predicted item ids
    for user_id, user_emb in zip(test_user[user_col], user_embedding):
        items_idx, items_scores = milvus.query(v=user_emb, n=topk)  # the index of topk match items
        match_res[user_map[user_id]] = np.vectorize(item_map.get)(all_item[item_col][items_idx])

    # get ground truth
    print("generate ground truth")

    data = pd.DataFrame({user_col: test_user[user_col], item_col: test_user[item_col]})
    data[user_col] = data[user_col].map(user_map)
    data[item_col] = data[item_col].map(item_map)
    user_pos_item = data.groupby(user_col).agg(list).reset_index()
    ground_truth = dict(zip(user_pos_item[user_col], user_pos_item[item_col]))  # user id -> ground truth

    print("compute topk metrics")
    out = topk_metrics(y_true=ground_truth, y_pred=match_res, topKs=topk)
    return out


# Main code
def main():
    # Set options for display and pytorch seeds
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)
    torch.manual_seed(2022)

    # Initialize Config and necessary directories
    cfg = Config()
    ensure_dir(cfg.save_dir)

    # Load the data
    data = load_data(cfg.file_path)

    # Encode features
    data, feature_max_idx, user_map, item_map = encode_features(data, cfg.sparse_features, cfg.user_col, cfg.item_col)


    # test code
    # print(data[cfg.user_cols][:120])

    # Save mappings
    np.save(os.path.join(cfg.save_dir, "raw_id_maps.npy"), (user_map, item_map))

    # Define user and item profiles  进塔之前的feature
    user_profile = data[cfg.user_cols].drop_duplicates(cfg.user_col)
    item_profile = data[cfg.item_cols].drop_duplicates(cfg.item_col)

    print(user_profile)
    # print(type(user_profile))
    # assert 1==0

    # test code
    # print(user_profile[:50])

    # Generate train and test data

    # print(data[:20])

    df_train, df_test = generate_seq_feature_match(data,
                                                   cfg.user_col,
                                                   cfg.item_col,
                                                   time_col="timestamp",
                                                   item_attribute_cols=[],
                                                   sample_method=1,
                                                   mode=0,
                                                   neg_ratio=3,
                                                   min_item=3)


    # test code
    # print(df_train[:30])
    # print("test")
    # print(df_test[:1])

    # Generate model input data
    x_train = gen_model_input(df_train, user_profile, cfg.user_col, item_profile, cfg.item_col, seq_max_len=cfg.seq_max_len)
    y_train = x_train["label"]
    x_test = gen_model_input(df_test, user_profile, cfg.user_col, item_profile, cfg.item_col, seq_max_len=cfg.seq_max_len)
    y_test = x_test["label"]
    print({k: v[:3] for k, v in x_train.items()})

    with open(cfg.user_train, "wb") as f:
        pickle.dump(x_train, f)

    with open(cfg.user_test, "wb") as f:
        pickle.dump(x_test, f)



    print(cfg.user_cols)

    user_features = [
        SparseFeature(feature_name, vocab_size=feature_max_idx[feature_name], embed_dim=16) for feature_name in
        cfg.user_cols
    ]

    print(user_features)

    user_features += [
        SequenceFeature("hist_movie_id",
                        vocab_size=feature_max_idx["movie_id"],
                        embed_dim=16,
                        pooling="mean",
                        shared_with="movie_id")
    ]

    # print(SequenceFeature("hist_movie_id",
    #                     vocab_size=feature_max_idx["movie_id"],
    #                     embed_dim=16,
    #                     pooling="mean",
    #                     shared_with="movie_id"))
    #
    # assert 1==0

    item_features = [
        SparseFeature(feature_name, vocab_size=feature_max_idx[feature_name], embed_dim=16) for feature_name in
        cfg.item_cols
    ]

    with open(cfg.user_features_path, "wb") as f:
        pickle.dump(user_features, f)

    with open(cfg.item_features_path, "wb") as f:
        pickle.dump(item_features, f)

    print(item_features)

    # print(user_features)
    # print(item_features)
    # print(neg_item_features)

    all_item = df_to_dict(item_profile)
    with open(cfg.all_item, "wb") as f:
        pickle.dump(all_item, f)

    assert 1==0
    print(all_item)

    test_user = x_test

    print(type(x_train))
    assert 1==0
    print(type(test_user))
    print({k: v[:3] for k, v in all_item.items()})
    print({k: v[0] for k, v in test_user.items()})

    dg = MatchDataGenerator(x=x_train, y=y_train)
    train_dl, test_dl, item_dl = dg.generate_dataloader(test_user, all_item, batch_size=256)

    # Model
    model = EmbedSpot(user_features,
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

    # 模型训练器
    trainer = MatchTrainer(model,
                           mode=0,  # 同上面的mode，需保持一致
                           optimizer_params={
                               "lr": 1e-4,
                               "weight_decay": 1e-6
                           },
                           n_epoch=2,
                           device='cpu',
                           model_path=cfg.save_dir)

    # training starts
    trainer.fit(train_dl)

    torch.save(model.state_dict(), cfg.model_save_path)

    user_embedding = trainer.inference_embedding(model=model, mode="user", data_loader=test_dl, model_path=cfg.save_dir)
    item_embedding = trainer.inference_embedding(model=model, mode="item", data_loader=item_dl, model_path=cfg.save_dir)

    print("offline batch training finished")
    print("item embeddings saved")

    with open(cfg.item_embedding_path, "wb") as f:
        pickle.dump(item_embedding, f)

    with open(cfg.user_embedding_path, "wb") as f:
        pickle.dump(user_embedding, f)

    # Efficiency experiments
    # annoy
    result = match_evaluation_annoy(user_embedding, item_embedding, test_user, all_item, topk=[200],
                     raw_id_maps=cfg.save_dir + "raw_id_maps.npy")
    # milvus
    # result = match_evaluation_milvus(user_embedding, item_embedding, test_user, all_item, topk=[10, 50],
    #                                 raw_id_maps=cfg.save_dir + "raw_id_maps.npy")
    print(result)
    # linear scan
    # result = match_evaluation_linear(user_embedding, item_embedding, test_user, all_item, topk=[10, 50],
    #                                  raw_id_maps=cfg.save_dir + "raw_id_maps.npy")
    print(result)

if __name__ == "__main__":
    main()


