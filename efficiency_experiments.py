import torch
import pickle
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch_rechub.utils.match import Annoy, Milvus
import collections
from torch_rechub.basic.metric import topk_metrics
from sklearn.metrics.pairwise import cosine_similarity

with open('temp_user/user_embedding.pkl', 'rb') as f:
    user_embedding = pickle.load(f)

with open('item_embeddings/item_embedding_10April_12_20.pkl', 'rb') as f:
    item_embedding = pickle.load(f)

with open("data/test_user.pkl", 'rb') as f:
    test_user = pickle.load(f)

with open("data/all_item.pkl", 'rb') as f:
    all_item = pickle.load(f)


# test_code
# print(type(test_user))
# test_user = test_user[:100]
# for key, value in test_user.items():
#     # Print the key and the length of its value
#     print(f"Length of value for key '{key}': {len(value)}")
# assert 1==0
# print(test_user[:100])

# print(len(test_user))
# assert 1==0
user_col='user_id'
item_col='movie_id'
# Linear scan
def evaluate_linear(num_users = 100, raw_id_maps="./data/ml-1m/saved/raw_id_maps.npy", topk=[10, 50]):

    start_time = time.time()  # Start timing

    user_map, item_map = np.load(raw_id_maps, allow_pickle=True)
    match_res = collections.defaultdict(dict)  # user id -> predicted item ids

    # print("user shape", user_embedding.detach().cpu().numpy().shape)

    user_emb_np = user_embedding.detach().cpu().numpy()[0,:]  # Call .detach() if it requires grad
    # print("user_emb_np.shape", user_emb_np.shape)
    # Assuming `item_embedding` is the PyTorch tensor you've shown,
    # convert the item_embedding tensor to a numpy array
    item_embedding_np = item_embedding.detach().cpu().numpy()
    # print("item_embedding_np.shape", item_embedding_np.shape)
    for i in topk:
        for user_id, user_emb in zip(test_user[user_col][:num_users], user_embedding):
            # Calculate cosine similarity between user and all items
            scores = cosine_similarity([user_emb_np], item_embedding_np)[0]
            # Get the indices of items with highest similarity scores
            items_idx = np.argsort(scores)[-i:]
            # Map the indices to actual item ids
            match_res[user_map[user_id]] = np.vectorize(item_map.get)(all_item[item_col][items_idx])

    # get ground truth
    # print("generate ground truth")

    data = pd.DataFrame({user_col: test_user[user_col], item_col: test_user[item_col]})
    data[user_col] = data[user_col].map(user_map)
    data[item_col] = data[item_col].map(item_map)
    user_pos_item = data.groupby(user_col).agg(list).reset_index()
    ground_truth = dict(zip(user_pos_item[user_col], user_pos_item[item_col]))  # user id -> ground truth items

    # print("compute topk metrics")
    # Assuming topk_metrics is a function that computes the evaluation metrics
    # out = topk_metrics(y_true=ground_truth, y_pred=match_res, topKs=topk)
    # print(out)
    end_time = time.time()  # End timing
    time_elapsed = end_time - start_time
    print("Linear scan time elapsed: {:.2f} seconds".format(time_elapsed))
    return time_elapsed


# annoy
def evaluate_annoy(num_users = 100, user_col='user_id', item_col='movie_id',
                     raw_id_maps="./data/ml-1m/saved/raw_id_maps.npy", topk=[10, 50]):
    start_time = time.time()  # Start timing
    annoy = Annoy(n_trees=10)
    annoy.fit(item_embedding)

    #for each user of test dataset, get ann search topk result
    user_map, item_map = np.load(raw_id_maps, allow_pickle=True)
    match_res = collections.defaultdict(dict)  # user id -> predicted item ids
    for i in topk:
        for user_id, user_emb in zip(test_user[user_col][:num_users], user_embedding):
            items_idx, items_scores = annoy.query(v=user_emb, n=i)  #the index of topk match items
            match_res[user_map[user_id]] = np.vectorize(item_map.get)(all_item[item_col][items_idx])

    #get ground truth
    # print("generate ground truth")

    data = pd.DataFrame({user_col: test_user[user_col], item_col: test_user[item_col]})
    data[user_col] = data[user_col].map(user_map)
    data[item_col] = data[item_col].map(item_map)
    user_pos_item = data.groupby(user_col).agg(list).reset_index()
    ground_truth = dict(zip(user_pos_item[user_col], user_pos_item[item_col]))  # user id -> ground truth

    # print("compute topk metrics")
    # out = topk_metrics(y_true=ground_truth, y_pred=match_res, topKs=topk)
    end_time = time.time()  # End timing
    time_elapsed = end_time - start_time
    print("Annoy time elapsed: {:.2f} seconds".format(time_elapsed))
    return time_elapsed


# milvus
def evaluate_milvus(num_users, raw_id_maps="./data/ml-1m/saved/raw_id_maps.npy", topk=[10, 50]):
    print("evaluate embedding matching on test data")
    start_time = time.time()  # Start timing
    milvus = Milvus(dim=64)
    milvus.fit(item_embedding)

    # for each user of test dataset, get ann search topk result
    # print("matching for topk")
    user_map, item_map = np.load(raw_id_maps, allow_pickle=True)
    match_res = collections.defaultdict(dict)  # user id -> predicted item ids
    for user_id, user_emb in zip(test_user[user_col][:num_users], user_embedding):
        items_idx, items_scores = milvus.query(v=user_emb, n=topk)  # the index of topk match items
        match_res[user_map[user_id]] = np.vectorize(item_map.get)(all_item[item_col][items_idx])

    # get ground truth
    # print("generate ground truth")

    data = pd.DataFrame({user_col: test_user[user_col], item_col: test_user[item_col]})
    data[user_col] = data[user_col].map(user_map)
    data[item_col] = data[item_col].map(item_map)
    user_pos_item = data.groupby(user_col).agg(list).reset_index()
    ground_truth = dict(zip(user_pos_item[user_col], user_pos_item[item_col]))  # user id -> ground truth

    end_time = time.time()  # End timing
    time_elapsed = end_time - start_time
    print("Milvus time elapsed: {:.2f} seconds".format(time_elapsed))
    return time_elapsed


if __name__ == '__main__':
    batch_sizes = [100, 1000, 2000]
    plt.figure(figsize=(10, 5))  # Optional: Define a larger figure size

    # Run the tests for linear scan
    average_times_linear = []
    for batch_size in batch_sizes:
        elapsed_times = []
        for _ in range(5):  # Repeat the test 5 times for each batch size
            elapsed_time = evaluate_linear(num_users=batch_size)
            elapsed_times.append(elapsed_time)
        print("---")
        average_elapsed_time = sum(elapsed_times) / len(elapsed_times)
        average_times_linear.append(average_elapsed_time)
    plt.plot(batch_sizes, average_times_linear, marker='o', label='Linear Scan')

    # Run the tests for Annoy
    average_times_annoy = []
    for batch_size in batch_sizes:
        elapsed_times = []
        for _ in range(5):  # Repeat the test 5 times for each batch size
            elapsed_time = evaluate_annoy(num_users=batch_size)
            elapsed_times.append(elapsed_time)
        print("---")
        average_elapsed_time = sum(elapsed_times) / len(elapsed_times)
        average_times_annoy.append(average_elapsed_time)
    plt.plot(batch_sizes, average_times_annoy, marker='s', label='Annoy')

    # Run the tests for Milvus
    average_times_milvus = []
    for batch_size in batch_sizes:
        elapsed_times = []
        for _ in range(5):  # Repeat the test 5 times for each batch size
            elapsed_time = evaluate_milvus(num_users=batch_size)
            elapsed_times.append(elapsed_time)
        print("---")
        average_elapsed_time = sum(elapsed_times) / len(elapsed_times)
        average_times_milvus.append(average_elapsed_time)
    plt.plot(batch_sizes, average_times_milvus, marker='p', label='Annoy')

    # Adding labels and title
    plt.xlabel('Number of Users')
    plt.ylabel('Average Time Elapsed (seconds)')
    plt.title('Time Elapsed vs Number of Users')

    # Adding a legend to distinguish the lines
    plt.legend()

    # Display the combined plot
    plt.show()