import numpy as np
import torch
def generate_batch(signals, labels, pos_label_size = 4, neg_label_size = 2):
    num_samples = len(signals)
    unique_label = np.unique(labels)

    pos_label = np.random.choice(unique_label)

    pos_indices = np.where(labels == pos_label)[0]
    chosen_pos_indices = np.random.choice(pos_indices, size=pos_label_size, replace=False)
    pos_signals = signals[chosen_pos_indices]

    neg_indices = np.where(labels != pos_label)[0]
    chosen_neg_indices = np.random.choice(neg_indices, size=neg_label_size, replace=False)
    neg_signals = signals[chosen_neg_indices]
    neg_labels = labels[chosen_neg_indices]

    batch_signals = np.concatenate([pos_signals, neg_signals])
    batch_labels = np.concatenate([np.array([pos_label]*pos_label_size), neg_labels])
    return batch_signals.reshape(-1,1,1000), batch_labels

def pearson_correlation(x, y, eps = 1e-8):
    x_mean = torch.mean(x, dim=-1, keepdim=True)
    y_mean = torch.mean(y, dim=-1, keepdim=True)

    # Center the tensors by subtracting the mean
    x_centered = x - x_mean
    y_centered = y - y_mean

    # Compute the numerator as the sum of element-wise products of centered tensors
    numerator = torch.sum(x_centered * y_centered, dim=-1)

    # Compute the denominator as the product of the L2 norms of the centered tensors
    denominator = torch.sqrt(torch.sum(x_centered ** 2, dim=-1)) * \
                  torch.sqrt(torch.sum(y_centered ** 2, dim=-1))

    # Compute the Pearson correlation coefficient, adding epsilon for numerical stability
    r = numerator / (denominator + eps)

    return r.abs()

def compute_pairwise_distances(pair_feature):
    features_1 = pair_feature[0]
    features_2 = pair_feature[1]
    # Compute Pearson correlation for one pair
    r = pearson_correlation(features_1, features_2)
    distance = 1-r  # Convert correlation to distance
    return distance

def custom_loss(labels, features):
    positive_pairs = []
    negative_pairs = []

    for i, (feat1, label1) in enumerate(zip(features, labels)):
        for j, (feat2, label2) in enumerate(zip(features, labels)):
            if i < j:  # Avoid duplicate pairs and self-pairs
                if label1 == label2:
                    positive_pairs.append(((feat1, feat2), 1)) # positive
                else:
                    negative_pairs.append(((feat1, feat2), 0)) # negative

    num_positive_pairs = len(positive_pairs)
    num_negative_pairs = len(negative_pairs)

    positive_distances = []
    negative_distances = []

    for pair_feature, label in positive_pairs:
        distance = compute_pairwise_distances(pair_feature)
        positive_distances.append(distance)

    for pair_feature, label in negative_pairs:
        distance = compute_pairwise_distances(pair_feature)
        negative_distances.append(distance)


    # Debugging prints
    print("Number of positive pairs:", num_positive_pairs)
    print("Number of negative pairs:", num_negative_pairs)

    # Compute D_P and D_N
    D_P = torch.sum(torch.stack(positive_distances)) / num_positive_pairs
    D_N = torch.sum(torch.stack(negative_distances)) / num_negative_pairs

    lambda_margin = 0.7  # Adjust as needed
    loss = D_P - D_N + lambda_margin
    loss = torch.clamp(loss, min = 0.0)  # Ensure non-negative loss
    return loss, D_P, D_N