import numpy as np 

def cal_pairwise_dists(embeddings, squared=False):
    """Compute the 2D matrix of distances between all the embeddings.

    Args:
        embeddings: numpy array of shape (num_samples, embed_dim)
        squared: Boolean. If true, output is the pairwise squared euclidean distance matrix.
                 If false, output is the pairwise euclidean distance matrix.

    Returns:
        pairwise_dist: numpy array of shape (num_samples, num_samples)
    """

    # Get the dot product between all embeddings
    # shape (num_samples, num_samples)
    dot_product = np.matmul(embeddings, embeddings.T)

    # Get squared L2 norm for each embedding. We can just take the diagonal of `dot_product`.
    # This also provides more numerical stability (the diagonal of the result will be exactly 0).
    # shape (num_samples,)
    squared_norm = np.diag(dot_product)

    # Compute the pairwise distance matrix as we have:
    # ||a - b||^2 = ||a||^2  - 2 <a, b> + ||b||^2
    # shape (num_samples, num_samples)
    distances = np.expand_dims(squared_norm, 1) - 2.0 * dot_product + np.expand_dims(squared_norm, 0)

    # Because of computation errors, some distances might be negative so we put everything >= 0.0
    distances = np.maximum(distances, 0.0)

    if not squared:
        # Because the gradient of sqrt is infinite when distances == 0.0 (ex: on the diagonal)
        # we need to add a small epsilon where distances == 0.0
        mask = np.equal(distances, 0.0).astype(np.float64)
        distances = distances + mask * 1e-16

        distances = np.sqrt(distances)

        # Correct the epsilon added: set the distances on the mask to be exactly 0.0
        distances = distances * (1.0 - mask)

    return distances

def l2_normalize(v, axis=1):
    norm = np.linalg.norm(v, axis=axis)
    mask = np.equal(norm, 0.0).astype(np.float64)
    norm = norm + mask * 1e-16      # avoid division-by-zero
    return v / norm[:, np.newaxis]  # add newaxis for correct broadcasting