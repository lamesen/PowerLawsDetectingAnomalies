import math
import pandas as pd
import numpy as np
import numpy.linalg as la
from scipy.fftpack import fft
from sklearn import manifold
from sklearn.neighbors import NearestNeighbors


def compute_squared_EDM_method1(X):
    # determine dimensions of data matrix X
    m, n = X.shape
    # initialize squared EDM D
    D = np.zeros((n, n))
    # iterate over upper triangle of D
    for i in range(n):
        for j in range(i+1, n):
            D[i,j] = la.norm(X[:,i] - X[:, j])**2
            D[j, i] = D[i, j]
    return D


if __name__ == "__main__":
    train_loc = './tmp/train_prepared.csv'
    train = pd.read_csv(train_loc, dtype={'meter_id': 'object', 'site_id': 'object'})
    train = train.drop_duplicates(['timestamp', 'meter_id'])
    meters = train["meter_id"].unique()
    print(meters)
    cluster_meters = {}

    meters = ['38_9686']

    for meter in meters:
        print(meter)
        subset = train.loc[train['meter_id'] == meter]
        subset['date'] = pd.to_datetime(subset['timestamp']).dt.date
        subset_group = subset.groupby('date').sum().reset_index()
        subset_group['date'] = pd.to_datetime(subset_group['date'])
        subset_group = subset_group.set_index('date')
        print('Number of days: {}'.format(subset_group.shape[0]))

        print('Computing fast Fourier transform')
        y = fft(subset_group['values'].as_matrix())
        x = np.absolute(y)

        print('Computing dissimilarity matrix')
        similarities = compute_squared_EDM_method1(np.asmatrix(x))
        seed = 123

        print('Dimension reduction via multidimensional scaling')
        mds = manifold.MDS(n_components=3, max_iter=1000, eps=1e-9, random_state=seed,
                           dissimilarity="precomputed", n_jobs=1)
        pos = mds.fit(similarities).embedding_

        nmds = manifold.MDS(n_components=3, metric=False, max_iter=1000, eps=1e-12,
                            dissimilarity="precomputed", random_state=seed, n_jobs=1,
                            n_init=1)
        npos = nmds.fit_transform(similarities, init=pos)

        embeddings_df = pd.DataFrame(npos)

        print('Finding kNN distances')
        X = embeddings_df.iloc[:, 0:3]
        nbrs = NearestNeighbors(n_neighbors=15, algorithm='ball_tree').fit(X)
        distances, indices = nbrs.kneighbors(X)

        embeddings_df.index = subset_group.index

        M, d = embeddings_df.shape

        k = int(float(M) ** (1 / float(d)))
        print('k: {}'.format(k))

        X = embeddings_df.iloc[:, 0:3]
        nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(X)
        distances, indices = nbrs.kneighbors(X)

        print('Calculating minimum hypersphere volume')
        R = distances.max(1)

        fhat = k / ((4 * math.pi / 3) * R ** 3)

        print('Calculating anomaly probability based on kNN density')
        prob = (fhat / np.max(fhat))

        subset_group['cluster_anomaly_probability'] = prob
        subset_group['meter_id'] = meter

        cluster_meters[meter] = subset_group[['meter_id', 'cluster_anomaly_probability']].copy()
        print(cluster_meters[meter].head())
        cluster_meters[meter].to_csv('./tmp/{}_cluster_temp.csv'.format(meter))

    df = pd.concat(cluster_meters.values(), ignore_index=True)
    df.to_csv('./tmp/cluster_output.csv')
