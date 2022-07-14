from sklearn.decomposition import PCA
import numpy as np
import faiss

target_feature_dir = 'result/qFeatures.npy'
pca_feature_dir = 'result/qFeatures_pca.npy'


def test_result():
    pca_qFeat = np.load(pca_feature_dir)
    print(pca_qFeat.shape)
    print(pca_qFeat.dtype)


def add_sklearn_pca():
    pca = PCA(n_components=2048)
    qFeat = np.load(target_feature_dir)
    pca_qFeat = pca.fit_transform(qFeat)
    pca_qFeat = pca_qFeat.astype('float32')

    np.save(pca_feature_dir, pca_qFeat)


def add_faiss_pca():
    qFeat = np.load(target_feature_dir).astype('float32')
    # PCA 2048->256
    # also does a random rotation after the reduction (the 4th argument)
    pca_matrix = faiss.PCAMatrix(65536, 2048)
    pca_matrix.train(qFeat)
    assert pca_matrix.is_trained
    tr = pca_matrix.apply_py(qFeat)
    # - the wrapping index
    pca_qFeat = faiss.IndexPreTransform(pca_matrix, qFeat)
    pca_qFeat = pca_qFeat.astype('float32')

    np.save(pca_feature_dir, pca_qFeat)


if __name__ == '__main__':
    add_faiss_pca()
