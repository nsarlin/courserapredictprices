import scipy.sparse as sp
from sklearn.decomposition import TruncatedSVD

X_train = sp.load_npz("data/processed_smpl/X_train.npz")
X_test = sp.load_npz("data/processed_smpl/X_test.npz")

tsvd = TruncatedSVD(10)
tsvd.fit(X_train)

print(tsvd.explained_variance_ratio_.sum())
# 0.6470229480963912

tsvd = TruncatedSVD(20)
tsvd.fit(X_train)

print(tsvd.explained_variance_ratio_.sum())
# 0.8445977715545548

tsvd = TruncatedSVD(40)
tsvd.fit(X_train)

tsvd.explained_variance_ratio_.sum()
# 0.9138168881828614


X_train_red = tsvd.transform(X_train)
np.save("data/processed_smpl/X_train_red.npy", X_train_red)
X_test_red = tsvd.transform(X_test)
np.save("data/processed_smpl/X_test_red.npy", X_test_red)
