# train_test_split using sklearn

# split a dataset into train and test sets
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
# Create dataset
X, y = make_blobs(n_samples=1000)
print(f"One point coordinates (X): {X[0]}, One label (y): {y[0]}")
print(f"Before split: X.shape: {X.shape}, y.shape: {y.shape}")
# split into train test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
print(f"After split: X_train.shape: {X_train.shape}, y_train.shape: {y_train.shape}, X_test.shape: {X_test.shape}, y_test.shape: {y_test.shape}")