from sklearn import datasets
from sklearn.preprocessing import StandardScaler

# Load dataset
wine = datasets.load_wine()
X = wine.data
y = wine.target

# Scale data
X = StandardScaler().fit(X).transform(X)
