from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Load dataset
wine = datasets.load_wine()
X = wine.data
y = wine.target

# Scale data
X = StandardScaler().fit(X).transform(X)

# Split data into training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
