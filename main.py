from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load dataset
wine = datasets.load_wine()
X = wine.data
y = wine.target

# Scale data
X = StandardScaler().fit(X).transform(X)

# Split data into training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

# Train model
clf = RandomForestClassifier(random_state=1)
clf.fit(X_train, y_train)
