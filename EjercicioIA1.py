import numpy as np
from sklearn.neighbors import KNeighborsClassifier

X_train = np.array([[2, 0],
                    [4, 4],
                    [1, 1],
                    [2, 4],
                    [2, 2],
                    [2, 3],
                    [3, 4],
                    [3, 3]])

y_train = np.array([0, 1, 0, 1, 0, 1, 0, 1])

knn = KNeighborsClassifier(n_neighbors=1, metric='manhattan')

knn.fit(X_train, y_train)

case_to_classify = np.array([[2.5, 2.5]])

predicted_class = knn.predict(case_to_classify)

print(f"La clase predicha para el caso (2.5, 2.5) es: {predicted_class[0]}")