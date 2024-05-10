import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, plot_confusion_matrix , classification_report

# Generating a synthetic dataset for binary classification
X = np.array([[4,2],[2,4],[2,3],[3,6],[4,4],[9,10],[6,8],[9,5],[8,7],[10,8]])
y = np.array([0,0,0,0,0,1,1,1,1,1])

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=42)

# Initializing SVM classifier with an RBF kernel
svm_classifier = SVC(kernel='rbf', random_state=42)

# Training the classifier
svm_classifier.fit(X_train, y_train)

# Visualizing the decision boundary
def plot_decision_boundary(X, y, clf):
    plt.figure()

    # Plotting data points
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')

    # Plotting decision boundary
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # Creating grid to evaluate model
    xx = np.linspace(xlim[0], xlim[1], 30)
    yy = np.linspace(ylim[0], ylim[1], 30)
    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T
    Z = clf.decision_function(xy).reshape(XX.shape)

    # Plot decision boundary and margins
    ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'])
    ax.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=100, linewidth=1, facecolors='none', edgecolors='k')
    plt.show()

# Plotting decision boundary
plot_decision_boundary(X_test, y_test,svm_classifier)

# Calculating predictions on the test set
y_pred = svm_classifier.predict(X_test)

# Calculating confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# Plotting confusion matrix
plot_confusion_matrix(svm_classifier, X_test, y_test)
plt.title('Confusion Matrix')
plt.show()

# Calculate precision
precision = precision_score(y_test, y_pred)

# Calculate recall
recall = recall_score(y_test, y_pred)

#Classification report
cr = classification_report(y_test,y_pred)

print("Precision:", precision)
print("Recall:",recall)
print("Classification report:\n", cr)
print("confusion matrix:\n", conf_matrix )

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
