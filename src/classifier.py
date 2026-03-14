from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

scaler = StandardScaler()


def train_model(X_train, y_train):
    X_train = scaler.fit_transform(X_train)
    model = SVC(kernel='rbf', C=10, gamma='scale')
    model.fit(X_train, y_train)
    return model


def predict(model, X_test):
    X_test = scaler.transform(X_test)
    return model.predict(X_test)


def evaluate_model(model, X_test, y_test):
    y_pred = predict(model, X_test)
    cm = confusion_matrix(y_test, y_pred)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Matrice de Confusion")
    plt.savefig("confusion_matrix.png")
    plt.show()

    accuracy = (y_pred == y_test).mean()
    print(f"Accuracy: {accuracy * 100:.2f}%")
    return y_pred