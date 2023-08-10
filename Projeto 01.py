import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, ConfusionMatrixDisplay
from sklearn.model_selection import GridSearchCV, train_test_split


# Carregando a base de dados.
data=pd.read_csv ("spambase.csv", header=None)
val=data.values
att=data.columns
X=val[:,0:57]
y=val[:,57]
print(X.shape)
print(y.shape)


# Definição dos parâmetros a serem avaliados.
param_knn=[{'n_neighbors':[3, 5, 7], 'weights': ['uniform', 'distance']}]

# Separar uma parte dos dados para teste.
X, X_test, y, y_test = train_test_split(X,y, train_size=0.7, random_state=46,stratify=y)

# Separar uma parte dos dados para validação.
X, X_val, y, y_val = train_test_split(X,y, train_size=0.8, random_state=46,stratify=y)

# Definindo modelo (classificador).
clf=KNeighborsClassifier()

# Usando GridSearch para definição dos parâmetros do KNN.
gs=GridSearchCV(clf, param_knn, cv=5)
gs.fit(X_val, y_val)
clf=gs.best_estimator_
print("Melhores parâmetros:", gs.best_params_)

# Treinando o modelo.
clf.fit(X,y)

# Testando o modelo - retorna a classe de cada exemplo de teste.
ypred=clf.predict(X_test)

# Calculando acurácia do modelo (taxa de acerto).
score=clf.score(X_test, y_test)
print("Acurácia: %.5f" % score)

# Calculando a precisão.
precision=precision_score(y_test, ypred, average="macro")
print("Precisão: %.3f" % precision)

# Calculando a revocação.
recall=recall_score(y_test, ypred, average="macro")
print("Precisão: %.3f" % recall)

# Calculando a f1.
f1=f1_score(y_test, ypred, average="macro")
print("Precisão: %.3f" % f1)

# Matriz de confusao.
ma=confusion_matrix(y_test, ypred)
disp=ConfusionMatrixDisplay(ma, display_labels=['no_spam', 'spam'])
disp.plot()