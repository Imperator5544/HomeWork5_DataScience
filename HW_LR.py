import os  # Модуль для взаємодії з операційною системою
import numpy as np  # Бібліотека для роботи з масивами даних
import pandas as pd  # Бібліотека для аналізу та обробки даних у форматі таблиць
import matplotlib.pyplot as plt  # Бібліотека для візуалізації даних
import sns

from sklearn.model_selection import KFold, StratifiedKFold  # Модулі для реалізації різних стратегій крос-валідації
from sklearn.linear_model import LogisticRegression  # Логістична регресія - модель класифікації
from sklearn.metrics import classification_report  # Метрики для оцінки якості класифікації

import warnings
warnings.filterwarnings(action="ignore")

df_train = pd.read_csv(os.path.join(".venv", "mnist_train.csv"))
df_test = pd.read_csv(os.path.join(".venv", "mnist_test.csv"))

print("Train data shape:", df_train.shape)
print("Test data shape:", df_test.shape)



def sample_as_img(sample):
    label = sample[0]
    img = np.reshape(sample[1:], (28, 28))
    return img, label


fig, axes = plt.subplots(2, 5, figsize=(10, 4))
for i, ax in enumerate(axes.flat):
    img, label = sample_as_img(df_train[df_train['label'] == i].iloc[0])
    ax.imshow(img, cmap='binary')
    ax.set_title(f'Label: {label}')
    ax.axis('off')
plt.show()


plt.figure(figsize=(8, 6))
class_counts = df_train['label'].value_counts()
plt.bar(class_counts.index, class_counts.values)
plt.title('Distribution of Classes')
plt.xlabel('Class')
plt.ylabel('Count')
plt.show()


X_train = df_train.iloc[:, 1:].values / 255.0  # Ознаки
y_train = df_train.iloc[:, 0].values  # Мітки класів

X_test = df_test.iloc[:, 1:].values / 255.0
y_test = df_test.iloc[:, 0].values


n_folds = 5


for fold_idx, (train_idxs, valid_idxs) in enumerate(KFold(n_splits=n_folds).split(X_train)):
    x_train, y_train_fold = X_train[train_idxs], y_train[train_idxs]
    x_valid, y_valid_fold = X_train[valid_idxs], y_train[valid_idxs]

    model = LogisticRegression(penalty=None)
    model.fit(x_train, y_train_fold)
    y_pred_fold = model.predict(x_valid)
    report_fold = classification_report(y_valid_fold, y_pred_fold)

    print(f"\n[Fold {fold_idx + 1}/{n_folds}]")
    print(report_fold)


model = LogisticRegression(penalty=None)
model.fit(X_train, y_train)
y_pred_test = model.predict(X_test)
report_test = classification_report(y_test, y_pred_test)
print("\n[Final Test Evaluation]")
print(report_test)
