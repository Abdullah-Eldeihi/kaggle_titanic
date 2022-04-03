import pandas as pd
import numpy as np

train = pd.read_csv("Datasets/EDA.csv")

train = train.drop(['PassengerId', 'Name', 'First Name', 'Last Name', 'Ticket'], axis=1) 

test = pd.read_csv("Datasets/test_cleaned.csv")

X_train = train.drop('Survived', axis=1)
y_train = train['Survived']

test_passengerid = test['PassengerId']
test = test.drop('PassengerId', axis=1)

test['Title'] = test['Title'].str.replace("Dona", "Mrs")

X_train_dummied = pd.get_dummies(X_train, columns=['Sex', 'Embarked', 'Title', 'Maturity'])
test_dummied = pd.get_dummies(test, columns=['Sex', 'Embarked', 'Title', 'Maturity'])

## Random Forest Classifier without any hypertuning.
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

rf = RandomForestClassifier(random_state=42)

# 3-Fold Cross validation
without_hyper = np.mean(cross_val_score(rf, X_train_dummied, y_train, cv=3, scoring='accuracy'))

without_hyper


## Hypertuning with randomizedsearch.
from sklearn.model_selection import RandomizedSearchCV
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}


# Use the random grid to search for best hyperparameters
# First create the base model to tune
rf = RandomForestClassifier()
# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations, and use all available cores
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid,
                               n_iter = 100, cv = 3, verbose=2, 
                               random_state=42, n_jobs = -1, scoring='accuracy')
# Fit the random search model
rf_random.fit(X_train_dummied, y_train)

best_param = rf_random.best_params_

print(best_param)

best_score_rf_ = rf_random.best_score_

print(best_score_rf_)


from sklearn.model_selection import GridSearchCV
## Tweak the best parameters for GridSearchCV
param_grid_tweaked = {
    'bootstrap' : [True],
    'max_depth' : [30, 40, 50, 60],
    'max_features' : ['sqrt'],
    'min_samples_leaf' : [1, 2, 3],
    'min_samples_split' : [4, 5, 6, 7],
    'n_estimators' : [100, 200, 300, 400, 800]
    }

# Create a based model
rf = RandomForestClassifier()
# Instantiate the grid search model
grid_search = GridSearchCV(estimator = rf, param_grid = param_grid_tweaked, 
                          cv = 3, n_jobs = -1, verbose = 2, scoring='accuracy')
grid_search.fit(X_train_dummied, y_train)
best_param_grid_ = grid_search.best_params_
best_score_grid_ = grid_search.best_score_

print(best_param_grid_)
print(best_score_grid_)

## This is the best model for random forest
best_rf_model = RandomForestClassifier(**best_param_grid_)
best_rf_model.fit(X_train_dummied, y_train)


## The following models needs to be scaled.
from sklearn.preprocessing import StandardScaler

ss = StandardScaler()
cols_to_scale = ['Age', 'SibSp', 'Parch', 'Fare', 'Size of Family', 'Fare Per Person']

## X_train_dummied
ss.fit(X_train_dummied[cols_to_scale].values)
transformed = ss.transform(X_train_dummied[cols_to_scale].values) 
X_train_dummied[cols_to_scale] = transformed

## test_dummied 
transformed = ss.transform(test_dummied[cols_to_scale].values)
test_dummied[cols_to_scale] = transformed

## Logistic Regression with Ridge and Lasso penalty because of the high multicolinearity
## between Fare and Fare Per Person, SibSp and Size of Family, Parch and Size of Family

from sklearn.linear_model import LogisticRegression

## base ridge
model_ridge = LogisticRegression()
base_ridge = np.mean(cross_val_score(model_ridge, X_train_dummied, y_train, cv=3, scoring='accuracy'))
base_ridge


# Hypertuing Ridge
model_ridge = LogisticRegression()
solvers = ['newton-cg', 'lbfgs', 'liblinear']
penalty = ['l2']
c_values = [100, 10, 1.0, 0.1, 0.01]
grid = dict(solver=solvers,penalty=penalty,C=c_values)

grid_search_ridge = GridSearchCV(estimator=model_ridge, param_grid=grid, n_jobs=-1, cv=3, scoring='accuracy')
grid_search_ridge.fit(X_train_dummied, y_train)

best_param_ridge_ = grid_search_ridge.best_params_
best_score_ridge_ = grid_search_ridge.best_score_

best_score_ridge_

## Hyptertuning Lasso
model_ridge = LogisticRegression()
solvers = ['liblinear']
penalty = ['l1']
c_values = [100, 10, 1.0, 0.1, 0.01]
grid = dict(solver=solvers,penalty=penalty,C=c_values)

grid_search_lasso = GridSearchCV(estimator=model_ridge, param_grid=grid, n_jobs=-1, cv=3, scoring='accuracy')
grid_search_lasso.fit(X_train_dummied, y_train)

best_param_lasso_ = grid_search_lasso.best_params_
best_score_lasso_ = grid_search_lasso.best_score_

best_score_lasso_

## base SVC
from sklearn.svm import SVC

model_svc = SVC()
base_svc = np.mean(cross_val_score(model_svc, X_train_dummied, y_train, cv=3, scoring='accuracy'))
base_svc

## Hypertuning SVC
param_grid = {'C': [0.1, 1, 10, 100, 1000],
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
              'kernel': ['rbf', 'linear']}
 
model_svc = SVC()
grid_search_svc = GridSearchCV(model_svc, param_grid = param_grid, cv= 3, scoring='accuracy', verbose=3)
 
# fitting the model for grid search
grid_search_svc.fit(X_train_dummied, y_train)


best_param_svc_ = grid_search_svc.best_params_
best_score_svc = grid_search_svc.best_score_

best_score_svc

## Base KNN
from sklearn.neighbors import KNeighborsClassifier
model_knn = KNeighborsClassifier()
base_knn = np.mean(cross_val_score(model_svc, X_train_dummied, y_train, cv=3, scoring='accuracy'))
base_knn


## Hypertuning KNN
leaf_size = list(range(1,50))
n_neighbors = list(range(1,30))
p=[1,2]

hyperparameters = dict(leaf_size=leaf_size, n_neighbors=n_neighbors, p=p)

model_knn = KNeighborsClassifier()
grid_search_knn = GridSearchCV(model_knn, hyperparameters, cv=3, verbose=2, scoring='accuracy')
grid_search_knn.fit(X_train_dummied, y_train)

best_score_knn_ = grid_search_knn.best_score_
best_score_knn_

## MLP
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

input_dim = len(X_train_dummied.columns)
input_dim

## The model was tweaked for a few times to by adding new layers and changing the number
## of neurons until i decided to leave it as the following.
model = Sequential({
    Dense(300, input_dim=input_dim, activation='relu'),
    Dense(150, activation='relu'),
    Dense(100, activation='relu'),
    Dense(1, activation='sigmoid')
    })

## Declare the optimizer and the learning rate
from tensorflow.keras.optimizers import SGD
opt = SGD(lr=0.01)

## Complie the model with loss as binary_crossentropy as there are only two value for
## survived column and the sigmoid activation to make sure it is either 0 or 1
model.compile(loss="binary_crossentropy",
optimizer=opt,
metrics=["accuracy"])

## Implemented early stopping with restore_best_weights = True to stop the model if it gets
## stuck on val_loss and then restors the best weight instead of just taking the last value
from tensorflow.keras.callbacks import EarlyStopping
early_stopping_monitor = EarlyStopping(
    monitor='val_loss',
    patience=10,
    verbose=1,
    mode='min',
    restore_best_weights=True
)

## fit the model for 30 epochs which it doesn't come close to as it usually early stops
## beforehand with a valdiation_split of 0.2 as we don't have the values for y_test
history = model.fit(X_train_dummied.values, y_train.values, batch_size=32,
                    callbacks=[early_stopping_monitor], epochs=70, validation_split=0.2)

## get the max of the list of acc inside the history which will be accurate as we implemented
## restore_best_weights
best_score_mlp_ = max(history.history['acc'])
best_score_mlp_

import matplotlib.pyplot as plt
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


## MLP model has the best accuracy but might be overfitting which we won't be able to
## know since we don't have the test set to accurately see if we are overfitting or not. 
## Regardless I am going to choose this model for the competition and see how it will fare.

best_model = model

def to_import_to_kaggle():
    y_pred = best_model.predict_classes(test_dummied)
    final = pd.DataFrame({
        'PassengerId' : test_passengerid,
        'Survived' : y_pred.flatten()
        })
    final.to_csv("Datasets/final.csv", index=False)
    
to_import_to_kaggle()


