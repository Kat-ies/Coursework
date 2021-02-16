"""
functions:
training(x_train, y_train, feature_type, time_df, index)

"""

from constants import *


def training(x_train, y_train, feature_type, time_df, index):
    C = 1.0

    # берём классификаторы
    from sklearn import svm

    classifiers = [LogisticRegression(random_state=RANDOM_SEED),
                   DecisionTreeClassifier(random_state=RANDOM_SEED),
                   KNeighborsClassifier(n_neighbors=300),
                   svm.LinearSVC(C=C, max_iter=10000),
                   RandomForestClassifier(random_state=RANDOM_SEED, n_estimators=200),
                   AdaBoostClassifier(n_estimators=100, random_state=RANDOM_SEED),
                   GradientBoostingClassifier(n_estimators=100, random_state=RANDOM_SEED)]

    # обучим классификаторы и сохраним обученные модели
    for i, clf in enumerate(classifiers):
        t0 = time.time()
        clf.fit(x_train, y_train)
        t = time.time()
        time_df.loc[categories[index]][col_list[i]] += (t - t0) / len(x_train)
        joblib.dump(clf, os.path.join(PATH, 'Classificators', col_list[i] + feature_type + '.pkl'))

def prediction(x_test, feature_type, df, time_df, index):
    #загружаем модели
    classifiers = []
    for i, clf_names in enumerate(col_list):
        classifiers.append(joblib.load(os.path.join(PATH, 'Classificators', col_list[i] + feature_type + '.pkl')))

    # проверяем способность прогнозирования после обучения
    for i, clf in enumerate(classifiers):
        t0 = time.time()
        df[col_list[i]] = clf.predict(x_test)
        t = time.time()
        time_df.loc[categories[index]][col_list[i]] += (t - t0)/len(x_test)
