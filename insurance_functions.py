from scipy.stats import zscore
from sklearn.neighbors import NearestNeighbors
from sklearn.experimental import enable_iterative_imputer
import winsound
from sklearn.impute import KNNImputer, IterativeImputer
import matplotlib.pyplot as plt
import random
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix, roc_auc_score
from sklearn.metrics import precision_score, classification_report, roc_curve, plot_roc_curve, ConfusionMatrixDisplay
from sklearn.metrics import auc, precision_recall_curve, plot_precision_recall_curve, average_precision_score
from sklearn.ensemble import BaggingClassifier
import numpy as np
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier


def to_words2(df):
    df_output = df.copy()
    df_output["AGE"].replace({"16-25": "Young", "26-39": "Middle Age", "40-64": "Old", "65+": "Very Old"},
                             inplace = True)
    return df_output


def to_words(df):
    df_output = df.copy()
    df_output["AGE"].replace({"16-25": "Young", "26-39": "Middle Age", "40-64": "Old", "65+": "Very Old"},
                             inplace = True)
    df_output["DRIVING_EXPERIENCE"].replace(
        {"0-9y": "Newbie", "10-19y": "Amateur", "20-29y": "Advanced", "30y+": "Expert"},
        inplace = True)
    df_output["POSTAL_CODE"].replace({32765: "Oviedo", 92101: "San Diego", 21217: "Baltimore", 10238: "New York"},
                                     inplace = True)
    return df_output


def to_numerical(df):
    df_output = df.copy()
    mean1 = (25 + 16.5) / 2
    mean2 = (39 + 26) / 2
    mean3 = (64 + 40) / 2
    mean4 = (100 + 65) / 2
    dist_1_2 = mean2 - mean1
    dist_2_3 = mean3 - mean2
    dist_3_4 = mean4 - mean3
    dist_tot = dist_1_2 + dist_2_3 + dist_3_4
    dist_1_2_norm = dist_1_2 / dist_tot
    dist_2_3_norm = dist_2_3 / dist_tot
    dist_3_4_norm = dist_3_4 / dist_tot
    age1 = 0
    age2 = age1 + dist_1_2_norm
    age3 = age2 + dist_2_3_norm
    age4 = 0.97
    mean1 = (0 + 9) / 2
    mean2 = (10 + 10) / 2
    mean3 = (20 + 29) / 2
    mean4 = (30 + 50) / 2
    dist_1_2 = mean2 - mean1
    dist_2_3 = mean3 - mean2
    dist_3_4 = mean4 - mean3
    dist_tot = dist_1_2 + dist_2_3 + dist_3_4
    dist_1_2_norm = dist_1_2 / dist_tot
    dist_2_3_norm = dist_2_3 / dist_tot
    dist_3_4_norm = dist_3_4 / dist_tot
    exp1 = 0
    exp2 = age1 + dist_1_2_norm
    exp3 = age2 + dist_2_3_norm
    exp4 = 0.9
    df_output["AGE"].replace({"Young": age1, "Middle Age": age2, "Old": age3, "Very Old": age4}, inplace = True)
    df_output["GENDER"] = (df["GENDER"] == 'male').astype(int)
    df_output["RACE"] = (df["RACE"] == 'majority').astype(int)
    df_output["DRIVING_EXPERIENCE"].replace({"Newbie": exp1, "Amateur": exp2, "Advanced": exp3, "Expert": exp4},
                                            inplace = True)
    df_output["EDUCATION"].replace({"none": 0, "high school": 0.5, "university": 1}, inplace = True)
    df_output["INCOME"].replace({"poverty": 0, "working class": 0.333, "middle class": 0.666, "upper class": 1},
                                inplace = True)
    df_output["VEHICLE_YEAR"] = (df["VEHICLE_YEAR"] == 'before 2015').astype(int)
    df_output["VEHICLE_TYPE"] = (df["VEHICLE_TYPE"] == 'sedan').astype(int)
    df_output["POSTAL_CODE_New_York"] = (df["POSTAL_CODE"] == 'New York').astype(int)
    df_output["POSTAL_CODE_Oviedo"] = (df["POSTAL_CODE"] == 'Oviedo').astype(int)
    df_output["POSTAL_CODE_San_Diego"] = (df["POSTAL_CODE"] == 'San Diego').astype(int)
    df_output["POSTAL_CODE_Baltimore"] = (df["POSTAL_CODE"] == 'Baltimore').astype(int)
    df_output.drop('POSTAL_CODE', axis = 1, inplace = True)
    return df_output


def detect_outlier_iqr(df):
    outlier_indices = {}
    print('Number of outlier in each column:')
    for column in df:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        outlier_indices[column] = df[column][(df[column] < (Q1 - 1.5 * IQR)) | (df[column] > (Q3 + 1.5 * IQR))].index
        if len(outlier_indices[column]) > 0:
            print(column, ': ', len(outlier_indices[column]))
    return outlier_indices


def remove_outlier(df, outlier_indices, columns):
    df_output = df.copy()
    for column in columns:
        for index in outlier_indices[column]:
            df_output.drop(index, axis = 0, inplace = True)
    return df_output


def detect_outlier_zscore(df):
    def condition(x):
        return x > 3.0 or x < -3.0

    outlier_indices = {}
    print('Number of outlier in each column:')
    for column in df:
        z_score = zscore(df[column], nan_policy = 'omit')
        outlier_indices[column] = [idx for idx, element in enumerate(z_score) if condition(element)]
        if len(outlier_indices[column]) > 0:
            print(column, ': ', len(outlier_indices[column]))
    return outlier_indices


def detect_outlier_knn(df, k, dist):
    # dist = 1.2
    knn = NearestNeighbors(n_neighbors = k)
    knn.fit(df)
    neighbors_and_distances = knn.kneighbors(df)
    distances = neighbors_and_distances[0]
    neighbors = neighbors_and_distances[1]
    plt.plot(distances.mean(axis = 1))
    outlier_indices = np.where(distances.mean(axis = 1) > dist)
    df_output = df.drop(outlier_indices[0], axis = 0)
    df_output.shape
    return df_output


def to_normal_minmax(df, columns):
    df_output = df.copy()
    df_output[columns] = (df[columns] - df[columns].min()) / (df[columns].max() - df[columns].min())
    return df_output


def to_normal_zscore(df, columns):
    df_output = df.copy()
    df_output[columns] = zscore(df[columns], nan_policy = 'omit')
    return df_output


def plot_missing(df, miss_per, color, edgecolor, height, width):
    # color = 'green', edgecolor = 'black', height = 3, width = 15
    plt.figure(figsize = (width, height))
    percentage = (df.isnull().mean()) * 100
    percentage.sort_values(ascending = False).plot.bar(color = color, edgecolor = edgecolor)
    plt.axhline(y = miss_per, color = 'r', linestyle = '-')

    plt.title('Missing values percentage per column', fontsize = 20, weight = 'bold')

    plt.text(len(df.isnull().sum() / len(df)) / 1.7, miss_per + 2.5,
             f'Columns with more than {miss_per}% missing values', fontsize = 12, color = 'crimson',
             ha = 'left', va = 'top')
    plt.text(len(df.isnull().sum() / len(df)) / 1.7, miss_per - 0.5,
             f'Columns with less than {miss_per}% missing values', fontsize = 12, color = 'green',
             ha = 'left', va = 'top')
    plt.xlabel('Columns', size = 15, weight = 'bold')
    plt.ylabel('Missing values percentage')
    plt.yticks(weight = 'bold')

    return plt.show()


def create_miss(df, col_list, percent):
    df_output = df.copy()
    for col in col_list:
        if col != 'OUTCOME':
            indices = []
            indices = random.sample(range(0, int(10000 * percent)), int(10000 * percent))
            for index in indices:
                df_output.loc[index, [col]] = None
    return df_output


def filling_miss_knn(df, k, columns):
    imputer = KNNImputer(n_neighbors = k, weights = 'distance', metric = 'nan_euclidean')
    imputer.fit(df)
    Xtrans = imputer.transform(df)
    df_output = pd.DataFrame(Xtrans, columns = columns)
    return df_output


def filling_miss_iterative(df, max_iteration, columns):
    imputer = IterativeImputer(max_iter = max_iteration, random_state = 20)
    imputer.fit(df)
    Xtrans = imputer.transform(df)
    df_output = pd.DataFrame(Xtrans, columns = columns)
    return df_output


def filling_miss_median(df, columns):
    df_output = df.copy()
    for column in columns:
        median_0 = df[df['OUTCOME'] == 0][column].median()
        median_1 = df[df['OUTCOME'] == 1][column].median()
        df_output.loc[(df_output[(df_output['OUTCOME'] == 0) & df_output[column].isnull()].index), column] = \
            df[df['OUTCOME'] == 0][column].fillna(median_0)
        df_output.loc[(df_output[(df_output['OUTCOME'] == 1) & df_output[column].isnull()].index), column] = \
            df[df['OUTCOME'] == 1][column].fillna(median_1)
    return df_output

    def rmse(df_real, col_real, df_filled, col_filled):
        return sqrt(sum((df_real[col_real] - df_filled[col_filled]) ** 2 / 1000))


def feature_selection_with_rf(df, impact_percentage):
    x = df.drop("OUTCOME", axis = 1)
    y = df['OUTCOME']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 20)
    rf = RandomForestRegressor(random_state = 20)
    rf.fit(x_train, y_train)
    features = x.columns
    f_i = list(zip(features, rf.feature_importances_))
    f_i = list(filter(lambda x: x[1] > impact_percentage, f_i))
    f_i.sort(key = lambda x: x[1])
    plt.barh([x[0] for x in f_i], [x[1] for x in f_i])
    return f_i


def split_after_balance(df, type, split_size):
    x = df.drop(['OUTCOME'], axis = 1)
    y = df['OUTCOME']
    if type == 'oversampling':
        oversample = RandomOverSampler(random_state = 20)
        x_over, y_over = oversample.fit_resample(x, y)
        x_train, x_test, y_train, y_test = train_test_split(x_over, y_over, test_size = split_size,
                                                            random_state = 20)
    elif type == 'undersampling':
        undersample = RandomUnderSampler(sampling_strategy = 'majority', random_state = 20)
        x_under, y_under = undersample.fit_resample(x, y)
        x_train, x_test, y_train, y_test = train_test_split(x_under, y_under, test_size = split_size,
                                                            random_state = 20)
    elif type == 'smoteenn':
        smote_enn = SMOTEENN(random_state = 20)
        x_smote, y_smote = smote_enn.fit_resample(x, y)
        x_train, x_test, y_train, y_test = train_test_split(x_smote, y_smote, test_size = split_size,
                                                            random_state = 20)
    else:
        print('please input right type.')
    return x_train, x_test, y_train, y_test


def split_before_balance(df, type, split_size):
    x = df.drop(['OUTCOME'], axis = 1)
    y = df['OUTCOME']
    x_train_unbalanced, x_test_unbalanced, y_train_unbalanced, y_test_unbalanced = \
        train_test_split(x, y, test_size = split_size, random_state = 20)
    if type == 'oversampling':
        oversample = RandomOverSampler(random_state = 20)
        x_train_balanced, y_train_balanced = oversample.fit_resample(x_train_unbalanced, y_train_unbalanced)
    elif type == 'undersampling':
        undersample = RandomUnderSampler(sampling_strategy = 'auto', random_state = 20)
        x_train_balanced, y_train_balanced = undersample.fit_resample(x_train_unbalanced, y_train_unbalanced)
    elif type == 'smoteenn':
        smote_enn = SMOTEENN(random_state = 20)
        x_train_balanced, y_train_balanced = smote_enn.fit_resample(x_train_unbalanced, y_train_unbalanced)
    else:
        print('please input right type.')
    x_train = x_train_balanced
    y_train = y_train_balanced
    x_test = x_test_unbalanced
    y_test = y_test_unbalanced
    return x_train, x_test, y_train, y_test


def grid_search(x_train, y_train, model,
                params, scoring_measure, cv_num):
    best = {}
    for type in ['measures', 'params']:
        temp = {}
        for measure in scoring_measure:
            grid = GridSearchCV(estimator = model,
                                param_grid = params,
                                scoring = measure,
                                cv = cv_num,
                                n_jobs = -1)
            grid.fit(x_train, y_train)
            if type == 'measures':
                temp.update({measure: round(100 * grid.best_score_, 2)})
            else:
                temp.update({measure: grid.best_params_})
        best.update({type: temp})
    winsound.Beep(2000, 50)
    return best


def random_search(x_train, y_train, model, params,
                  scoring_measure, cv_num, iteration):
    best = {}
    for type in ['measures', 'params']:
        temp = {}
        for measure in scoring_measure:
            randm_src = RandomizedSearchCV(estimator = model,
                                           param_distributions = params,
                                           scoring = measure, cv = cv_num,
                                           n_iter = iteration, n_jobs = -1)
            randm_src.fit(x_train, y_train)
            if type == 'measures':
                temp.update({measure: round(randm_src.best_score_ * 100, 2)})
            else:
                temp.update({measure: randm_src.best_params_})
        best.update({type: temp})
    winsound.Beep(2000, 50)
    return best


################################## Model Fit ##################################

def model_fit(x_train, y_train, x_test, y_test,
              classifier, scoring_measures):
    classifier.fit(x_train, y_train)
    measures = {}
    for type in ['train', 'test']:
        if type == 'train':
            y_act = y_train
            y_pred = classifier.predict(x_train)
        else:
            y_act = y_test
            y_pred = classifier.predict(x_test)
        temp = {}
        for measure in scoring_measures:
            if measure == 'recall':
                temp.update({measure: round(100 * recall_score(y_act, y_pred), 2)})
            elif measure == 'accuracy':
                temp.update({measure: round(100 * accuracy_score(y_act, y_pred), 2)})
            elif measure == 'f1':
                temp.update({measure: round(100 * f1_score(y_act, y_pred), 2)})
            elif measure == 'precision':
                temp.update({measure: round(100 * precision_score(y_act, y_pred), 2)})
        measures.update({type: temp})
    return measures


def model_fit_plot(x_train, y_train, x_test, y_test, classifier):
    classifier.fit(x_train, y_train)
    y_pred = classifier.predict(x_test)
    y_prob = classifier.predict_proba(x_test)[:, 1]
    cm = confusion_matrix(y_test, y_pred)
    print(classification_report(y_test, y_pred))
    print(f'ROC AUC score: {roc_auc_score(y_test, y_prob):0.2f}')
    print(f'Accuracy Score: {accuracy_score(y_test, y_pred):0.2f}')
    # Visualizing Confusion Matrix
    plt.figure(figsize = (6, 6))
    sns.heatmap(cm, cmap = 'Blues', annot = True, fmt = 'd', linewidths = 5,
                cbar = False, annot_kws = {'fontsize': 15},
                yticklabels = ['No Loan', 'Claimed Loan'],
                xticklabels = ['Predicted No Loan', 'Predicted Claimed Loan'])
    plt.yticks(rotation = 0)
    plt.show()
    # Roc AUC Curve
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_prob)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    sns.set_theme(style = 'white')
    plt.figure(figsize = (6, 6))
    plt.plot(false_positive_rate, true_positive_rate, color = '#b01717',
             label = 'AUC = %0.3f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1], linestyle = '--', color = '#174ab0')
    plt.axis('tight')
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.title('ROC AUC Curve')
    plt.legend()
    plt.show()
    # Precision Recall Curve
    average_precision = average_precision_score(y_test, y_prob)
    disp = plot_precision_recall_curve(classifier, x_test, y_test)
    plt.title('Precision-Recall Curve')
    plt.show()


################################## Decision Tree ##################################

def dt(x_train, y_train, x_test, y_test, criterion, max_depth, min_split, scoring_measures):
    classifier = DecisionTreeClassifier(criterion = criterion, max_depth = max_depth,
                                        min_samples_split = min_split, random_state = 20)
    measures = model_fit(x_train, y_train, x_test, y_test, classifier, scoring_measures)
    return measures


def dt_plot(x_train, y_train, x_test, y_test, criterion, max_depth, min_split):
    classifier = DecisionTreeClassifier(criterion = criterion, max_depth = max_depth,
                                        min_samples_split = min_split, random_state = 20)
    print('--------- Result for Decision Tree on test data ---------')
    print('')
    model_fit_plot(x_train, y_train, x_test, y_test, classifier)


def min_overfit_dt(x_train, y_train, x_test, y_test, search_params, measures, cv, iter, repeat):
    base_measure = measures[0]  # ='recall'
    difference = 100
    measure_value = 0
    for i in range(repeat):
        best = random_search(x_train, y_train, DecisionTreeClassifier(), search_params, measures,
                             cv_num = cv, iteration = iter)
        dt_measures = dt(x_train, y_train, x_test, y_test, best['params'][base_measure]['criterion'],
                         best['params'][base_measure]['max_depth'],
                         best['params'][base_measure]['min_samples_split'], [base_measure])
        temp_dif = abs(dt_measures['train'][base_measure] - dt_measures['test'][base_measure])
        temp_measure = dt_measures['test'][base_measure]
        print(f"\ntry {i:<4}", end = '')
        if (temp_dif < difference) or (temp_dif == difference and temp_measure > measure_value):
            difference = temp_dif
            measure_value = temp_measure
            print(f"{base_measure}: {measure_value}   overfit: {difference:0.3f}   "
                  f"{best['params'][base_measure]['criterion']}  {best['params'][base_measure]['max_depth']}  "
                  f"{best['params'][base_measure]['min_samples_split']}")
            condition = best['params'][base_measure]
            measure = {}
            measure['train'] = dt_measures['train'][base_measure]
            measure['test'] = dt_measures['test'][base_measure]
    return difference, condition, measure


################################## Bagging ##################################

def bagging(x_train, y_train, x_test, y_test, base, max_features, max_samples, estimators, scoring_measures):
    classifier = BaggingClassifier(base_estimator = base, max_features = max_features,
                                   max_samples = max_samples, n_estimators = estimators,
                                   random_state = 20)
    measures = model_fit(x_train, y_train, x_test, y_test, classifier, scoring_measures)
    return measures


def bagging_plot(x_train, y_train, x_test, y_test, base,
                 max_features, max_samples, estimators):
    classifier = BaggingClassifier(base_estimator = base,
                                   max_features = max_features,
                                   max_samples = max_samples,
                                   n_estimators = estimators,
                                   random_state = 20)
    print('--------- Result for Bagging on test data ---------')
    print('')
    model_fit_plot(x_train, y_train, x_test, y_test, classifier)


def min_overfit_bagging(x_train, y_train, x_test, y_test, base, search_params, measures, cv, iter, repeat):
    base_measure = measures[0]  # ='recall'
    difference = 100
    measure_value = 0
    for i in range(repeat):
        best = random_search(x_train, y_train, BaggingClassifier(), search_params, measures,
                             cv_num = cv, iteration = iter)
        bagging_measures = bagging(x_train, y_train, x_test, y_test, base,
                                   best['params'][base_measure]['max_features'],
                                   best['params'][base_measure]['max_samples'],
                                   best['params'][base_measure]['n_estimators'],
                                   [base_measure])
        temp_dif = abs(bagging_measures['train'][base_measure] - bagging_measures['test'][base_measure])
        temp_measure = bagging_measures['test'][base_measure]
        print(f"\ntry {i:<4}", end = '')
        if (temp_dif < difference) or (temp_dif == difference and temp_measure > measure_value):
            difference = temp_dif
            measure_value = temp_measure
            print(f"{base_measure}: {measure_value}   overfit: {difference:0.3f}   "
                  f"max_features: {best['params'][base_measure]['max_features']:<3}  "
                  f"max_samples: {best['params'][base_measure]['max_samples']:0.2f}  "
                  f"n_estimators: {best['params'][base_measure]['n_estimators']}")
            condition = best['params'][base_measure]
            measure = {}
            measure['train'] = bagging_measures['train'][base_measure]
            measure['test'] = bagging_measures['test'][base_measure]
        winsound.Beep(2000, 50)
    return difference, condition, measure


################################## Random Forest ##################################

def rf(x_train, y_train, x_test, y_test, criterion,
       max_depth, max_features, max_samples, min_split,
       estimators, scoring_measures):
    classifier = RandomForestClassifier(criterion = criterion,
                                        max_depth = max_depth,
                                        min_samples_split = min_split,
                                        max_features = max_features,
                                        max_samples = max_samples,
                                        n_estimators = estimators,
                                        random_state = 20)
    measures = model_fit(x_train, y_train, x_test, y_test,
                         classifier, scoring_measures)
    return measures


def rf_plot(x_train, y_train, x_test, y_test,
            criterion, max_depth, max_features,
            max_samples, min_split, estimators):
    classifier = RandomForestClassifier(criterion = criterion,
                                        max_depth = max_depth,
                                        min_samples_split = min_split,
                                        max_features = max_features,
                                        max_samples = max_samples,
                                        n_estimators = estimators,
                                        random_state = 20)
    print('--------- Result for Random Forest on test data ---------')
    print('')
    model_fit_plot(x_train, y_train, x_test, y_test, classifier)


def min_overfit_rf(x_train, y_train, x_test, y_test,
                   search_params, measures, cv, iter, repeat):
    base_measure = measures[0]  # ='recall'
    difference = 100
    measure_value = 0
    for i in range(repeat):
        best = random_search(x_train, y_train, RandomForestClassifier(),
                             search_params, measures, cv_num = cv, iteration = iter)
        rf_measures = rf(x_train, y_train, x_test, y_test,
                         best['params'][base_measure]['criterion'],
                         best['params'][base_measure]['max_depth'],
                         best['params'][base_measure]['max_features'],
                         best['params'][base_measure]['max_samples'],
                         best['params'][base_measure]['min_samples_split'],
                         best['params'][base_measure]['n_estimators'],
                         [base_measure])
        temp_dif = abs(rf_measures['train'][base_measure] - rf_measures['test'][base_measure])
        temp_measure = rf_measures['test'][base_measure]
        print(f"\ntry {i:<4}", end = '')
        if (temp_dif < difference) or (temp_dif == difference and temp_measure > measure_value):
            difference = temp_dif
            measure_value = temp_measure
            print(f"{base_measure}: {measure_value}   overfit: {difference:0.3f}   "
                  f"criterion: {best['params'][base_measure]['criterion']:<10}"
                  f"max_depth: {best['params'][base_measure]['max_depth']:<4}"
                  f"min_samples_split: {best['params'][base_measure]['min_samples_split']:<6}"
                  f"max_features: {best['params'][base_measure]['max_features']:<4}"
                  f"max_samples: {best['params'][base_measure]['max_samples']:0.2f}"
                  f"n_estimators: {best['params'][base_measure]['n_estimators']:<5}"
                  )
            condition = best['params'][base_measure]
            measure = {}
            measure['train'] = rf_measures['train'][base_measure]
            measure['test'] = rf_measures['test'][base_measure]
    return difference, condition, measure


################################## AdaBoost ##################################

def ab(x_train, y_train, x_test, y_test, algorithm,
       learning_rate, estimators, scoring_measures):
    classifier = AdaBoostClassifier(algorithm = algorithm,
                                    learning_rate = learning_rate,
                                    n_estimators = estimators,
                                    random_state = 20)
    measures = model_fit(x_train, y_train, x_test, y_test,
                         classifier, scoring_measures)
    return measures


def ab_plot(x_train, y_train, x_test, y_test,
            algorithm, learning_rate, estimators):
    classifier = AdaBoostClassifier(algorithm = algorithm,
                                    learning_rate = learning_rate,
                                    n_estimators = estimators,
                                    random_state = 20)
    print('--------- Result for AdaBoost on test data ---------')
    print('')
    model_fit_plot(x_train, y_train, x_test, y_test, classifier)


def min_overfit_ab(x_train, y_train, x_test, y_test,
                   search_params, measures, cv, iter, repeat):
    base_measure = measures[0]  # ='recall'
    difference = 100
    measure_value = 0
    for i in range(repeat):
        best = random_search(x_train, y_train, AdaBoostClassifier(), search_params,
                             measures, cv_num = cv, iteration = iter)
        ab_measures = ab(x_train, y_train, x_test, y_test,
                         best['params'][base_measure]['algorithm'],
                         best['params'][base_measure]['learning_rate'],
                         best['params'][base_measure]['n_estimators'],
                         [base_measure])
        temp_dif = abs(ab_measures['train'][base_measure] - ab_measures['test'][base_measure])
        temp_measure = ab_measures['test'][base_measure]
        print(f"\ntry {i:<4}", end = '')
        if (temp_dif < difference) or (temp_dif == difference and temp_measure > measure_value):
            difference = temp_dif
            measure_value = temp_measure
            print(f"{base_measure}: {measure_value}   overfit: {difference:0.3f}   "
                  f"algorithm: {best['params'][base_measure]['algorithm']:<10}"
                  f"learning_rate: {best['params'][base_measure]['learning_rate']:<9}"
                  f"n_estimators: {best['params'][base_measure]['n_estimators']:<6}"
                  )
            condition = best['params'][base_measure]
            measure = {}
            measure['train'] = ab_measures['train'][base_measure]
            measure['test'] = ab_measures['test'][base_measure]
    return difference, condition, measure


################################## K-Nearest Neighbors ##################################

def knn(x_train, y_train, x_test, y_test,
        dist_func, neighbors, scoring_measures):
    classifier = KNeighborsClassifier(metric = dist_func,
                                      n_neighbors = neighbors)
    measures = model_fit(x_train, y_train, x_test, y_test,
                         classifier, scoring_measures)
    return measures


def knn_plot(x_train, y_train, x_test, y_test, dist_func, neighbors):
    classifier = KNeighborsClassifier(metric = dist_func, n_neighbors = neighbors)
    print('--------- Result for KNN on test data ---------')
    print('')
    model_fit_plot(x_train, y_train, x_test, y_test, classifier)


def min_overfit_knn(x_train, y_train, x_test, y_test, search_params, measures, cv, iter, repeat):
    base_measure = measures[0]  # ='recall'
    difference = 100
    measure_value = 0
    for i in range(repeat):
        best = random_search(x_train, y_train, KNeighborsClassifier(), search_params,
                             measures, cv_num = cv, iteration = iter)
        knn_measures = knn(x_train, y_train, x_test, y_test,
                           best['params'][base_measure]['metric'],
                           best['params'][base_measure]['n_neighbors'],
                           [base_measure])
        temp_dif = abs(knn_measures['train'][base_measure] - knn_measures['test'][base_measure])
        temp_measure = knn_measures['test'][base_measure]
        print(f"\ntry {i:<4}", end = '')
        if (temp_dif < difference) or (temp_dif == difference and temp_measure > measure_value):
            difference = temp_dif
            measure_value = temp_measure
            print(f"{base_measure}: {measure_value}   overfit: {difference:0.3f}   "
                  f"metric: {best['params'][base_measure]['metric']:<15}"
                  f"n_neighbors: {best['params'][base_measure]['n_neighbors']:<9}"
                  )
            condition = best['params'][base_measure]
            measure = {}
            measure['train'] = knn_measures['train'][base_measure]
            measure['test'] = knn_measures['test'][base_measure]
    return difference, condition, measure


################################## Support Vector Machine (SVM) ##################################

def svm(x_train, y_train, x_test, y_test,
        C_value, kernel, scoring_measures):
    classifier = SVC(C = C_value, kernel = kernel, random_state = 20, probability = True)
    measures = model_fit(x_train, y_train, x_test, y_test, classifier, scoring_measures)
    return measures


def svm_plot(x_train, y_train, x_test, y_test,
             C_value, kernel):
    classifier = SVC(C = C_value, kernel = kernel,
                     random_state = 20, probability = True)
    print('--------- Result for SVM on test data ---------')
    print('')
    model_fit_plot(x_train, y_train, x_test, y_test, classifier)


def min_overfit_svm(x_train, y_train, x_test, y_test,
                    search_params, measures, cv, iter, repeat):
    base_measure = measures[0]  # ='recall'
    difference = 100
    measure_value = 0
    for i in range(repeat):
        best = random_search(x_train, y_train, SVC(), search_params,
                             measures, cv_num = cv, iteration = iter)
        svm_measures = svm(x_train, y_train, x_test, y_test,
                           best['params'][base_measure]['C'],
                           best['params'][base_measure]['kernel'],
                           [base_measure])
        temp_dif = abs(svm_measures['train'][base_measure] - svm_measures['test'][base_measure])
        temp_measure = svm_measures['test'][base_measure]
        print(f"\ntry {i:<4}", end = '')
        if (temp_dif < difference) or (temp_dif == difference and temp_measure > measure_value):
            difference = temp_dif
            measure_value = temp_measure
            print(f"{base_measure}: {measure_value}   overfit: {difference:0.3f}   "
                  f"C: {best['params'][base_measure]['C']:<15}"
                  f"kernel: {best['params'][base_measure]['kernel']:<9}"
                  )
            condition = best['params'][base_measure]
            measure = {}
            measure['train'] = svm_measures['train'][base_measure]
            measure['test'] = svm_measures['test'][base_measure]
    return difference, condition, measure


################################## Change Plot ##################################

def change_plot(variable, min, max, step, train, test, measure):
    table = pd.DataFrame({variable: np.arange(min, max + step, step),
                          'Train': train, 'Test': test})
    table['Difference'] = table['Train'] - table['Test']
    plt.plot(np.arange(min, max + step, step), train,
             label = 'train ' + measure)
    plt.plot(np.arange(min, max + step, step), test,
             label = 'test ' + measure)
    plt.legend()
    plt.xlabel(variable)
    plt.ylabel(measure)
    return plt, table


def change_plot2(variable, min, max, step, train_1, test_1,
                 measure_1, train_2, test_2, measure_2):
    table = pd.DataFrame({variable: np.arange(min, max + step, step),
                          'Train_' + measure_1: train_1,
                          'Test_' + measure_1: test_1,
                          'Train_' + measure_2: train_2,
                          'Test_' + measure_2: test_2})
    table['Difference'] = table['Train_recall'] - table['Test_recall']
    plt.plot(np.arange(min, max + step, step),
             train_1, label = 'train ' + measure_1)
    plt.plot(np.arange(min, max + step, step),
             test_1, label = 'test ' + measure_1)
    plt.plot(np.arange(min, max + step, step),
             train_2, label = 'train ' + measure_2)
    plt.plot(np.arange(min, max + step, step),
             test_2, label = 'test ' + measure_2)
    plt.legend()
    plt.xlabel(variable)
    plt.ylabel(measure_1 + ', ' + measure_2)
    return plt, table
