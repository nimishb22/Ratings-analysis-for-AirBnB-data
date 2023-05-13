import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import sklearn.metrics as met
from sklearn.preprocessing import LabelEncoder
import sklearn.naive_bayes as nb
import sklearn.ensemble as en
import sys
import warnings
import missingno as msno
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.svm import LinearSVC
from itertools import cycle
from scipy import stats
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler, label_binarize
from statsmodels.stats.outliers_influence import variance_inflation_factor
from pycm import ConfusionMatrix

# Handling Warning Messages
if not sys.warnoptions:
    warnings.simplefilter("ignore")

# Load Data
airbnb_data = pd.read_csv('./Listings_Clean_All.csv')
# print(airbnb_data.head())

# Remove rows where review score rating is blank
airbnb_data = airbnb_data.dropna(subset='review_scores_rating')

# Add the column Ratings Star based on the Ratings value and populate based on review ratings

airbnb_data.loc[airbnb_data['review_scores_rating'] <= 89, 'Ratings_Star'] = 'Silver'
airbnb_data.loc[airbnb_data['review_scores_rating'] == 100, 'Ratings_Star'] = 'Diamond'
airbnb_data['Ratings_Star'] = airbnb_data['Ratings_Star'].fillna('Gold')

# print(airbnb_data.Ratings_Star)

# split the dataset
X_train, X_test = train_test_split(airbnb_data, test_size=0.5, random_state=0)
print(len(X_train))
print(len(X_test))
# Checking class imbalance

cs = pd.concat([X_train.Ratings_Star, X_test.Ratings_Star], axis=0)
cs = pd.DataFrame(cs.value_counts())
print(cs)

fig = px.pie(cs, values='Ratings_Star', names=cs.index.values, title='Class Distribution')

# fig.show()

# Data cleaning and pre-processing
# Drop fields host_has_profile_pic and host_identity_verified
X_train.drop(['host_has_profile_pic', 'host_identity_verified', 'longitude', 'latitude', 'host_since'], axis=1,
             inplace=True)
X_test.drop(['host_has_profile_pic', 'host_identity_verified', 'longitude', 'latitude', 'host_since'], axis=1,
            inplace=True)

# To enumerate labels into values for enumerated columns in the dataset
# Could not optimize this code due to time constraint

le = LabelEncoder()
X_train['listing_id'] = le.fit_transform(X_train.listing_id.values)
X_train['host_id'] = le.fit_transform(X_train.host_id.values)
X_train['name'] = le.fit_transform(X_train.name.values)
X_train['host_location'] = le.fit_transform(X_train.host_location.values)
X_train['host_is_superhost'] = le.fit_transform(X_train.host_is_superhost.values)
X_train['host_total_listings_count'] = le.fit_transform(X_train.host_total_listings_count.values)
X_train['neighbourhood'] = le.fit_transform(X_train.neighbourhood.values)
X_train['city'] = le.fit_transform(X_train.city.values)
X_train['property_type'] = le.fit_transform(X_train.property_type.values)
X_train['room_type'] = le.fit_transform(X_train.room_type.values)
X_train['accommodates'] = le.fit_transform(X_train.accommodates.values)
X_train['amenities'] = le.fit_transform(X_train.amenities.values)
X_train['price'] = le.fit_transform(X_train.price.values)
X_train['minimum_nights'] = le.fit_transform(X_train.minimum_nights.values)
X_train['maximum_nights'] = le.fit_transform(X_train.maximum_nights.values)
X_train['review_scores_rating'] = le.fit_transform(X_train.review_scores_rating.values)
X_train['review_scores_accuracy'] = le.fit_transform(X_train.review_scores_accuracy.values)
X_train['review_scores_cleanliness'] = le.fit_transform(X_train.review_scores_cleanliness.values)
X_train['review_scores_checkin'] = le.fit_transform(X_train.review_scores_checkin.values)
X_train['review_scores_communication'] = le.fit_transform(X_train.review_scores_communication.values)
X_train['review_scores_location'] = le.fit_transform(X_train.review_scores_location.values)
X_train['review_scores_value'] = le.fit_transform(X_train.review_scores_value.values)
X_train['instant_bookable'] = le.fit_transform(X_train.instant_bookable.values)
X_train['Ratings_Star'] = le.fit_transform(X_train.Ratings_Star.values)

# For test data
X_test['listing_id'] = le.fit_transform(X_test.listing_id.values)
X_test['host_id'] = le.fit_transform(X_test.host_id.values)
X_test['name'] = le.fit_transform(X_test.name.values)
X_test['host_location'] = le.fit_transform(X_test.host_location.values)
X_test['host_is_superhost'] = le.fit_transform(X_test.host_is_superhost.values)
X_test['host_total_listings_count'] = le.fit_transform(X_test.host_total_listings_count.values)
X_test['neighbourhood'] = le.fit_transform(X_test.neighbourhood.values)
X_test['city'] = le.fit_transform(X_test.city.values)
X_test['property_type'] = le.fit_transform(X_test.property_type.values)
X_test['room_type'] = le.fit_transform(X_test.room_type.values)
X_test['accommodates'] = le.fit_transform(X_test.accommodates.values)
X_test['amenities'] = le.fit_transform(X_test.amenities.values)
X_test['price'] = le.fit_transform(X_test.price.values)
X_test['minimum_nights'] = le.fit_transform(X_test.minimum_nights.values)
X_test['maximum_nights'] = le.fit_transform(X_test.maximum_nights.values)
X_test['review_scores_rating'] = le.fit_transform(X_test.review_scores_rating.values)
X_test['review_scores_accuracy'] = le.fit_transform(X_test.review_scores_accuracy.values)
X_test['review_scores_cleanliness'] = le.fit_transform(X_test.review_scores_cleanliness.values)
X_test['review_scores_checkin'] = le.fit_transform(X_test.review_scores_checkin.values)
X_test['review_scores_communication'] = le.fit_transform(X_test.review_scores_communication.values)
X_test['review_scores_location'] = le.fit_transform(X_test.review_scores_location.values)
X_test['review_scores_value'] = le.fit_transform(X_test.review_scores_value.values)
X_test['instant_bookable'] = le.fit_transform(X_test.instant_bookable.values)
X_test['Ratings_Star'] = le.fit_transform(X_test.Ratings_Star.values)

# For train dataset
cat_col = [c for c in X_train.columns if X_train[c].dtype == 'object']

# For test dataset
cat_cols = [c for c in X_test.columns if X_test[c].dtype == 'object']

cat_col = [c for c in X_train.columns if X_train[c].dtype == 'object']
# Data cleaning for both train and test
# Cleaning Data
for col in cat_col:

    try:
        X_train[col] = X_train[col].astype('float64')
    except:
        X_train[col] = X_train[col]
# print(X_train.dtypes)
for col in cat_cols:
    try:
        X_test[col] = X_test[col].astype('float64')
    except:
        X_test[col] = X_test[col]
# Identify the missing values in the columns

msno.matrix(X_train)
# plt.show()

# Deleting these columns due to high volume of blank rows and data not too relevant for analysis
X_train.drop(['host_response_time'], axis=1, inplace=True)
X_train.drop(['host_response_rate'], axis=1, inplace=True)
X_train.drop(['host_acceptance_rate'], axis=1, inplace=True)
X_train.drop(['district'], axis=1, inplace=True)
X_train.drop(['bedrooms'], axis=1, inplace=True)
X_test.drop(['host_response_time'], axis=1, inplace=True)
X_test.drop(['host_response_rate'], axis=1, inplace=True)
X_test.drop(['host_acceptance_rate'], axis=1, inplace=True)
X_test.drop(['district'], axis=1, inplace=True)
X_test.drop(['bedrooms'], axis=1, inplace=True)
msno.matrix(X_train)


# plt.show()

# Remove garbage values from dataset

# -------------------------------------------
# Function for classifier results execution

def cl_run(m, te, pr, pr_prob, x):
    print('======================================================')
    print(m)
    print('------------------------------------------------------')
    print('')
    precision = met.precision_score(te, pr, average='micro')
    recall = met.recall_score(te, pr, average='micro')
    accuracy = met.accuracy_score(te, pr)
    f1 = met.f1_score(te, pr, average='micro')
    print('Accuracy of the model is', round(accuracy, 2))
    cl_rep = met.classification_report(te, pr)
    print(cl_rep)
    # Save the results in the DataFrame
    if x == 1:
        Result_tb.loc[len(Result_tb.index)] = [m, round(accuracy, 2),
                                               round(precision, 2), round(recall, 2), round(f1, 2)]
    elif x == 5:
        Result_ig.loc[len(Result_ig.index)] = [m, round(accuracy, 2),
                                               round(precision, 2), round(recall, 2), round(f1, 2)]
    elif x == 2:
        Result_l1.loc[len(Result_l1.index)] = [m, round(accuracy, 2),
                                               round(precision, 2), round(recall, 2), round(f1, 2)]

    elif x == 6:
        Result_fs.loc[len(Result_fs.index)] = [m, round(accuracy, 2),
                                               round(precision, 2), round(recall, 2), round(f1, 2)]
    else:
        Result.loc[len(Result.index)] = [m, round(accuracy, 2),
                                         round(precision, 2), round(recall, 2), round(f1, 2)]

    # Plotting Confusion Matrix
    cf_mtx = met.confusion_matrix(te, pr)
    # cf_mtx = ConfusionMatrix(te, pr)
    ax = sns.heatmap(cf_mtx, annot=True, cmap='coolwarm', fmt='g')

    ax.set_title('Confusion Matrix for %s' % m)
    ax.set_xlabel('Predicted Values')
    ax.set_ylabel('Actual Values ')

    ax.xaxis.set_ticklabels(['0', '1', '2'])
    ax.yaxis.set_ticklabels(['0', '1', '2'])
    #    plt.show()

    mtx_rep = ConfusionMatrix(list(te), list(pr))
    print(mtx_rep)

    # Plotting ROC Curve
    plot_roc_curve(y_te, pr)


#   plt.show()

def plot_roc_curve(y_test, y_pred):
    n_classes = len(np.unique(y_test))
    y_test = label_binarize(y_test, classes=(np.arange(n_classes) + 1))
    y_pred = label_binarize(y_pred, classes=(np.arange(n_classes) + 1))

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Plot all ROC curves
    lw = 2
    colors = cycle(["darkred", "darkgreen", "darkblue"])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label="ROC curve of class {0} (area = {1:0.2f})".format(i + 1, roc_auc[i]), )

    plt.plot([0, 1], [0, 1], "k--", lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC) curve")
    plt.legend()


# Storing results in a DataFrame
metrics = ['Model', 'Accuracy Score', ' Precision', 'Recall', 'F1 Score']
Result = pd.DataFrame(columns=metrics)
Result_l1 = pd.DataFrame(columns=metrics)
Result_tb = pd.DataFrame(columns=metrics)
Result_ig = pd.DataFrame(columns=metrics)
Result_fs = pd.DataFrame(columns=metrics)


# Start of code - 1. Attribute selection Pearson Co-relation and the 5 classifier algorithms

x = X_train.drop('Ratings_Star', axis=1)

corr_mtx = x.corr()
plt.figure(figsize=(30, 30))
sns.heatmap(corr_mtx, annot=True)
# plt.show()

col = [i for i in X_train.columns]
col.remove('Ratings_Star')
y = np.array(X_train['Ratings_Star'])
p_corr = pd.DataFrame(columns=['Cor_Coef', 'P_Val'])
for i in col:
    x = np.array(X_train[i])
    r, p = stats.pearsonr(x, y)
    p_corr.loc[i] = [round(r, 2), round(p, 2)]

p_corr = p_corr.sort_values(by=['Cor_Coef'], ascending=True)

fig = px.bar(p_corr, x=p_corr.index.values, y='Cor_Coef', hover_data=['Cor_Coef'], color='Cor_Coef',
             title='Correlation of all attributes w.r.t Class attribute')
# fig.show()
# Top 7 and bottom 7 attributes selected
pcor_col = np.append(p_corr.head(5).index.values, p_corr.tail(5).index.values)
pcor_col = np.append(pcor_col, np.array(['Ratings_Star']))
print(pcor_col, len(pcor_col))
pcor_train = X_train[pcor_col]
pcor_train.head()
pcor_test = X_test[pcor_col]
pcor_test.head()

# Training the model

x_tr = pcor_train.drop(['Ratings_Star'], axis=1)
x_te = pcor_test.drop(['Ratings_Star'], axis=1)
y_tr = pcor_train.Ratings_Star
y_te = pcor_test.Ratings_Star

# 1) Naive Bayes - Gaussian
nbg_cl = nb.GaussianNB()
nbg_cl.fit(x_tr, y_tr)
pr_nbg = nbg_cl.predict(x_te)
pr_prob_nbg = nbg_cl.predict_proba(x_te)
cl_run('Naive Bayes - Gaussian', y_te, pr_nbg, pr_prob_nbg[:, -1], 0)

# 2) Random Forest - Gini
rf_cl = en.RandomForestClassifier(max_features='sqrt', criterion='gini', random_state=4)
rf_cl.fit(x_tr, y_tr)
pr_rf = rf_cl.predict(x_te)
pr_prob_rf = rf_cl.predict_proba(x_te)
cl_run('Random Forest - Entropy', y_te, pr_rf, pr_prob_rf[:, -1], 0)

# 3) GridSearchCV over Random Forest
rf_para = {'criterion': ['gini', 'entropy'], 'max_features': ['sqrt']}
clf = GridSearchCV(en.RandomForestClassifier(), rf_para)
clf.fit(x_tr, y_tr)
pr_clf = clf.predict(x_te)
pr_prob_clf = clf.predict_proba(x_te)
cl_run('GridSearchCV -> Random Forest', y_te, pr_clf, pr_prob_clf[:, -1], 0)

# 4) Gradient Boosting
gb_cl = en.GradientBoostingClassifier()
gb_cl.fit(x_tr, y_tr)
pr_gb = gb_cl.predict(x_te)
pr_prob_gb = gb_cl.predict_proba(x_te)
cl_run('Gradient Boosting', y_te, pr_gb, pr_prob_gb[:, -1], 0)

# 5) AdaBoost
ad_cl = en.AdaBoostClassifier(n_estimators=250, random_state=4)
ad_cl.fit(x_tr, y_tr)
pr_ad = ad_cl.predict(x_te)
pr_prob_ad = ad_cl.predict_proba(x_te)
cl_run('AdaBoost', y_te, pr_ad, pr_prob_ad[:, -1], 0)

# End of code for attribute selection -Pearson co-relation and five classifier algorithms

# Start of code - 2. Extra Trees Classifier for Attribute selection and 5 classifier algorithms
# Feature selection algorithm - Extra Trees
x = X_train.drop('Ratings_Star', axis=1)
y = X_train.Ratings_Star
# X_train['name'] = X_train['name'].astype('float64')
# clf = ExtraTreesClassifier(n_estimators=5, random_state=2)
clf = ExtraTreesClassifier()
clf = clf.fit(x, y)

coef = pd.concat([pd.DataFrame(x.columns), pd.DataFrame(np.transpose(clf.feature_importances_))], axis=1)
coef.columns = ['Feature', 'Importance']
coef.sort_values(by='Importance', inplace=True)

fig = px.bar(coef, x="Feature", y="Importance", title='Feature Importance from Extra Trees classifier',
             color='Importance')
# fig.show()

model = SelectFromModel(clf, prefit=True)
feature_idx = model.get_support()
feature_name = x.columns[feature_idx]
print(feature_name, len(feature_name))

# Training the model
tb_train = X_train[list(feature_name)]
tb_test = X_test[list(feature_name)]
y_te = X_test.Ratings_Star
y_tr = X_train.Ratings_Star

# 1) Naive Bayes - Gaussian
nbg_cl = nb.GaussianNB()
nbg_cl.fit(tb_train, y_tr)
pr_nbg = nbg_cl.predict(tb_test)
pr_prob_nbg = nbg_cl.predict_proba(tb_test)
cl_run('Naive Bayes - Gaussian', y_te, pr_nbg, pr_prob_nbg[:, -1], 1)

# 2) Random Forest - Gini
rf_cl = en.RandomForestClassifier(max_features='sqrt', criterion='gini', random_state=4)
rf_cl.fit(tb_train, y_tr)
pr_rf = rf_cl.predict(tb_test)
pr_prob_rf = rf_cl.predict_proba(tb_test)
cl_run('Random Forest - Entropy', y_te, pr_rf, pr_prob_rf[:, -1], 1)

# 3) GridSearchCV over Random Forest
rf_para = {'criterion': ['gini', 'entropy'], 'max_features': ['sqrt']}
clf = GridSearchCV(en.RandomForestClassifier(), rf_para)
clf.fit(tb_train, y_tr)
pr_clf = clf.predict(tb_test)
pr_prob_clf = clf.predict_proba(tb_test)
cl_run('GridSearchCV -> Random Forest', y_te, pr_clf, pr_prob_clf[:, -1], 1)

# 4) Gradient Boosting
gb_cl = en.GradientBoostingClassifier()
gb_cl.fit(tb_train, y_tr)
pr_gb = gb_cl.predict(tb_test)
pr_prob_gb = gb_cl.predict_proba(tb_test)
cl_run('Gradient Boosting', y_te, pr_gb, pr_prob_gb[:, -1], 1)

# 5) AdaBoost
ad_cl = en.AdaBoostClassifier(n_estimators=50, random_state=10)
ad_cl.fit(tb_train, y_tr)
pr_ad = ad_cl.predict(tb_test)
pr_prob_ad = ad_cl.predict_proba(tb_test)
cl_run('AdaBoost', y_te, pr_ad, pr_prob_ad[:, -1], 1)

# End of code - 2. Extra Trees Classifier for Attribute selection and 5 classifier algorithms


# Start of code - 3. L1 based attribute selection and 5 classifier algorithms
x = X_train.drop('Ratings_Star', axis=1)
y = X_train.Ratings_Star

lsvc = LinearSVC(C=0.001, penalty="l1", dual=False, random_state=4).fit(x, y)
model = SelectFromModel(lsvc, prefit=True)
feature_idx = model.get_support()
feature_name = x.columns[feature_idx]
print(feature_name, len(feature_name))

l1_train = X_train[list(feature_name)]
l1_test = X_test[list(feature_name)]
y_tr = X_train.Ratings_Star
y_te = X_test.Ratings_Star

# 1) Naive Bayes - Gaussian
nbg_cl = nb.GaussianNB()
nbg_cl.fit(l1_train, y_tr)
pr_nbg = nbg_cl.predict(l1_test)
pr_prob_nbg = nbg_cl.predict_proba(l1_test)
cl_run('Naive Bayes - Gaussian', y_te, pr_nbg, pr_prob_nbg[:, -1], 2)

# 2) Random Forest - Gini
rf_cl = en.RandomForestClassifier(max_features='sqrt', criterion='gini', random_state=4)
rf_cl.fit(l1_train, y_tr)
pr_rf = rf_cl.predict(l1_test)
pr_prob_rf = rf_cl.predict_proba(l1_test)
cl_run('Random Forest - Entropy', y_te, pr_rf, pr_prob_rf[:, -1], 2)

# 3) GridSearchCV over Random Forest
rf_para = {'criterion': ['gini', 'entropy'], 'max_features': ['sqrt']}
clf = GridSearchCV(en.RandomForestClassifier(), rf_para)
clf.fit(l1_train, y_tr)
pr_clf = clf.predict(l1_test)
pr_prob_clf = clf.predict_proba(l1_test)
cl_run('GridSearchCV -> Random Forest', y_te, pr_clf, pr_prob_clf[:, -1], 2)

# 4) Gradient Boosting
gb_cl = en.GradientBoostingClassifier()
gb_cl.fit(l1_train, y_tr)
pr_gb = gb_cl.predict(l1_test)
pr_prob_gb = gb_cl.predict_proba(l1_test)
cl_run('Gradient Boosting', y_te, pr_gb, pr_prob_gb[:, -1], 2)

# 5) AdaBoost
ad_cl = en.AdaBoostClassifier(n_estimators=250, random_state=4)
ad_cl.fit(l1_train, y_tr)
pr_ad = ad_cl.predict(l1_test)
pr_prob_ad = ad_cl.predict_proba(l1_test)
cl_run('AdaBoost', y_te, pr_ad, pr_prob_ad[:, -1], 2)

# End of code - 3. L1 based attribute selection with 5 classifier algorithms


# Start of code - 4. Information gain for attribute selection with 5 classifier algorithms
x = X_train.drop('Ratings_Star', axis=1)
y = X_train.Ratings_Star

importance = mutual_info_classif(x, y)
feat_importance = pd.Series(importance, X_train.columns[0: len(X_train.columns) - 1])

feat_importance = feat_importance.sort_values()

fig = px.bar(feat_importance, x=feat_importance.index.values,
             y=feat_importance, title='Feature Importance from Information Gain',
             color=feat_importance)
# fig.show()

# Select top # features

sel_feat = feat_importance.sort_values(ascending=False).index.values[0:14]
sel_feat = np.concatenate((sel_feat, ['Ratings_Star']))
print(sel_feat)

ig_train = X_train[list(sel_feat)]
ig_train.head()

ig_test = X_test[list(sel_feat)]
ig_test.head()

x_tr = ig_train.drop(['Ratings_Star'], axis=1)
x_te = ig_test.drop(['Ratings_Star'], axis=1)
y_tr = ig_train.Ratings_Star
y_te = ig_test.Ratings_Star

# 1) Naive Bayes - Gaussian
nbg_cl = nb.GaussianNB()
nbg_cl.fit(x_tr, y_tr)
pr_nbg = nbg_cl.predict(x_te)
pr_prob_nbg = nbg_cl.predict_proba(x_te)
cl_run('Naive Bayes - Gaussian', y_te, pr_nbg, pr_prob_nbg[:, -1], 5)

# 2) Random Forest - Gini
rf_cl = en.RandomForestClassifier(max_features='sqrt', criterion='gini', random_state=4)
rf_cl.fit(x_tr, y_tr)
pr_rf = rf_cl.predict(x_te)
pr_prob_rf = rf_cl.predict_proba(x_te)
cl_run('Random Forest - Entropy', y_te, pr_rf, pr_prob_rf[:, -1], 5)

# 3) GridSearchCV over Random Forest
rf_para = {'criterion': ['gini', 'entropy'], 'max_features': ['sqrt']}
clf = GridSearchCV(en.RandomForestClassifier(), rf_para)
clf.fit(x_tr, y_tr)
pr_clf = clf.predict(x_te)
pr_prob_clf = clf.predict_proba(x_te)
cl_run('GridSearchCV -> Random Forest', y_te, pr_clf, pr_prob_clf[:, -1], 5)

# 4) Gradient Boosting
gb_cl = en.GradientBoostingClassifier()
gb_cl.fit(x_tr, y_tr)
pr_gb = gb_cl.predict(x_te)
pr_prob_gb = gb_cl.predict_proba(x_te)
cl_run('Gradient Boosting', y_te, pr_gb, pr_prob_gb[:, -1], 5)

# 5) AdaBoost
ad_cl = en.AdaBoostClassifier(n_estimators=250, random_state=4)
ad_cl.fit(x_tr, y_tr)
pr_ad = ad_cl.predict(x_te)
pr_prob_ad = ad_cl.predict_proba(x_te)
cl_run('AdaBoost', y_te, pr_ad, pr_prob_ad[:, -1], 5)

# End of code - 4. Information gain for attribute selection with 5 classifier algorithms

# Start of code - 5. Univariate selection from F Test for attribute selection with 5 classifier algorithms
x = X_train.drop('Ratings_Star', axis=1)
y = X_train.Ratings_Star
fs = SelectKBest(f_classif, k=9)
fs.fit(x, y)
feature_idx = fs.get_support()
feature_name = x.columns[feature_idx]
print(feature_name, len(feature_name))

fs_tr = X_train[feature_name]
fs_te = X_test[feature_name]
y_tr = X_train.Ratings_Star
y_te = X_test.Ratings_Star

# 1) Naive Bayes - Gaussian
nbg_cl = nb.GaussianNB()
nbg_cl.fit(fs_tr, y_tr)
pr_nbg = nbg_cl.predict(fs_te)
pr_prob_nbg = nbg_cl.predict_proba(fs_te)
cl_run('Naive Bayes - Gaussian', y_te, pr_nbg, pr_prob_nbg[:, -1], 6)

# 2) Random Forest - Gini
rf_cl = en.RandomForestClassifier(max_features='sqrt', criterion='gini', random_state=4)
rf_cl.fit(fs_tr, y_tr)
pr_rf = rf_cl.predict(fs_te)
pr_prob_rf = rf_cl.predict_proba(fs_te)
cl_run('Random Forest - Entropy', y_te, pr_rf, pr_prob_rf[:, -1], 6)

# 3) GridSearchCV over Random Forest
rf_para = {'criterion': ['gini', 'entropy'], 'max_features': ['sqrt']}
clf = GridSearchCV(en.RandomForestClassifier(), rf_para)
clf.fit(fs_tr, y_tr)
pr_clf = clf.predict(fs_te)
pr_prob_clf = clf.predict_proba(fs_te)
cl_run('GridSearchCV -> Random Forest', y_te, pr_clf, pr_prob_clf[:, -1], 6)

# 4) Gradient Boosting
gb_cl = en.GradientBoostingClassifier()
gb_cl.fit(fs_tr, y_tr)
pr_gb = gb_cl.predict(fs_te)
pr_prob_gb = gb_cl.predict_proba(fs_te)
cl_run('Gradient Boosting', y_te, pr_gb, pr_prob_gb[:, -1], 6)

# 5) AdaBoost
ad_cl = en.AdaBoostClassifier(n_estimators=250, random_state=4)
ad_cl.fit(fs_tr, y_tr)
pr_ad = ad_cl.predict(fs_te)
pr_prob_ad = ad_cl.predict_proba(fs_te)
cl_run('AdaBoost', y_te, pr_ad, pr_prob_ad[:, -1], 6)

# End of code - 5. Univariate selection from F Test for attribute selection with 5 classifier algorithms

print(Result)

print(Result_tb)

print(Result_l1)

print(Result_ig)

print(Result_fs)
