import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing

if __name__=="__main__":
    #exploring data
    #read data
    data = pd.read_csv('diabetic_data.csv')

    print "data shape is: ",data.shape
    print "data head is: ",data.head()
    print "data information is: ",data.info()
    #look at the result variable
    data.groupby('readmitted').size().plot(kind='bar',color='green')
    plt.ylabel('number')
    plt.show()
    #revise the readmiited varibales
    data['readmitted'] = pd.Series([0 if val == 'NO' else 1 for val in data['readmitted']])
    #drop id information features
    data.drop(['encounter_id', 'patient_nbr', 'payer_code'], axis=1, inplace=True)
    #check features with a lot of NA
    weight_na=data[data['weight'] == '?'].shape[0] * 1.0 / data.shape[0]
    print 'weight feature NA rate is: ',weight_na
    medical_specaiality_na=data[data['medical_specialty'] == '?'].shape[0] * 1.0 / data.shape[0]
    print 'medical_speciality NA rate is: ',medical_specaiality_na
    #drop these two features
    data.drop(['weight', 'medical_specialty'], axis=1, inplace=True)
    #check other features and remove related rows
    print 'race NA is: ',data[data['race'] == '?'].shape[0] * 1.0 / data.shape[0]
    print 'diag1 NA is: ',data[data['diag_1'] == '?'].shape[0] * 1.0 / data.shape[0]
    print 'diag2 NA is: ',data[data['diag_2'] == '?'].shape[0] * 1.0 / data.shape[0]
    print 'diag3 NA is: ',data[data['diag_3'] == '?'].shape[0] * 1.0 / data.shape[0]
    print 'gender NA is: ',data[data['gender'] == 'Unknown/Invalid'].shape[0] * 1.0 / data.shape[0]
    data = data[data['race'] != '?']
    data = data[data['diag_1'] != '?']
    data = data[data['diag_2'] != '?']
    data = data[data['diag_3'] != '?']
    data = data[data['gender'] != 'Unknown/Invalid']
    #regroup age feature
    data.groupby('age').size().plot(kind='bar',color='red')
    plt.ylabel('number')
    plt.show()
    #the data of age is unbalanced
    data['age'] = pd.Series(['[0-40)' if val in ['[0-10)', '[10-20)', '[20-30)', '[30-40)'] else val
                             for val in data['age']], index=data.index)
    data['age'] = pd.Series(['[80-100)' if val in ['[80-90)', '[90-100)'] else val
                             for val in data['age']], index=data.index)
    data.groupby('age').size().plot(kind='bar',color='red')
    plt.ylabel('number')
    plt.show()
    #deal with discharge_disposition_id,admission_source_id,
    data['discharge_disposition_id'] = pd.Series(['Home' if val == 1 else 'Others' for val in data['discharge_disposition_id']], index=data.index)
    data['admission_source_id'] = pd.Series(['Emergency Room' if val == 7 else 'Referral' if val == 1 else 'Others' for val in data['admission_source_id']], index=data.index)
    data['admission_type_id'] = pd.Series(['Emergency' if val == 1 else 'Urgent' if val==2 else 'Others'for val in data['admission_type_id']], index=data.index)
    #clean some featues which have unbalanced distribution
    features=['metformin', 'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride',
               'acetohexamide', 'glipizide', 'glyburide', 'tolbutamide', 'pioglitazone',
               'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone', 'tolazamide', 'examide',
               'citoglipton', 'glyburide-metformin', 'glipizide-metformin', 'glimepiride-pioglitazone',
               'metformin-rosiglitazone', 'metformin-pioglitazone','insulin']
    for i in range(23):
        data.groupby(features[i]).size().plot(kind='bar')
        #plt.show()
    #except insulin,others are really unbalanced
    data.drop(['metformin', 'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride',
               'acetohexamide', 'glipizide', 'glyburide', 'tolbutamide', 'pioglitazone',
               'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone', 'tolazamide', 'examide',
               'citoglipton', 'glyburide-metformin', 'glipizide-metformin', 'glimepiride-pioglitazone',
               'metformin-rosiglitazone', 'metformin-pioglitazone'], axis=1, inplace=True)
    data['diag_1'] = pd.Series([1 if val.startswith('250') else 0 for val in data['diag_1']], index=data.index)
    data['diag_2'] = pd.Series([1 if val.startswith('250') else 0 for val in data['diag_2']], index=data.index)
    data['diag_3'] = pd.Series([1 if val.startswith('250') else 0 for val in data['diag_3']], index=data.index)
    #one-hot encoding for categorical features
    df_age = pd.get_dummies(data['age'])
    df_race = pd.get_dummies(data['race'])
    df_gender = pd.get_dummies(data['gender'])
    df_max_glu_serum = pd.get_dummies(data['max_glu_serum'])
    df_A1Cresult = pd.get_dummies(data['A1Cresult'])
    df_insulin = pd.get_dummies(data['insulin'])
    df_change = pd.get_dummies(data['change'])
    df_diabetesMed = pd.get_dummies(data['diabetesMed'])
    df_discharge_disposition_id = pd.get_dummies(data['discharge_disposition_id'])
    df_admission_source_id = pd.get_dummies(data['admission_source_id'])
    df_admission_type_id = pd.get_dummies(data['admission_type_id'])

    data = pd.concat([data, df_age, df_race, df_gender, df_max_glu_serum, df_A1Cresult,
                      df_insulin, df_change, df_diabetesMed, df_discharge_disposition_id,
                      df_admission_source_id, df_admission_type_id], axis=1)
    data.drop(['age', 'race', 'gender', 'max_glu_serum', 'A1Cresult', 'insulin', 'change',
                      'diabetesMed', 'discharge_disposition_id', 'admission_source_id',
                      'admission_type_id'], axis=1, inplace=True)
    #deal with extreme values,to do root transformation and feature scaling
    data['number_outpatient'] = data['number_outpatient'].apply(lambda x: np.sqrt(x + 0.5))
    data['number_emergency'] = data['number_emergency'].apply(lambda x: np.sqrt(x + 0.5))
    data['number_inpatient'] = data['number_inpatient'].apply(lambda x: np.sqrt(x + 0.5))
    feature_scale_cols = ['time_in_hospital', 'num_lab_procedures', 'num_procedures', 'num_medications',
                          'number_diagnoses', 'number_inpatient', 'number_emergency', 'number_outpatient']
    scaler = preprocessing.StandardScaler().fit(data[feature_scale_cols])
    data_scaler = scaler.transform(data[feature_scale_cols])
    data_scaler_df = pd.DataFrame(data=data_scaler, columns=feature_scale_cols, index=data.index)
    data.drop(feature_scale_cols, axis=1, inplace=True)
    data = pd.concat([data, data_scaler_df], axis=1)
    X = data.drop(['readmitted'], axis=1)
    y = data['readmitted']
    from sklearn.cross_validation import train_test_split
    X_cv, X_test, y_cv, y_test = train_test_split(X, y, test_size=0.25)
    from sklearn.ensemble import RandomForestClassifier
    forest = RandomForestClassifier()
    forest.fit(X_cv, y_cv)
    importances = forest.feature_importances_
    feature_importance = 100.0 * (importances / importances.max())
    sorted_idx = np.argsort(feature_importance)
    feature_names = list(X_cv.columns.values)
    feature_names_sort = [feature_names[indice] for indice in sorted_idx]
    pos = np.arange(sorted_idx.shape[0]) + .5
    print 'Top 10 important features are: '
    for feature in feature_names_sort[::-1][:10]:
        print feature
    plt.figure(figsize=(12, 10))
    plt.barh(pos, feature_importance[sorted_idx], align='center')
    plt.yticks(pos, feature_names_sort)
    plt.title('Relative Feature Importance', fontsize=20)
    plt.show()
    cleansed_data = X_cv[['num_lab_procedures', 'num_medications', 'time_in_hospital', 'num_procedures', 'number_inpatient',
               'number_diagnoses','number_outpatient','number_emergency','diag_3','Female']]

    #clustring
    inertias=[]
    from sklearn.cluster import KMeans

    for N in range(1,9):
            km = KMeans(n_clusters=N)
            km.fit(data.values)
            inertias.append(km.inertia_)
            print inertias
    n=[i for i in range(1,9)]
    plt.bar(n, inertias)
    plt.title("cluster elbow")
    plt.ylabel("KMeans inertia")
    plt.xlabel("n_clusters")
    plt.show()
    # According to elbow clustering,n should be 2
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import Normalizer
    # using PCA to reduce dimensions
    reducer = PCA(n_components=2)
    X = data.values.astype(np.float32)
    norm = Normalizer()
    Xnorm = norm.fit_transform(X)
    XX =reducer.fit_transform(Xnorm)
    plt.figure()
    plt.scatter(XX[:, 0], XX[:, 1])
    plt.show()
    print XX[0]
    # using k-means to do clustering,k=2
    km=KMeans(n_clusters=2).fit(XX)
    colors=['r','g']
    ax1=np.where(km.labels_==1)
    ax2=np.where(km.labels_==0)
    result1=XX[ax1]
    result2=XX[ax2]
    plt.figure()
    plt.scatter(result1[:,0], result1[:,1], color="r")
    plt.scatter(result2[:, 0], result2[:, 1], color="g")
    plt.show()

    ##classification
    from sklearn.cross_validation import cross_val_score
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.naive_bayes import GaussianNB
    classifier_LR = LogisticRegression()
    classifier_RF = RandomForestClassifier()
    classifier_NB = GaussianNB()
    NB_score = cross_val_score(classifier_NB, X_cv, y_cv, cv=10, scoring='accuracy').mean()
    RF_score = cross_val_score(classifier_RF, X_cv, y_cv, cv=10, scoring='accuracy').mean()
    LR_score = cross_val_score(classifier_LR, X_cv, y_cv, cv=10, scoring='accuracy').mean()
    print "LR score is: ",LR_score
    print "RF score is: ",RF_score
    print "NB score is: ",NB_score
    #because lr gets the highest score ,we select the lr model
    ##evaluation
    logreg=LogisticRegression(C=0.1, class_weight=None, dual=False,
              fit_intercept=True, intercept_scaling=1, max_iter=100,
              multi_class='ovr', n_jobs=1, penalty='l2', random_state=None,
              solver='liblinear', tol=0.0001, verbose=0, warm_start=False)
    logreg.fit(X_cv, y_cv)
    from sklearn import metrics
    y_pred_class = logreg.predict(X_test)
    print metrics.accuracy_score(y_test, y_pred_class)
    y_pred_prob = logreg.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred_prob)
    plt.plot(fpr, tpr)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.title('ROC curve')
    plt.xlabel('False Positive')
    plt.ylabel('True Positive')
    plt.grid(True)
    plt.show()
    print "AUC score is: ",metrics.roc_auc_score(y_test, y_pred_prob)
