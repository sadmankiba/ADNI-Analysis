import numpy as np
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import xgboost              # Extreme Gradient Boosting
import lightgbm as lgb      # Light Gradient Boosting Machine
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import matthews_corrcoef, confusion_matrix

from collections import Counter

def save_selected_features(df):
    """
    Save three different preprocessed data as CSV from ADNIMERGE data
       1. only bl columns and rows 
       2. current columns. last visit
       3. current columns. all visit
    """ 

    # Select columns 
    df = df[["RID", "EXAMDATE", "DX_bl", "AGE", "PTGENDER","PTEDUCAT", "PTMARRY", "APOE4", \
        "CDRSB", "ADAS11","ADAS13","ADASQ4","MMSE","RAVLT_immediate","RAVLT_learning","RAVLT_forgetting","RAVLT_perc_forgetting","LDELTOTAL","TRABSCOR", \
            "Ventricles","Hippocampus","WholeBrain","Entorhinal","Fusiform","MidTemp","DX", \
            "CDRSB_bl","ADAS11_bl","ADAS13_bl","ADASQ4_bl","MMSE_bl","RAVLT_immediate_bl","RAVLT_learning_bl","RAVLT_forgetting_bl","RAVLT_perc_forgetting_bl","LDELTOTAL_BL","TRABSCOR_bl", \
            "Ventricles_bl","Hippocampus_bl","WholeBrain_bl","Entorhinal_bl","Fusiform_bl","MidTemp_bl","FDG_bl"]]

    # Sort in ascending order by RID, EXAMDATE. Take first / last rows of each RID.
    sorted_df = df.sort_values(by=['RID', 'EXAMDATE'], ignore_index=True)
    # if ignore_index does not work  
    # sorted_df.reset_index(inplace=True, drop=True)

    bl_df = sorted_df.drop_duplicates(subset=['RID'], keep='first')
    lv_df = sorted_df.drop_duplicates(subset=['RID'], keep='last')

    # Without considering CDRSB
    bl_df = bl_df[["AGE", "PTGENDER","PTEDUCAT", "PTMARRY", "APOE4", "DX", \
    "ADAS11_bl","ADAS13_bl","ADASQ4_bl","MMSE_bl","RAVLT_immediate_bl","RAVLT_learning_bl","RAVLT_forgetting_bl","RAVLT_perc_forgetting_bl","LDELTOTAL_BL","TRABSCOR_bl",\
        "Ventricles_bl","Hippocampus_bl","WholeBrain_bl","Entorhinal_bl","Fusiform_bl","MidTemp_bl","FDG_bl"]]

    lv_df = lv_df[["PTGENDER","PTEDUCAT", "PTMARRY", "APOE4", \
        "ADAS11","ADAS13","ADASQ4","MMSE","RAVLT_immediate","RAVLT_learning","RAVLT_forgetting","RAVLT_perc_forgetting","LDELTOTAL","TRABSCOR", \
            "Ventricles","Hippocampus","WholeBrain","Entorhinal","Fusiform","MidTemp","DX"]]

    av_df = df[["PTGENDER","PTEDUCAT", "PTMARRY", "APOE4", \
        "ADAS11","ADAS13","ADASQ4","MMSE","RAVLT_immediate","RAVLT_learning","RAVLT_forgetting","RAVLT_perc_forgetting","LDELTOTAL","TRABSCOR", \
            "Ventricles","Hippocampus","WholeBrain","Entorhinal","Fusiform","MidTemp","DX"]]

    # Filter rows : 
    #   row with any empty field
    def drop_rows_with_empty_field(df):
        nan_value = float("NaN")
        dropped_df = df.replace("", nan_value)
        dropped_df.dropna(inplace=True)
        print(f'dropped {len(df) - len(dropped_df)} rows')
        return dropped_df
    
    bl_df = drop_rows_with_empty_field(bl_df)
    lv_df = drop_rows_with_empty_field(lv_df)
    av_df = drop_rows_with_empty_field(av_df)

    bl_df.to_csv('adnimerge_bl.csv', index=False)
    lv_df.to_csv('adnimerge_lv.csv', index=False)
    av_df.to_csv('adnimerge_av.csv', index=False) 

def preprocess(df):
    """
    Preprocess data
    """
    # ------ Oversample ---------
    # oversample to balance the dataset
    # Print Number of AD, MCI, CN
    print('Preprocessing data...')

    print('Before oversampling:', sorted(Counter(df.DX).items()))
    
    ros = RandomOverSampler(random_state=0)
    df_resampled, _ = ros.fit_resample(df, df.DX)
    df_resampled = df_resampled.sample(frac=1).reset_index(drop=True) # shuffle
    print('After oversampling and shuffling:', sorted(Counter(df_resampled.DX).items()))

    # ------------- Manage Categorical Columns -----------
    cat_names = ['PTGENDER','PTMARRY']
    
    # convert categorical columns to one-hot encodings
    df_ohe = df_resampled.copy()
    for cat_col in cat_names:
        df_ohe = pd.get_dummies(df_ohe, columns=[cat_col])
    
    print('After One-hot encoding of categorical columns\n', df_ohe.head())

    # ---------- Normalize Continuous Columns ----------
    cont_names = list(df.columns).copy()

    cont_names.remove('DX')
    for cat_col in cat_names:
        cont_names.remove(cat_col)

    for cont_name in cont_names:
        df_ohe[cont_name] = (df_ohe[cont_name] - min(df_ohe[cont_name]) )/(max(df_ohe[cont_name]) - min(df_ohe[cont_name]))

    # verify that columns are normalized
    print('max\tmin\tavg')
    for cont_name in cont_names:
        print(f"{max(df_ohe[cont_name])}\t{min(df_ohe[cont_name])}\t{np.mean(df_ohe[cont_name])}")
    
    X = df_ohe.drop(['DX'], axis=1).astype(float)
    y = df_ohe.DX
     
    return X, y


class Metrics:    
    def compute_metrics(self, y_true, y_pred):
        accuracy = (y_true==y_pred).sum()/len(y_true)
        
        mcc = matthews_corrcoef(y_true, y_pred)
        
        return (accuracy, mcc)


class Model:
    def build_model(self, X, y):
        lgb_params  = {
          "objective" : "multiclass",
          "num_class" : 4,
          "num_leaves" : 60,
          "max_depth": -1,
          "learning_rate" : 0.01,
          "bagging_fraction" : 0.9,  # subsample
          "feature_fraction" : 0.9,  # colsample_bytree
          "bagging_freq" : 5,        # subsample_freq
          "bagging_seed" : 2018,
          "verbosity" : -1 }

        NN_params = [ {'solver': 'adam', 'learning_rate_init': 0.01} ]
        NN_labels = [ "NN_Adam" ]

        le = preprocessing.LabelEncoder()
        y_label_enc = le.fit_transform(y)
        print('class labels:', dict(zip(range(len(le.classes_)),le.classes_)))

        metrics = Metrics()
        metrics_names = ('acc','mcc',)
        classifier_namelist = [ "RF", "SVM", "LDA", "XGBoost", "LightGBM"]

        # result table header formation
        clf_metrics_labels = [ j +'_'+ i for i in metrics_names for j in classifier_namelist + NN_labels ]
        print('clf_metrics_labels\n', clf_metrics_labels)

        # Train
        cv_results_df = pd.DataFrame({**dict.fromkeys(clf_metrics_labels, [])})
        cv_confusion_mats = []

        outer_fold_idx = 0
        
        skf = StratifiedKFold(n_splits = 5)
        
        for train_idx, test_idx in skf.split(X, y_label_enc):
            print('In outer fold: ', outer_fold_idx)
            outer_fold_idx += 1
            
            X_train = X.iloc[train_idx]
            y_train = y_label_enc[train_idx]
            
            X_test = X.iloc[test_idx]
            y_test = y_label_enc[test_idx]  
    
            # Run 5 / 6 sklearn models
            # initialize the classifiers
            rf_clf = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 42)
            svm_clf = svm.SVC(kernel='rbf', probability=True)
            lda_clf = LinearDiscriminantAnalysis()
            xgb = xgboost.XGBClassifier()
            lgb_clf = lgb.LGBMClassifier(**lgb_params)

            # list of classifiers to iterate over (except for NN)
            classifiers = [rf_clf, svm_clf, lda_clf, xgb, lgb_clf ]

            # add the NNs to the list    
            for NN_config in NN_params:
                classifiers.append(MLPClassifier(alpha=1e-5, hidden_layer_sizes=(100, 75, 25, 16), random_state=1, **NN_config))
            
            clf_results = []
            confusion_mats = []    
        
            for clf in classifiers:
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_test)

                results_metrics = metrics.compute_metrics(y_test, y_pred)
                confusion_mat = confusion_matrix(y_test, y_pred)

                clf_results.append(results_metrics)
                confusion_mats.append(confusion_mat)

            # store the results from the outer fold
            tmp = pd.DataFrame(np.array(clf_results).transpose().reshape(1,-1), columns=cv_results_df.columns )

            # append the results from this outer fold to the collection
            cv_results_df = cv_results_df.append(tmp)
            cv_confusion_mats.append(confusion_mats)
        
        return cv_results_df, cv_confusion_mats

def print_confusion_matrix(cv_confusion_mats, selected_classifier_idx):
    """Confusion Matrix of Selected Classifier Averaged Over Folds"""  
    cv_confusion_mats_np = np.array(cv_confusion_mats)
    conf_mat_freq_meaned = cv_confusion_mats_np[:, selected_classifier_idx, :, :].mean(axis = 0)
    conf_mat_pct_meaned = conf_mat_freq_meaned/conf_mat_freq_meaned.sum(axis =1)
    print(f"conf_mat_pct_meaned\n{conf_mat_pct_meaned}")

if __name__ == '__main__':
    df = pd.read_csv("data/ADNIMERGE.csv")
    print('ADNI Merge Dataset\n', df)
    
    save_selected_features(df)
    
    bl_df = pd.read_csv("adnimerge_bl.csv")
    lv_df = pd.read_csv("adnimerge_lv.csv")
    av_df = pd.read_csv("adnimerge_av.csv")

    for train_df, train_data_name in zip([bl_df, lv_df, av_df], ['bl', 'lv', 'av']):
        print('train_df\n', train_df)
        
        X, y = preprocess(train_df)
        print('Data preprocessed. X, y\n', X, '\n' , y)

        label_list = sorted(y.unique())
        
        model = Model()
        cv_results_df, cv_confusion_mats = model.build_model(X, y)
        
        cv_results_df.to_csv(f"results/results_{train_data_name}.csv", index=False)

        # print accuracy
        # print(cv_results_df.mean(axis = 0))
        print('cv_results_df mean:\n', cv_results_df.mean())

        print_confusion_matrix(cv_confusion_mats, 3)

            