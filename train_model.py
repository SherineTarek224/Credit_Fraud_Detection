
import os

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import GridSearchCV, StratifiedKFold, RandomizedSearchCV
from sklearn.neural_network import MLPClassifier
import joblib
from evaluation import *
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings("ignore")


from data_helper import *

r_s=42
np.random.seed(r_s)


def logistic_regression(x_train, y_train, x_val, y_val, title='logistic_regression',sampling_T="SMOTE",
                            optimal_threshold=False, grid_search=False):
        best_param = {
            "C": 0.1,
            "penalty": "l2",
            "solver": "sag",
            "class_weight": {0: 0.20, 1: 0.80},
            "max_iter": 1000,
            "random_state": r_s
        }

        if grid_search:
            param= {  # L2  parameters
                    'penalty': ['l2'],
                    'solver': ['lbfgs', 'sag', 'saga'],
                    'C': [ 0.01, 0.1, 1, 10],
                    'class_weight': [None, 'balanced'],
                    'max_iter': [400,500,600]
                }


            stkf = StratifiedKFold(n_splits=5, shuffle=True, random_state=r_s)
            search = GridSearchCV(
                estimator=LogisticRegression(random_state=r_s),
                param_grid=param,
                scoring=make_scorer(f1_score, pos_label=1),
                cv=stkf,
                n_jobs=-1,
                verbose=1
            )
            search.fit(x_train, y_train)
            best_param.update(search.best_params_)
            print(f"Best Logistic Regression Parameters: {best_param}")

        model = LogisticRegression(**best_param)
        model.fit(x_train, y_train)
        default_metrics, optimal_metrics,optimal_threshold_value = evaluate_model(
            model, title,sampling_T, x_train, y_train, x_val, y_val, optimal_threshold=optimal_threshold
        )
        return default_metrics, optimal_metrics, model,optimal_threshold_value


def NN(x_train, y_train, x_val, y_val, title='Neural_Nets',sampling_T="SMOTE",optimal_threshold=False, random_search=False):
    best_param = {
        "hidden_layer_sizes": (128, 64),
        "activation": 'tanh',
        "solver": 'sgd',
        "alpha": 0.1,
        "batch_size": 512,
        "learning_rate_init": 0.1,
        "max_iter": 500,

    }

    if random_search:
        param = {
            "hidden_layer_sizes": [(60, 40), (64, 32), (100, 50), (128, 64)],
            "activation": ["relu", "tanh", "logistic"],
            "solver": ["sgd", "adam"],
            "alpha": [ 0.001, 0.01, 0.1],
            "batch_size": [ 128, 256, 512],
            "max_iter": [300, 500, 700],
            "learning_rate_init": [0.001, 0.01, 0.1]
        }

        stkf = StratifiedKFold(n_splits=5, random_state=r_s)
        search = RandomizedSearchCV(
            estimator=MLPClassifier(random_state=r_s),
            param_distributions=param,
            scoring=make_scorer(f1_score, pos_label=1),
            cv=stkf,
            n_iter=25,
            n_jobs=-1,
            verbose=2,
            random_state=r_s
        )
        search.fit(x_train, y_train)
        best_param.update(search.best_params_)
        print(f"Best Neural Network Parameters: {best_param}")

    model = MLPClassifier(**best_param)
    model.fit(x_train, y_train)
    default_metrics, optimal_metrics,optimal_threshold_value = evaluate_model(
        model, title,sampling_T, x_train, y_train, x_val, y_val, optimal_threshold=optimal_threshold
    )
    return default_metrics, optimal_metrics, model,optimal_threshold_value


def random_forest(x_train,y_train,x_val,y_val,title="Random_Forest",sampling_T="SMOTE",optimal_threshold=False,random_search=False):
    best_parameters = {
        "n_estimators": 400,
        "min_samples_leaf": 10,
        "min_samples_split": 10,
        'class_weight': {0: 0.15, 1: 0.85},
        "bootstrap": True,
         "n_jobs": -1,
    }
    if random_search==True:
        param={
            "n_estimators":[200,400,600,800],
            "max_depth":[20,30,50,70],
            'min_samples_split': [5, 10, 15],
            'min_samples_leaf': [5, 10, 15],
            'class_weight': [{0: 0.20, 1: 0.80}, 'balanced_subsample', {0: 0.15, 1: 0.85}]
        }
        stkf=StratifiedKFold(n_splits=5,random_state=r_s)
        score=make_scorer(f1_score,pos_label=1)
        search=RandomizedSearchCV(
            estimator=RandomForestClassifier(),
            param_distributions=param,
            cv=stkf,
            n_iter=20,
            scoring=score,n_jobs=-1,verbose=3,random_state=r_s)

        search.fit(x_train,y_train)

        best_parameters.update(search.best_params_)
        print("Best Hyperparameter",best_parameters)

    model = RandomForestClassifier(**best_parameters)
    model.fit(x_train, y_train)
    default_metrics,optimal_metrics,optimal_threshold_value = evaluate_model(model,title,sampling_T,x_train,y_train,x_val,y_val,optimal_threshold=optimal_threshold)
    return default_metrics,optimal_metrics,model,optimal_threshold_value


def save_models(models,root_path=r"F:\Machine_Learning\projects\Fraud_detection",model_type="SMOTE"):
    os.makedirs(root_path, exist_ok=True)
    save_path=os.path.join(root_path,f"trained_models_{model_type}.pkl")
    joblib.dump(models,save_path)
    print(f"Saved Successfully in {save_path}")

if __name__=="__main__":
   x_train, y_train, x_val, y_val = load_data(training=True)
   x_train, x_val,preprocessor= preprocess_data(x_train, x_val,3)
   x_train ,y_train = balancing_data(x_train, y_train,type="SMOTE")
   default_metrics,optimal_metrics,model,optimal_threshold= logistic_regression(x_train,y_train,x_val,y_val,title='logistic_regression',sampling_T="SMOTE",optimal_threshold=False,grid_search=False)
   #default_metrics,optimal_metrics,model=random_forest(x_train,y_train,x_val,title,y_val,title='RF',sampling_T="SMOTE",optimal_threshold=False,random_search=False)
   #default_metrics,optimal_metrics,model=NN(x_train,y_train,x_val,y_val,"NN","SMOTE", optimal_threshold=False,random_search=False)
   print("METRICS and optimal matrics")
   print(default_metrics,optimal_metrics)