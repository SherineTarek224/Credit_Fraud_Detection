
from Models_Comparison import *
from sklearn.metrics import confusion_matrix,classification_report
import matplotlib.pyplot as plt
import seaborn as sns
r_s=17
np.random.seed(r_s)

def confusion_matrix_heatmap(y_actual, y_pred, title='Training'):
        cr = classification_report(y_actual, y_pred, output_dict=True)
        print(f"{title} Classification Report")
        print(classification_report(y_actual,y_pred))

        cm = confusion_matrix(y_actual, y_pred)

        plt.figure(figsize=(8, 8))
        sns.heatmap(cm, annot=True, center=0, cmap='coolwarm', fmt='g')
        plt.title(f"{title} Confusion Matrix")
        plt.xlabel("Prediction")
        plt.ylabel("Actual")
        plt.show()
        return cr


if __name__=="__main__":
        x_train,y_train,x_val,y_val=load_data(training=True)
        x_train,x_val,preprocessor=preprocess_data(x_train,x_val,preprocess_option=3)

        #Test
        x_test,y_test=load_data(training=False)
        _,x_test,_=preprocess_data(None,x_test,preprocess_option=3,preprocessor=preprocessor)
        print(f"SHape after preprocessing {x_test.shape}")


       # path to the models.pkl
        #models = joblib.load(r"F:\Machine_Learning\projects\Fraud_detection\trained_models_SMOTE.pkl")

        #models = joblib.load(r"F:\Machine_Learning\projects\Fraud_detection\trained_models_UnderSampling.pkl")

        models_comp={}

        for model_name in ['logistic_regression', 'Random_Forest', 'NN']:

                model_data = models[model_name]
                model = model_data['model']
                optimal_threshold = model_data['optimal_threshold']
                print(f"{model_name}--->{optimal_threshold}")


                y_pred_test_proba = model.predict_proba(x_test)[:, 1]

                # Default threshold (0.5)
                y_pred_default = model.predict(x_test)
                cr=confusion_matrix_heatmap(y_test, y_pred_default,
                                         title=f"{model_name} - Default Threshold (0.5)")

                # Optimal threshold
                y_pred_optimal = (y_pred_test_proba >= optimal_threshold).astype(int)
                opt_cr=confusion_matrix_heatmap(y_test, y_pred_optimal,
                                         title=f"{model_name} - Optimal Threshold ({optimal_threshold:.2f})")

                # Store metrics for both thresholds
                default_metrics = {

                                "F1_score (Class 1)": cr['1']['f1-score'],
                                "Precision (Class 1)": cr['1']['precision'],
                                "Recall (Class 1)": cr['1']['recall']
                }

                optimal_metrics={
                                "F1_score (Class 1)": opt_cr['1']['f1-score'],
                                "Precision (Class 1)": opt_cr['1']['precision'],
                                "Recall (Class 1)": opt_cr['1']['recall']
                        }

                models_comp[model_name] = default_metrics
                models_comp[f"optimal_threshold{model_name}"]=optimal_metrics
        compare_models_heatmap(models_comp,None,None)