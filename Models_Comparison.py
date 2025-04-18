from data_helper import *
from train_model import *
from evaluation import *
import argparse
from joblib import dump

r_s=17
np.random.seed(r_s)
models={}
def compare_models_heatmap(model_metrics_dict,sampling_T,sampling_S):

    metrics_df = pd.DataFrame()

    for model_name, metrics in model_metrics_dict.items():
        metrics_df = metrics_df._append(pd.DataFrame(metrics, index=[model_name]))

    # Plot the heatmap
    plt.figure(figsize=(10, 6))
    sns.heatmap(metrics_df, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title(f'Model Comparison Heatmap with {sampling_T} and {sampling_S}')
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fraud Detection')

    parser.add_argument('--dataset', type=str, default=True)
    parser.add_argument('--preprocessing', type=int, default=3, choices=[0, 1, 2, 3],
                        help='0 for no processing, 1 for min/max scaling and 2 for Std ,3 for RobustScaler')

    parser.add_argument("--Sampling_Technique",type=str,default='UnderSampling',choices=["SMOTE", "Under", "Over", "SMOTEENN"],
                        help="Sampling technique to use (SMOTE, Under, Over, SMOTEENN)")
    parser.add_argument("--Sampling_strategy",type=float,default=0.05,
                        help="Sampling strategy (ratio of minority to majority class)")

    parser.add_argument('--choice', type=int, default=1,  #it was 6
                        help='1 logistic regression , '
                             '2 Random First '
                             '3 Neural Network'
                        )

    args = parser.parse_args()

    x_train, y_train, x_val, y_val = load_data(args.dataset)


    preprocess_option = args.preprocessing

    preprocessor=get_preprocessor(preprocess_option)

    x_train, x_val,_=  preprocess_data(x_train, x_val,preprocess_option,preprocessor=None)
    x_train,y_train=balancing_data(x_train,y_train,type=args.Sampling_Technique,sampling_strategy=args.Sampling_strategy)

    model_comparison_dict = {}

   # if args.choice == 1:

    lr_default_metrics,lr_optimal_metrics,lr_model,lr_optimal_threshold=logistic_regression(x_train,y_train,x_val,y_val,title='logistic_regression'
                                                        ,sampling_T=args.Sampling_Technique,optimal_threshold=True,grid_search=False)
    print(f"optimal_threshold_of LR {lr_optimal_threshold}")
    model_comparison_dict["logistic_regression"]=lr_default_metrics
    models["logistic_regression"]={
        "model":lr_model,
        "optimal_threshold":lr_optimal_threshold
    }


    if lr_optimal_metrics:
           model_comparison_dict["optimal_logistic_regression"]=lr_optimal_metrics
   # if args.choice==2:
    rf_default_metrics,rf_optimal_metrics,rf_model,rf_optimal_threshold=random_forest(x_train,y_train,x_val,y_val,title="Random_Forest",sampling_T=args.Sampling_Technique,optimal_threshold=True,random_search=False)
    print(f"optimal_threshold_of RF {rf_optimal_threshold}")

    model_comparison_dict["Random_Forest"]=rf_default_metrics
    models["Random_Forest"]={
        "model":rf_model,
        "optimal_threshold":rf_optimal_threshold
    }
    if rf_optimal_metrics:
            model_comparison_dict["optimal_Random_Forest"]=rf_optimal_metrics
    #if args.choice==3:
    nn_default_metrics, nn_optimal_metrics,nn_model,nn_optimal_threshold = NN(x_train, y_train, x_val, y_val,title="NN",sampling_T=args.Sampling_Technique, optimal_threshold=True,
                                                       random_search=False)
    print(f"optimal_threshold_of NN {nn_optimal_threshold}")

    models["NN"]={
        "model":nn_model,
        "optimal_threshold":nn_optimal_threshold
    }
    model_comparison_dict["Neural Nets"] = nn_default_metrics
    if nn_optimal_metrics:
            model_comparison_dict["optimal_Neural Net"] = nn_optimal_metrics

   # if models_comparision:
    compare_models_heatmap(model_comparison_dict,args.Sampling_Technique,args.Sampling_strategy)
    save_models(models,root_path=r"F:\Machine_Learning\projects\Fraud_detection",model_type=args.Sampling_Technique)





