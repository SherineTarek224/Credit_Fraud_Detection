import  matplotlib.pyplot  as plt
from sklearn.metrics import f1_score,auc, confusion_matrix, precision_recall_curve, classification_report, make_scorer
import seaborn as sns
from numpy import argmax

def confusion_matrix_eval(y_actual, y_pred, title='Training'):
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


def precision_recall_curve_eval(y_actual, y_pred, title='Training'):
    precision, recall, thresholds = precision_recall_curve(y_actual, y_pred)

    plt.figure(figsize=(9, 9))
    plt.plot(recall, precision)
    plt.title(f"{title} Precision-Recall Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.show()
    return precision, recall

def precision_recall_different_thresholds(y_actual, y_pred, title="Training"):
    precision, recall, thresholds = precision_recall_curve(y_true=y_actual,y_score=y_pred)
    plt.figure(figsize=(9, 9))
    plt.plot(thresholds, precision[:-1], label="Precision")
    plt.plot(thresholds, recall[:-1], label='Recall')
    plt.legend()
    plt.title(f"{title} Precision-Recall vs Thresholds")
    plt.xlabel("Thresholds")
    plt.ylabel("Score")
    plt.show()


def best_threshold(y_actual, y_pred, respect='f1_score'):
    #percision,recall has 1 more element than the array of threshold
    precision, recall, thresholds = precision_recall_curve(y_actual, y_pred)

    precision=precision[:-1]
    recall=recall[:-1]

    f1_scores = ((2 * precision * recall) / (precision + recall))
    print(f"{len(precision)} / {len(recall)} / {len(thresholds)} / {len(f1_scores)}")


    if respect == 'f1_score':
        threshold_idx = argmax(f1_scores)
    elif respect == 'precision':
        threshold_idx = argmax(precision)
    elif respect == 'recall':
        threshold_idx = argmax(recall)
    else:
        raise ValueError("Invalid value for respect to. choose 'f1_score', 'precision' or 'recall'")

    print(f" threshold_index {threshold_idx}  ")
    optimal_threshold = thresholds[threshold_idx]

    print(f" Optimal Threshold {optimal_threshold} --> F1_score {f1_scores[threshold_idx]}")
    return optimal_threshold, f1_scores[threshold_idx]


def eval_optimal_threshold(model, x_val, opt_threshold):
    y_val_pred_proba = model.predict_proba(x_val)
    y_pred = (y_val_pred_proba[:, 1] >= opt_threshold).astype(int)  # probability of positive class
    return y_pred


def auc_precision_recall_eval(y_actual, y_pred):
    precision, recall, _ = precision_recall_curve(y_score=y_pred, y_true=y_actual)

    return float(auc(x=recall, y=precision))


def evaluate_model(model, title,sampling_T, x_train, y_train, x_val, y_val, optimal_threshold=False):
    y_pred_train = model.predict(x_train)
    y_pred_train_proba = model.predict_proba(x_train)[:, 1]

    confusion_matrix_eval(y_train, y_pred_train, f'{title} {sampling_T} Training')
    precision_recall_curve_eval(y_train, y_pred_train_proba, f"{title} { sampling_T} Training")
    precision_recall_different_thresholds(y_train, y_pred_train_proba, f"{title} {sampling_T} Training")

    # Validation data evaluation

    y_pred_val = model.predict(x_val)
    y_pred_val_proba = model.predict_proba(x_val)[:, 1]
    cr = confusion_matrix_eval(y_val, y_pred_val, f'{title} {sampling_T} Validation')
    precision_recall_curve_eval(y_val, y_pred_val_proba, f"{title} {sampling_T} Validation")

    auc_model = auc_precision_recall_eval(y_val, y_pred_val_proba)
    metrics_default = {

        "F1_score (Class 1)": cr['1']['f1-score'],
        "Precision (Class 1)": cr['1']['precision'],
        "Recall (Class 1)": cr['1']['recall'],
        "AUC": auc_model

    }
    metrics_optimal = None
    opt_threshold=None
    if optimal_threshold == True:
        #find best threshold on training data
        #opt_threshold, its_f1_score = best_threshold(y_train, y_pred_train_proba, respect='f1_score')#
        opt_threshold,its_f1_score=best_threshold(y_val,y_pred_val_proba,respect="f1_score")
        y_pred_val = eval_optimal_threshold(model, x_val, opt_threshold)
        cr_optimal=confusion_matrix_eval(y_val, y_pred_val, f"{title} {sampling_T} Validation with optimal threshold with respect to F1_Score")
        auc_optimal=auc_precision_recall_eval(y_val, y_pred_val_proba)
        metrics_optimal={
            "F1_score (Class 1)": cr_optimal['1']['f1-score'],
            "Precision (Class 1)": cr_optimal['1']['precision'],
            "Recall (Class 1)": cr_optimal['1']['recall'],
            "AUC": auc_optimal

        }
    return metrics_default,metrics_optimal,opt_threshold

