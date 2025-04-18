import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from imblearn.over_sampling import SMOTE,RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN
from sklearn.model_selection import train_test_split

Random_State=17
np.random.seed(Random_State)
def load_data(training=True):
    if training:
        # may need to extract zip files first
        train=pd.read_csv(r"Dataset/train.csv")
        val=pd.read_csv(r"Dataset/val.csv")

        x_train = train.drop('Class', axis=1).values
        y_train = train['Class'].values.reshape(-1,1)
        x_val = val.drop('Class', axis=1).values
        y_val = val['Class'].values.reshape(-1,1)
        print(f"Shape of training Features and Target  {x_train.shape,y_train.shape}")
        print(f"Shape of Val Features and Target {x_val.shape,y_val.shape} ")
        return x_train, y_train, x_val, y_val
    else:
        test=pd.read_csv(r"Dataset/test.zip")
        x_test=test.drop('Class',axis=1).values
        y_test=test['Class'].values.reshape(-1,1)
        return x_test,y_test




def detect_outlier(x_train,x_val,y_train,y_val,threshold=3,method=1,clipping=True):
    lower_bound=None
    upper_bound=None
    if method==1:
        Q1=np.percentile(x_train,0.25,0)
        Q3=np.percentile(x_train,0.75,0)
        IQR=Q3-Q1
        upper_bound=Q3+3*IQR
        lower_bound=Q1-3*IQR
    elif method==2:
        mean=np.mean(x_train,axis=0)
        std=np.std(x_train,axis=0)  # axis =0 correct for column
        upper_bound=mean+std*threshold
        lower_bound=mean-std*threshold

    if clipping:

        # No DataLeakage
        x_train=np.clip(x_train,lower_bound,upper_bound)
        x_val=np.clip(x_val,lower_bound,upper_bound)
        return x_train,x_val,y_train,y_val
    else:
        # remove
        non_outlier_rows_t=((x_train >= lower_bound) & (x_train <= upper_bound)).all(axis=1)
                      # axis=1 ir correct for all features in Row
        x_train=x_train[non_outlier_rows_t]
        y_train=y_train[non_outlier_rows_t]

        non_outlier_rows_v=((x_val >= lower_bound) & (x_val <= upper_bound)).all(axis=1)
        x_val=x_val[non_outlier_rows_v]
        y_val = y_val[non_outlier_rows_v]
        print(f"x_train Shape after removing outlier{x_train.shape}")
        print(f"y_train shape after removing outliers{y_train.shape}")
        print(f"x_val shape after removing outliers{x_val.shape}")
        print(f"y_val shape after removing outlier{y_val.shape}")

    return x_train,x_val,y_train,y_val



  # create here preprcessing pipeline ?
def get_preprocessor(option=3):
    if option == 0:
        return None
    elif option == 1:
        return MinMaxScaler()
    elif option == 2:
        return StandardScaler()
    elif option == 3:
        return RobustScaler()
    else:
        raise ValueError("Invalid Scalar Type")


def preprocess_data(x_train,x_val,preprocess_option=3,preprocessor=None):
    if preprocessor is None:
       preprocessor=get_preprocessor(preprocess_option)

    if preprocessor is not None:
        if x_train is not None:
            x_train=preprocessor.fit_transform(x_train)
        if x_val is not None:
         x_val=preprocessor.transform(x_val)

    return x_train,x_val,preprocessor

def balancing_data(x_train,y_train,type="SMOTE",r_s=17,sampling_strategy=0.05,k_n=5):
    print(f"Before Sampling")
    print(f"Number of Fraud Transaction {len(y_train[y_train==1])}")
    print(f"Number of Non fraud transaction {len(y_train[y_train==0])}")

     # To use sampler the y should be flatten (1d)
    y_train=y_train.ravel()

    if type=="SMOTE":
        sampler=SMOTE(random_state=r_s,
                      sampling_strategy=sampling_strategy,k_neighbors=k_n)
    elif type=="UnderSampling":
        sampler=RandomUnderSampler(random_state=r_s,
                                   sampling_strategy=sampling_strategy)
    elif type=="OverSampling":
        sampler=RandomOverSampler(random_state=r_s,
                                  sampling_strategy=sampling_strategy)
    else:
        raise ValueError ("Invalid Sampler Type")
    x_train_s,y_train_s=sampler.fit_resample(x_train,y_train)
    print("Shape after Sampling")
    print(f"Number of Fraud Transaction {len(y_train_s[y_train_s == 1])}")
    print(f"Number of Non fraud transaction {len(y_train_s[y_train_s == 0])}")
    return x_train_s,y_train_s




if __name__=="__main__":
    x_train,y_train,x_val,y_val=load_data(training=True)
    #x_train,x_val,y_train,y_val=detect_outlier(x_train,x_val,y_train,y_val,threshold=3,method=1,clipping=False)
    x_train,x_val,preprocessor=preprocess_data(x_train,x_val,preprocess_option=3)
    train,y_train= balancing_data(x_train,y_train,type='Over')