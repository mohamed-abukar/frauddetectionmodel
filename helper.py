import matplotlib.pyplot as plt
from sklearn.metrics import classification_report,confusion_matrix
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
import pandas as pd

def preprocessDataset(data):
    """
    This method wil pre process dataset according to analysis result of part1 notebook
    """
    # step 1: remove isFlaggedFraud, nameOrg and nameDest, refer to part 1 to see the reason
    data = data.drop(['isFlaggedFraud', 'nameOrig',  'nameDest'], axis=1)
    #data = data.drop(['nameOrig', 'nameDest'], axis=1)

    # step 2: remove all records with type is not eith cashout or transfer
    #data = data.drop(data[(data.type != 'CASH_OUT') & (data.type != 'TRANSFER')].index)

    # step 3: encode attribute 'type' using ordinalEncoder
    from sklearn.preprocessing import OrdinalEncoder
    encoder = OrdinalEncoder()
    data["type"] = encoder.fit_transform(data[["type"]])

    # step 3: add new two features that records error in balances
    
    # storing balance difference for the sender
    #senderBalanceDifference = data.newbalanceOrig + data.amount - data.oldbalanceOrg
    #data.insert(loc=5, column="senderBalanceDifference", value=senderBalanceDifference)

    # storing balance difference for the receiver
    #receiverBalanceDifference = data.oldbalanceDest + data.amount - data.newbalanceDest
    #data.insert(loc=8, column="receiverBalanceDifference", value=receiverBalanceDifference)

    # Dataset is huge around 6 milion rows, we will reduce this by first dividing it into classes
    # then taking forst 50 k record of class 0 then combining it with all class 1 data
    class0 = data[data.isFraud == 0]
    class1 = data[data.isFraud == 1]

    class0 = class0.head(92000)

    # data = class0.append(class1)
    #data =  data.reset_index(drop=True) #data.sample(frac=1).reset_index(drop=True)
    return data

    

def balanceDataSet(X, y, type='undersample', under_strategy=0.5, over_strategy=0.3):
    if type == 'undersample':
        over = SMOTE(random_state=42)
        # fit and apply the transform
        X, y = over.fit_resample(X, y)

    # summarize class distribution
    #print(Counter(y))
    if type == 'undersample':
        # define undersampling strategy
        under = RandomUnderSampler(random_state=42)

        # fit and apply the transform
        X, y = under.fit_resample(X, y)

    if type == 'combine':
        over = SMOTE(sampling_strategy=over_strategy, random_state=42)
        # fit and apply the transform
        X, y = over.fit_resample(X, y)

        # define undersampling strategy
        under = RandomUnderSampler(sampling_strategy=under_strategy, random_state=42)

        # fit and apply the transform
        X, y = under.fit_resample(X, y)

    # summarize class distribution
    #print(Counter(y))
    


    #print("Resampled shape of X: ", X.shape)
    #print("Resampled shape of Y: ", y.shape)

    return X,y



# this function takes y_actual and y_predict and returns TP, FP,TN, FN
from prettytable import PrettyTable
def get_perf_messure(y_actual,y_pred):
    """
        This will calculate performance measures
        it takes y_true and y_pred as input
        it returns tuple (success_rate, error_rate , sensitivity, specificity, tbForm)
    """
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(len(y_pred)):
        if y_actual[i]==y_pred[i]==1:
            TP += 1
        if y_pred[i]==1 and y_actual[i]!=y_pred[i]:
            FP += 1
        if y_actual[i]==y_pred[i]==0:
            TN += 1
        if y_pred[i]==0 and y_actual[i]!=y_pred[i]:
            FN += 1
    tbForm = PrettyTable(['', 'Predicted Y', 'Predicted N'])
    tbForm.add_row(['Actually Y', TP, FN])
    tbForm.add_row(['Actually N', FP, TN])

    sensitivity = TP/(TP+FN)
    # Specificity or true negative rate
    specificity = TN/(TN+FP)

    success_rate = (TP + TN) / (TP + TN + FP + FN)
    error_rate = (FP + FN)  / (TP + TN + FP + FN)
    recall = TP/(TP+FN)
    precision = TP/(TP+FP)
    f1score = ( 2 * recall * precision) / ( recall+precision)

    return (success_rate, error_rate , sensitivity, specificity, tbForm)


def show_conf_mattr(y_test, y_pred, model,X_test):
    """
        This will print confusion matrix, 
        it takes y_true, y_pred, model and X_test as input values
    """
    conf_matrix = confusion_matrix(y_test, y_pred, labels=model.classes_)
    #print('Confusion matrix: \n',conf_matrix)

    #plot_confusion_matrix(estimator=model,X=X_test,y_true=y_test)
    #plt.show()

    fig, ax = plt.subplots(figsize=(3.5, 3.5))
    ax.matshow(conf_matrix, cmap=plt.cm.Blues, alpha=0.3)
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            ax.text(x=j, y=i,s=conf_matrix[i, j], va='center', ha='center', size='xx-large')

    plt.xlabel('Predictions', fontsize=12)
    plt.ylabel('Actuals', fontsize=12)
    plt.title('Confusion Matrix', fontsize=12)
    plt.show()



def display_measures(y_true, y_predicted):
    cm = confusion_matrix(y_true,y_predicted)
    clfreport = classification_report(y_true,y_predicted, zero_division=1)
    fp, recall, threshold = roc_curve(y_true, y_predicted)
    aucurv = auc(fp, recall)

    report = {"Confusion Matrix":cm,"Classification Report":clfreport,"Area Under Curve":aucurv}
    # showing results from Random Forest

    for measure in report:
        print(measure,": \n",report[measure])