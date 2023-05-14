from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score, roc_curve, auc, precision_score
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
import pandas as pd

class imbalance:
    """
    This class will handle imbalanced datasets;
    inputs: 
        imb_ratio = 0.1
        class_labels = (0,1)
        prediction = 'class' it can be either 'class' or 'probability'
        bal_ratio = 1
        resamble_scenario = 'combination', it can be 'oversample', 'undersample'
        resample_type = 'all', 'split_resamble', 'resamble_split'




    """
    def __init__(self, dataset, imb_ratio=0.1, class_labels=[0,1], prediction='class', bal_ratio=1, 
            resamble_scenario='undersample', resamble_type='all', model='DecisionTree', over_strategy=1, 
            under_strategy=1, class_attr='class', measure_factors = {'positive_class':1}):
        self.dataset = dataset
        self.class_attr = class_attr
        self.imb_ratio = imb_ratio
        self.class_labels = class_labels
        self.prediction = prediction
        self.bal_ratio = bal_ratio
        self.resamble_type = resamble_type
        self.over_strategy = over_strategy
        self.under_strategy = under_strategy
        self.resamble_scenario = resamble_scenario
        self.model = model
        self.binary_class = (len(class_labels) == 2)
        self.measure_factors = measure_factors
        self.FN = 1
        self.FP = 1

    def balance(self):
        if self.resamble_scenario == 'oversample':
            if self.resamble_type == 'all':
                over = SMOTE(random_state=42)
                X = self.dataset.drop(self.class_attr, axis=1)
                y = self.dataset[self.class_attr]
                X, y = over.fit_resample(X, y)
                X[self.class_attr] = y
                return X
        if self.resamble_scenario == 'undersample':
                under = RandomUnderSampler(random_state=42)
                X = self.dataset.drop(self.class_attr, axis=1)
                y = self.dataset[self.class_attr]
                X, y = under.fit_resample(X, y)
                X[self.class_attr] = y
                return X
        if self.resamble_scenario == 'combine':
                over = SMOTE(random_state=42, sampling_strategy=self.over_strategy)
                X = self.dataset.drop(self.class_attr, axis=1)
                y = self.dataset[self.class_attr]
                X, y = over.fit_resample(X, y)

                under = RandomUnderSampler(random_state=42, sampling_strategy = self.under_strategy)
                X = self.dataset.drop(self.class_attr, axis=1)
                y = self.dataset[self.class_attr]
                X, y = under.fit_resample(X, y)

                X[self.class_attr] = y
                return X


    def measure_metric(self):
        """
        This function will just give you an idea on what measure to use based on stakeholders decision
        """
        if self.prediction == 'probability':
            if self.binary_class == True:
                if self.measure_factors.positive_class == 1:
                    return 'Oercision-Recall AUC'
                else:
                    return 'ROC AUC'
            else:
                return 'Brier Score'
        
        if self.prediction == 'class':
            if self.binary_class == True:
                if self.measure_factors['positive_class'] == 1:
                    if self.FN == 1 and self.FP == 1:
                        return 'F1-Score'
                    if self.FP < 1:
                        return 'F2-SCORE'
                    if self.FN < 1:
                        return 'F0.5-SCORE'
                if self.measure_factors['positive_class'] < 1:
                    return 'G-Mean'
                
