# Inspired by GPT4

# Information on type hints
# https://peps.python.org/pep-0585/

# GPT on testing functions, mock functions, testing number of calls, and argument values
# https://chat.openai.com/share/b3fd7739-b691-48f2-bb5e-0d170be4428c


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    ShuffleSplit,
    cross_validate,
    KFold,
)

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV

from typing import Any
from numpy.typing import NDArray

import numpy as np
import utils as u

# Initially empty. Use for reusable functions across
# Sections 1-3 of the homework
import new_utils as nu


# ======================================================================
class Section1:
    def __init__(
        self,
        normalize: bool = True,
        seed: int | None = None,
        frac_train: float = 0.2,
    ):
        """
        Initializes an instance of MyClass.

        Args:
            normalize (bool, optional): Indicates whether to normalize the data. Defaults to True.
            seed (int, optional): The seed value for randomization. If None, each call will be randomized.
                If an integer is provided, calls will be repeatable.

        Returns:
            None

        Notes: notice the argument `seed`. Make sure that any sklearn function that accepts
        `random_state` as an argument is initialized with this seed to allow reproducibility.
        You change the seed ONLY in the section of run_part_1.py, run_part2.py, run_part3.py
        below `if __name__ == "__main__"`
        """
        self.normalize = normalize
        self.frac_train = frac_train
        self.seed = seed

    # ----------------------------------------------------------------------
    """
    A. We will start by ensuring that your python environment is configured correctly and 
       that you have all the required packages installed. For information about setting up 
       Python please consult the following link: https://www.anaconda.com/products/individual. 
       To test that your environment is set up correctly, simply execute `starter_code` in 
       the `utils` module. This is done for you. 
    """

    def partA(self):
        # Return 0 (ran ok) or -1 (did not run ok)
        answer = u.starter_code()
        return answer

    # ----------------------------------------------------------------------
    """
    B. Load and prepare the mnist dataset, i.e., call the prepare_data and filter_out_7_9s 
       functions in utils.py, to obtain a data matrix X consisting of only the digits 7 and 9. Make sure that 
       every element in the data matrix is a floating point number and scaled between 0 and 1 (write
       a function `def scale() in new_utils.py` that returns a bool to achieve this. Checking is not sufficient.) 
       Also check that the labels are integers. Print out the length of the filtered 𝑋 and 𝑦, 
       and the maximum value of 𝑋 for both training and test sets. Use the routines provided in utils.
       When testing your code, I will be using matrices different than the ones you are using to make sure 
       the instructions are followed. 
    """

    def partB(
        self,
    ):
        X, y, Xtest, ytest = u.prepare_data()
        Xtrain, ytrain = u.filter_out_7_9s(X, y)
        Xtest, ytest = u.filter_out_7_9s(Xtest, ytest)
        Xtrain = nu.scale_data(Xtrain)  
        Xtest = nu.scale_data(Xtest)   

        # Checking that the labels are integers
        ytrain_test = nu.scale_data_1(ytrain)
        ytest_test = nu.scale_data_1(ytest)

        print("1(B) - Are elements in Xtrain a floating point number and scaled between 0 to 1: " +str(Xtrain))
        print("1(B) - Are elements in a floating point number and scaled between 0 to 1: " +str(Xtest))
        print("1(B) - Are elements in ytrian an integer: " +str(ytrain_test))
        print("1(B) - Are elements in ytest an integer: " +str(ytest_test)) 

       
        answer = {}

        # Enter your code and fill the `answer` dictionary

        answer["length_Xtrain"] = len(Xtrain)
        answer["length_Xtest"] = len(Xtest)
        answer["length_ytrain"] = len(ytrain)
        answer["length_ytest"] = len(ytest)
        answer["max_Xtrain"] = np.max(Xtrain)
        answer["max_Xtest"] = np.max(Xtest)
        return answer, Xtrain, ytrain, Xtest, ytest

    """
    C. Train your first classifier using k-fold cross validation (see train_simple_classifier_with_cv 
       function). Use 5 splits and a Decision tree classifier. Print the mean and standard deviation 
       for the accuracy scores in each validation set in cross validation. (with k splits, cross_validate
       generates k accuracy scores.)  
       Remember to set the random_state in the classifier and cross-validator.
    """

    # ----------------------------------------------------------------------
    def partC(
        self,
        X: NDArray[np.floating],
        y: NDArray[np.int32],
    ):

        # Enter your code and fill the `answer` dictionary
        answer = {}
        n_splits=5

        clf = DecisionTreeClassifier(random_state=self.seed)

        # Initializing KFold cross-validator with 5 splits
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=self.seed)

        answer["clf"] = clf
        answer["cv"] = cv  # the cross validator instance
        
        # Training the classifier using k-fold cross-validation
        scores= u.train_simple_classifier_with_cv(Xtrain=X, ytrain=y, clf=clf, cv=cv)

        mean_accuracy = scores["test_score"].mean()
        std_accuracy = scores["test_score"].std()
        mean_fit_time = scores["fit_time"].mean()
        std_fit_time = scores["fit_time"].std()

        scores_dict = {
            "mean_accuracy": mean_accuracy,
            "std_accuracy": std_accuracy,
            "mean_fit_time": mean_fit_time,
            "std_fit_time": std_fit_time,
        }
        answer["scores"] = scores_dict
        return answer
    # ---------------------------------------------------------
    """
    D. Repeat Part C with a random permutation (Shuffle-Split) 𝑘-fold cross-validator.
    Explain the pros and cons of using Shuffle-Split versus 𝑘-fold cross-validation.
    """

    def partD(
        self,
        X: NDArray[np.floating],
        y: NDArray[np.int32],
    ):
        # Enter your code and fill the `answer` dictionary
        n_splits = 5
        answer = {}

        # Creating a decision tree classifier
        clf = DecisionTreeClassifier(random_state=self.seed)
        
        # Creating a Shuffle-Split cross-validator
        cv = ShuffleSplit(n_splits=n_splits, random_state=self.seed)

        answer["clf"] = clf
        answer["cv"] = cv

        scores= u.train_simple_classifier_with_cv(Xtrain=X, ytrain=y, clf=clf, cv=cv)
        mean_accuracy = scores["test_score"].mean()
        std_accuracy = scores["test_score"].std()
        mean_fit_time = scores["fit_time"].mean()
        std_fit_time = scores["fit_time"].std()

        scores_dict = {
            "mean_accuracy": mean_accuracy,
            "std_accuracy": std_accuracy,
            "mean_fit_time": mean_fit_time,
            "std_fit_time": std_fit_time,
        }
        answer["scores"] = scores_dict
        answer["explain_kfold_vs_shuffle_split"] = '''
        k-fold Cross-Validation:
        Pros: Stable estimate, valuable for smaller datasets.
        Cons: Computationally expensive, may not suit ordered data.
    
        Shuffle-Split Cross-Validation:
        Pros: Efficient for larger datasets, suitable for ordered data.
        Cons: Less stable estimates, potential exclusion of data points.
        '''

        return answer
    # ----------------------------------------------------------------------
    """
    E. Repeat part D for 𝑘=2,5,8,16, but do not print the training time. 
       Note that this may take a long time (2–5 mins) to run. Do you notice 
       anything about the mean and/or standard deviation of the scores for each k?
    """

    def partE(
        self,
        X: NDArray[np.floating],
        y: NDArray[np.int32],
    ):
        n_splits = [2, 5, 8, 16]
        answer = {}

        for n_split in n_splits:
            cv = ShuffleSplit(n_splits=n_split, random_state=self.seed)
            clf = DecisionTreeClassifier(random_state=self.seed)

            scores= u.train_simple_classifier_with_cv(Xtrain=X, ytrain=y, clf=clf, cv=cv)

            mean_accuracy = scores["test_score"].mean()
            std_accuracy = scores["test_score"].std()
            #mean_fit_time = scores["fit_time"].mean()
            #std_fit_time = scores["fit_time"].std()

            scores_dict = {
                "mean_accuracy": mean_accuracy,
                "std_accuracy": std_accuracy,
                #"mean_fit_time": mean_fit_time,
                #"std_fit_time": std_fit_time,
            }

            all_data = {}
            all_data["scores"] = scores_dict
            all_data["cv"] = cv
            all_data["clf"] = clf
            answer[n_split] = all_data
            """
        answer = {
            2: {
                'scores': scores_dict,
                'cv': ShuffleSplit(n_splits=n_split, random_state=self.seed),
                'clf': DecisionTreeClassifier(random_state=self.seed)
                },
            5: {
                'scores': scores_dict,
                'cv': ShuffleSplit(n_splits=n_split, random_state=self.seed),
                'clf': DecisionTreeClassifier(random_state=self.seed)
               },
            8: {
                'scores': scores_dict,
                'cv': ShuffleSplit(n_splits=n_split, random_state=self.seed),
                'clf': DecisionTreeClassifier(random_state=self.seed)
               },
            16: {
                'scores': scores_dict,
                'cv': ShuffleSplit(n_splits=n_split, random_state=self.seed),
                'clf': DecisionTreeClassifier(random_state=self.seed)
                }
        }
"""
        return answer

    # ----------------------------------------------------------------------
    """
    F. Repeat part D with a Random-Forest classifier with default parameters. 
       Make sure the train test splits are the same for both models when performing 
       cross-validation. (Hint: use the same cross-validator instance for both models.)
       Which model has the highest accuracy on average? 
       Which model has the lowest variance on average? Which model is faster 
       to train? (compare results of part D and part F)

       Make sure your answers are calculated and not copy/pasted. Otherwise, the automatic grading 
       will generate the wrong answers. 
       
       Use a Random Forest classifier (an ensemble of DecisionTrees). 
    """

    def partF(
        self,
        X: NDArray[np.floating],
        y: NDArray[np.int32],
    ) -> dict[str, Any]:
        
        n_splits=5
        clf = RandomForestClassifier(random_state=self.seed)
        cv = ShuffleSplit(n_splits=n_splits, random_state=self.seed)
        
        scores= u.train_simple_classifier_with_cv(Xtrain=X, ytrain=y, clf=clf, cv=cv)
        
        mean_accuracy = scores["test_score"].mean()
        std_accuracy = scores["test_score"].std()
        mean_fit_time = scores["fit_time"].mean()
        std_fit_time = scores["fit_time"].std()

        scores_dict = {
            "mean_accuracy": mean_accuracy,
            "std_accuracy": std_accuracy,
            "mean_fit_time": mean_fit_time,
            "std_fit_time": std_fit_time,
        }
        answer = {}
        answer["clf_RF"] = clf
        answer["cv"] = cv
        answer["scores_RF"] = scores_dict

        DT_answer = self.partD(X, y)
        answer["clf_DT"] = DT_answer["clf"]
        answer["scores_DT"] = DT_answer["scores"]
        
        DT_score = answer["scores_DT"]["mean_accuracy"]
        RF_score = answer["scores_RF"]["mean_accuracy"]
        DT_fit_time = answer["scores_DT"]["mean_fit_time"]
        RF_fit_time = answer["scores_RF"]["mean_fit_time"]

        DT_var = answer["scores_DT"]["std_accuracy"] ** 2
        RF_var = answer["scores_RF"]["std_accuracy"] ** 2

        if DT_score > RF_score:     
           answer["model_highest_accuracy"] = "Decision-tree"
        else:
           answer["model_highest_accuracy"] = "Random-forest"

        answer["model_lowest_variance"] = min(RF_var, DT_var)
        answer["model_fastest"] = min(RF_fit_time, DT_fit_time)

        """
         Answer is a dictionary with the following keys: 
            "clf_RF",  # Random Forest class instance
            "clf_DT",  # Decision Tree class instance
            "cv",  # Cross validator class instance
            "scores_RF",  Dictionary with keys: "mean_fit_time", "std_fit_time", "mean_accuracy", "std_accuracy"
            "scores_DT",  Dictionary with keys: "mean_fit_time", "std_fit_time", "mean_accuracy", "std_accuracy"
            "model_highest_accuracy" (string)
            "model_lowest_variance" (float)
            "model_fastest" (float)
        """

        return answer
    """
    G. For the Random Forest classifier trained in part F, manually (or systematically, 
       i.e., using grid search), modify hyperparameters, and see if you can get 
       a higher mean accuracy.  Finally train the classifier on all the training 
       data and get an accuracy score on the test set.  Print out the training 
       and testing accuracy and comment on how it relates to the mean accuracy 
       when performing cross validation. Is it higher, lower or about the same?

       Choose among the following hyperparameters: 
         1) criterion, 
         2) max_depth, 
         3) min_samples_split, 
         4) min_samples_leaf, 
         5) max_features 
    """

    def partG(
        self,
        X: NDArray[np.floating],
        y: NDArray[np.int32],
        Xtest: NDArray[np.floating],
        ytest: NDArray[np.int32],
    ) -> dict[str, Any]:
        """
        Perform classification using the given classifier and cross validator.

        Parameters:
        - clf: The classifier instance to use for classification.
        - cv: The cross validator instance to use for cross validation.
        - X: The test data.
        - y: The test labels.
        - n_splits: The number of splits for cross validation. Default is 5.

        Returns:
        - y_pred: The predicted labels for the test data.

        Note:
        This function is not fully implemented yet.
        """
        # refit=True: fit with the best parameters when complete
        # A test should look at best_index_, best_score_ and best_params_
        """
        List of parameters you are allowed to vary. Choose among them.
         1) criterion,
         2) max_depth,
         3) min_samples_split, 
         4) min_samples_leaf,
         5) max_features 
         5) n_estimators
        """

        # Enter your code, construct the `answer` dictionary, and return it.


        """
           `answer`` is a dictionary with the following keys: 
            
            "clf", base estimator (classifier model) class instance
            "default_parameters",  dictionary with default parameters 
                                   of the base estimator
            "best_estimator",  classifier class instance with the best
                               parameters (read documentation)
            "grid_search",  class instance of GridSearchCV, 
                            used for hyperparameter search
            "mean_accuracy_cv",  mean accuracy score from cross 
                                 validation (which is used by GridSearchCV)
            "confusion_matrix_train_orig", confusion matrix of training 
                                           data with initial estimator 
                                (rows: true values, cols: predicted values)
            "confusion_matrix_train_best", confusion matrix of training data 
                                           with best estimator
            "confusion_matrix_test_orig", confusion matrix of test data
                                          with initial estimator
            "confusion_matrix_test_best", confusion matrix of test data
                                            with best estimator
            "accuracy_orig_full_training", accuracy computed from `confusion_matrix_train_orig'
            "accuracy_best_full_training"
            "accuracy_orig_full_testing"
            "accuracy_best_full_testing"
               
        """

        clf = RandomForestClassifier(random_state=self.seed)
        clf.fit(X, y)
        
        default_parameters = {
            "criterion": "gini",
            "max_features": 100,
            "n_estimators": 100,
        }

        parameters = {
            "criterion": ["entropy", "gini", "log_loss"],
            "max_features": [50], 
            "n_estimators": [200],
        }

        #clf = RandomForestClassifier(random_state=self.seed)
        grid_search = GridSearchCV(
            clf, param_grid=parameters, refit=True, cv=5, return_train_score=True
        )
        # Grid search with cv, then fits the training data
        grid_search.fit(X, y)
        best_estimator = grid_search.best_estimator_

        clf.fit(X, y)
        best_estimator.fit(X, y)

        ytest_pred_best = best_estimator.predict(Xtest)
        ytest_pred_orig = clf.predict(Xtest)

        print(confusion_matrix(ytest, ytest_pred_best))

        ytrain_pred_best = best_estimator.predict(X)
        ytrain_pred_orig = clf.predict(X)

        answer = {}
        answer["clf"] = RandomForestClassifier(random_state=self.seed)
        answer["best_estimator"] = best_estimator
        answer["default_parameters"] = default_parameters
        answer["grid_search"] = grid_search
        answer["mean_accuracy_cv"] = grid_search.best_score_

        cm_train_orig = confusion_matrix(ytrain_pred_orig, y)
        cm_train_best = confusion_matrix(ytrain_pred_best, y)
        cm_test_orig = confusion_matrix(ytest_pred_orig, ytest)
        cm_test_best = confusion_matrix(ytest_pred_best, ytest)

        score_train_orig = (cm_train_orig[0, 0] + cm_train_orig[1, 1]) / y.size
        score_train_best = (cm_train_orig[0, 0] + cm_train_best[1, 1]) / y.size
        score_test_orig = (cm_test_orig[0, 0] + cm_test_orig[1, 1]) / y.size
        score_test_best = (cm_test_orig[0, 0] + cm_test_best[1, 1]) / y.size

        answer["confusion_matrix_train_orig"] = cm_train_orig
        answer["confusion_matrix_train_best"] = cm_train_best
        answer["confusion_matrix_test_orig"] = cm_test_orig
        answer["confusion_matrix_test_best"] = cm_test_best

        answer["accuracy_orig_full_training"] = score_train_orig
        answer["accuracy_best_full_training"] = score_train_best
        answer["accuracy_orig_full_testing"] = score_test_orig
        answer["accuracy_best_full_testing"] = score_test_best


        return answer
