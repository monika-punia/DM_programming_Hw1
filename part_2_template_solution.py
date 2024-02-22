# Add your imports here.
# Note: only sklearn, numpy, utils and new_utils are allowed.

import numpy as np
from numpy.typing import NDArray
from typing import Any, Dict, Tuple
import utils as u
import new_utils as nu
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import ShuffleSplit
from part_1_template_solution import Section1 as Part1
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import (
    ShuffleSplit,
    cross_validate,
    KFold,
)

# ======================================================================

# I could make Section 2 a subclass of Section 1, which would facilitate code reuse.
# However, both classes have the same function names. Better to pass Section 1 instance
# as an argument to Section 2 class constructor.


class Section2:
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
        """
        self.normalize = normalize
        self.seed = seed
        self.frac_train = frac_train
        self.part1 = Part1(seed=seed, frac_train=frac_train)

    # ---------------------------------------------------------

    """
    A. Repeat part 1.B but make sure that your data matrix (and labels) consists of
        all 10 classes by also printing out the number of elements in each class y and 
        print out the number of classes for both training and testing datasets. 
    """

    def partA(
        self,
    ) -> tuple[
        dict[str, Any],
        NDArray[np.floating],
        NDArray[np.int32],
        NDArray[np.floating],
        NDArray[np.int32],
    ]:
        answer = {}
        Xtrain, ytrain, Xtest, ytest = u.prepare_data()
        Xtrain = nu.scale_data(Xtrain)
        Xtest = nu.scale_data(Xtest)

        # Check that labels contain all classes
        # count number of elements of each class in train and test sets

  
        # Enter your code and fill the `answer`` dictionary

        # `answer` is a dictionary with the following keys:
        # - nb_classes_train: number of classes in the training set
        # - nb_classes_test: number of classes in the testing set
        # - class_count_train: number of elements in each class in the training set
        # - class_count_test: number of elements in each class in the testing set
        # - length_Xtrain: number of elements in the training set
        # - length_Xtest: number of elements in the testing set
        # - length_ytrain: number of labels in the training set
        # - length_ytest: number of labels in the testing set
        # - max_Xtrain: maximum value in the training set
        # - max_Xtest: maximum value in the testing set

        # return values:
        # Xtrain, ytrain, Xtest, ytest: the data used to fill the `answer`` dictionary
        answer['nb_classes_train'] = len(np.unique(ytrain))
        answer['nb_classes_test'] = len(np.unique(ytest))
        answer['class_count_train'] = np.bincount(ytrain)
        answer['class_count_test'] = np.bincount(ytest)
        answer['length_Xtrain'] = len(Xtrain)
        answer['length_Xtest'] = len(Xtest)
        answer['length_ytrain'] = len(ytrain)
        answer['length_ytest'] = len(ytest)
        answer['max_Xtrain'] = np.max(Xtrain.flatten())
        answer['max_Xtest'] = np.max(Xtest)

        return answer, Xtrain, ytrain, Xtest, ytest

    """
    B.  Repeat part 1.C, 1.D, and 1.F, for the multiclass problem. 
        Use the Logistic Regression for part F with 300 iterations. 
        Explain how multi-class logistic regression works (inherent, 
        one-vs-one, one-vs-the-rest, etc.).
        Repeat the experiment for ntrain=1000, 5000, 10000, ntest = 200, 1000, 2000.
        Comment on the results. Is the accuracy higher for the training or testing set?
        What is the scores as a function of ntrain.

        Given X, y from mnist, use:
        Xtrain = X[0:ntrain, :]
        ytrain = y[0:ntrain]
        Xtest = X[ntrain:ntrain+test]
        ytest = y[ntrain:ntrain+test]
    """

    def partB(
        self,
        X: NDArray[np.floating],
        y: NDArray[np.int32],
        Xtest: NDArray[np.floating],
        ytest: NDArray[np.int32],
        ntrain_list: list[int] = [],
        ntest_list: list[int] = [],
    ) -> dict[int, dict[str, Any]]:

        # Enter your code and fill the `answer`` dictionary
        answer = {}
        
        def partB_sub(X, y, Xtest, ytest):
            answer = self.part1.partC(X, y)
            c_scores = answer["scores"]
            partC = {"scores_C": c_scores, "clf": answer["clf"], "cv": answer["cv"]}
            
            answer_D = self.part1.partD(X, y)
            d_scores = answer_D["scores"]
            partD = {"scores_D": d_scores, "clf": answer_D["clf"], "cv": answer_D["cv"]}


            cv = ShuffleSplit(n_splits=5, random_state=self.seed)
            clf = LogisticRegression(
                random_state=self.seed, multi_class="multinomial", max_iter=300
            )
            
            scores= u.train_simple_classifier_with_cv(Xtrain=X, ytrain=y, clf=clf, cv=cv)
            clf.fit(X, y)
            scores_train_F = clf.score(X, y)  
            scores_test_F = clf.score(Xtest, ytest) 
            mean_cv_accuracy_F = scores["test_score"].mean()

            y_pred = clf.predict(X)
            ytest_pred = clf.predict(Xtest)

            conf_mat_train = confusion_matrix(y_pred, y)
            conf_mat_test = confusion_matrix(ytest_pred, ytest)

            partF = {
                "scores_train_F": scores_train_F,
                "scores_test_F": scores_test_F,
                "mean_cv_accuracy_F": mean_cv_accuracy_F,
                "clf": clf,
                "cv": cv,
                "conf_mat_train": conf_mat_train,
                "conf_mat_test": conf_mat_test,
            }

            
            answer = {}
            answer["partC"] = partC
            answer["partD"] = partD
            answer["partF"] = partF
            answer["ntrain"] = len(y)
            answer["ntest"] = len(ytest)
            answer["class_count_train"] = np.unique(y, return_counts=True)[1]
            answer["class_count_test"] = np.unique(ytest, return_counts=True)[1]
            return answer
            


        for ntr, nte in zip(ntrain_list, ntest_list):
            X_r = X[0:ntr, :]
            y_r = y[0:ntr]
            Xtest_r = Xtest[0:nte, :]
            ytest_r = ytest[0:nte]
            answer[ntr] = partB_sub(X_r, y_r, Xtest_r, ytest_r)

        return answer
