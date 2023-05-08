import numpy as np
from scipy import stats

import mysklearn.myutils as utils
from mysklearn.mysimplelinearregressor import MySimpleLinearRegressor
from mysklearn.myclassifiers import MySimpleLinearRegressionClassifier,\
    MyKNeighborsClassifier,\
    MyDummyClassifier
    
from sklearn.neighbors import KNeighborsClassifier
    

# from in-class #1  (4 instances)
X_train_class_example1 = [[1, 1], 
                          [1, 0], 
                          [2/6, 0], 
                          [0, 0]]
y_train_class_example1 = ["bad", "bad", "good", "good"]
    

# from in-class #2 (8 instances)
# assume normalized
X_train_class_example2 = [
    [3, 2],
    [6, 6],
    [4, 1],
    [4, 4],
    [1, 2],
    [2, 0],
    [0, 3],
    [1, 6]]

y_train_class_example2 = ["no", "yes", "no", "no", "yes", "no", "yes", "yes"]

# from Bramer
header_bramer_example = ["Attribute 1", "Attribute 2"]
X_train_bramer_example = [
    [0.8, 6.3],
    [1.4, 8.1],
    [2.1, 7.4],
    [2.6, 14.3],
    [6.8, 12.6],
    [8.8, 9.8],
    [9.2, 11.6],
    [10.8, 9.6],
    [11.8, 9.9],
    [12.4, 6.5],
    [12.8, 1.1],
    [14.0, 19.9],
    [14.2, 18.5],
    [15.6, 17.4],
    [15.8, 12.2],
    [16.6, 6.7],
    [17.4, 4.5],
    [18.2, 6.9],
    [19.0, 3.4],
    [19.6, 11.1]]

y_train_bramer_example = ["-", "-", "-", "+", "-", "+", "-", "+", "+", "+", "-", "-", "-",
                          "-", "-", "+", "+", "+", "-", "+"]

# note: order is actual/received student value, expected/solution

def high_low_discretizer(value):
    if value <= 100:
        return "low"
    return "high"

def test_simple_linear_regression_classifier_fit():
    # TDD: test driven development
    # write unit tests before write units themselves
    # fully understand how to write the unit
    # if you full understand how to know it is correct
    np.random.seed(0)
    X_train = [[value] for value in range(100)]
    y_train = [row[0] * 2 + np.random.normal(0, 25) for row in X_train]
    
    lin_clf = MySimpleLinearRegressionClassifier(high_low_discretizer)
    lin_clf.fit(X_train, y_train) # "fits" slope (m) and intercept (b)
    
    # assert against "desk calculation" or "desk check"
    slope_solution = 1.924917458430444
    intercept_solution = 5.211786196055144
    # order: actual, expected (solution)
    assert np.isclose(lin_clf.regressor.slope, slope_solution)
    assert np.isclose(lin_clf.regressor.intercept, intercept_solution)


def test_simple_linear_regression_classifier_predict():
    lin_clf = MySimpleLinearRegressionClassifier(high_low_discretizer, MySimpleLinearRegressor(2,10)) # y = 2x + 10
    
    X_test = [[78], [12], [7]]
    y_predicted_solution = ["high", "low", "low"]
    y_predicted =  lin_clf.predict(X_test)
    assert y_predicted == y_predicted_solution


def test_kneighbors_classifier_kneighbors():
    # Use the 4 instance training set example traced in class on the iPad, asserting against our desk check
    
    # Note: MyKNeighborsClassifier assumes data is already normalized!! 
    #       Normalization is a preprocessing step 
    
    example1_clf = MyKNeighborsClassifier() # by default it takes 3 neighbors
    example1_clf.fit(X_train_class_example1, y_train_class_example1)
    
    X_test_ex1 = [[2/6,1]] # the unseen instance normalized
    ex1_distances, ex1_neighbor_indices = example1_clf.kneighbors(X_test_ex1)
    
    desk_check_indices = [0, 2, 3]
    desk_check_distances = [0.67, 1.0, 1.053]
    
    print(ex1_distances)
    print(ex1_neighbor_indices)
    
    assert ex1_neighbor_indices == desk_check_indices
    assert np.allclose(ex1_distances, desk_check_distances, 1e-2)
    
    


def test_kneighbors_classifier_predict():
    # Use the 4 instance training set example traced in class on the iPad, asserting against our desk check
    example1_clf = MyKNeighborsClassifier() # by default it takes 3 neighbors
    example1_clf.fit(X_train_class_example1, y_train_class_example1)
    
    X_test_ex1 = [[2/6,1]] # the unseen instance normalized
    
    y_predicted_ex1 = example1_clf.predict(X_test_ex1)
    y_actual = ["good"]
    
    assert y_predicted_ex1 == y_actual


def test_dummy_classifier_fit():
    # Note: in this case, "yes" should be more frequent than "no"
    y_train_1 = list(np.random.choice(["yes", "no"], 100, replace=True, p=[0.7, 0.3]))
    X_train_1 = [value for value in range(100)]
    
    dummy_clf_1 = MyDummyClassifier()
    dummy_clf_1.fit(X_train_1, y_train_1)
    assert dummy_clf_1.most_common_label == "yes"
    
    # Note: in this case, "no" should be more frequent than "yes" and "maybe"
    y_train_2 = list(np.random.choice(["yes", "no", "maybe"], 100, replace=True, p=[0.2, 0.6, 0.2]))
    X_train_2 = [value for value in range(100)]
    dummy_clf_2 = MyDummyClassifier()
    dummy_clf_2.fit(X_train_2, y_train_2)
    assert dummy_clf_2.most_common_label == "no"
    
    # this most frequent label can either be acidic or basic since they are equally frequent 
    y_train_3 = ["acidic", "acidic", "acidic", "acidic", "acidic", "basic", "basic", "basic", "basic", "basic", "neutral"]
    X_train_3 = [5.5, 2.3, 5.5, 6.2, 4.0, 8.9, 8.0, 7.9, 10.2, 11.3, 7.0]
    dummy_clf_3 = MyDummyClassifier()
    dummy_clf_3.fit(X_train_3, y_train_3)
    assert (dummy_clf_3.most_common_label == "acidic" or dummy_clf_3.most_common_label == "basic") == True
    


def test_dummy_classifier_predict():
    # Use the same test cases as for fit()
    
    # Note: in this case, "yes" should be more frequent than "no"
    y_train_1 = list(np.random.choice(["yes", "no"], 100, replace=True, p=[0.7, 0.3]))
    X_train_1 = [value for value in range(100)]
    
    dummy_clf_1 = MyDummyClassifier()
    dummy_clf_1.fit(X_train_1, y_train_1)
    
    X_test_1 = [[2],[102],[102]]
    y_predicted_solution_1 = ["yes", "yes", "yes"]
    y_predicted_1 = dummy_clf_1.predict(X_test_1)
    assert y_predicted_1 == y_predicted_solution_1
    
    # Note: in this case, "no" should be more frequent than "yes" and "maybe"
    y_train_2 = list(np.random.choice(["yes", "no", "maybe"], 100, replace=True, p=[0.2, 0.6, 0.2]))
    X_train_2 = [value for value in range(100)]
    
    dummy_clf_2 = MyDummyClassifier()
    dummy_clf_2.fit(X_train_2, y_train_2)
    
    X_test_2 = [[205], [110], [5], [12]]
    y_predicted_solution_2 = ["no", "no", "no", "no"]
    y_predicted_2 = dummy_clf_2.predict(X_test_2)
    assert y_predicted_2 == y_predicted_solution_2
    
    # this most frequent label can either be acidic or basic since they are equally frequent 
    y_train_3 = ["acidic", "acidic", "acidic", "acidic", "acidic", "basic", "basic", "basic", "basic", "basic", "neutral"]
    X_train_3 = [5.5, 2.3, 5.5, 6.2, 4.0, 8.9, 8.0, 7.9, 10.2, 11.3, 7.0]
    
    dummy_clf_3 = MyDummyClassifier()
    dummy_clf_3.fit(X_train_3, y_train_3)
    
    X_test_3 = [[8.2], [1.2], [9.7]]
    basic = ["basic", "basic", "basic"]
    acidic = ["acidic", "acidic", "acidic"]
    
    y_predicted_3 = dummy_clf_3.predict(X_test_3)
    assert y_predicted_3 == basic or y_predicted_3 == acidic
