from models.base.metric import SupervisedMetric, UnsupervisedMetric
import pytest

def test_supervised_metric():
    # Ensure the supervised metric has an as_objective function with the correct parameters
    with pytest.raises(NotImplementedError):
        SupervisedMetric().as_objective(model=None, X=None, y=None)

    # Ensure the unsupervised metric is callable with the correct parameters
    with pytest.raises(NotImplementedError):
        SupervisedMetric()(y_true=None, y_pred=None)
        SupervisedMetric()(None, None, None, some=None)

def test_unsupervised_metric():
    # Ensure the unsupervised metric has an as_objective function with the correct parameters
    with pytest.raises(NotImplementedError):
        UnsupervisedMetric().as_objective(model=None, X=None, y=None)

    # Ensure the unsupervised metric is callable with the correct parameters
    with pytest.raises(NotImplementedError):
        UnsupervisedMetric()(X=None, y_pred=None)
        UnsupervisedMetric()(None, None, None, some=None)