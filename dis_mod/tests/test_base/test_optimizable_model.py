from dis_mod.base.model import OptimizableModel

def test_model_api():
    wrapper = OptimizableModel(None, None, {})
    assert hasattr(wrapper, "optimize") and callable(wrapper.optimize)
    assert hasattr(wrapper, "objective") and not callable(wrapper.objective)
    assert hasattr(wrapper, "parameter_importance") and not callable(wrapper.parameter_importance)
    assert hasattr(wrapper, "parameter_slices") and not callable(wrapper.parameter_slices)