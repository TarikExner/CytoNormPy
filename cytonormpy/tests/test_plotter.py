import pytest
from types import SimpleNamespace

import cytonormpy._plotting._plotter as plotter_mod
from cytonormpy._plotting._plotter import Plotter


class DummyCN:
    """Fake CytoNorm just to pass into Plotter."""

    pass


def test_init_raises_deprecation():
    """Creating Plotter should emit a DeprecationWarning."""
    with pytest.warns(DeprecationWarning):
        Plotter(DummyCN())


@pytest.mark.parametrize(
    "method, func_name, extra_args, extra_kwargs",
    [
        ("scatter", "scatter_func", (1, 2), {"a": 3}),
        ("histogram", "histogram_func", (4,), {"b": 5}),
        ("emd", "emd_func", (), {"c": 6}),
        ("mad", "mad_func", (), {"d": 7}),
        ("splineplot", "splineplot_func", ("ch1",), {"e": 8}),
    ],
)
def test_methods_forward_to_functions(method, func_name, extra_args, extra_kwargs, monkeypatch):
    """Each Plotter.method should call its scatter_func, etc., with self.cnp first."""
    dummy_cnp = SimpleNamespace()
    with pytest.warns(DeprecationWarning):
        p = Plotter(dummy_cnp)

    sentinel = object()

    def fake_fn(cnp_arg, *args, **kwargs):
        return (cnp_arg, args, kwargs, sentinel)

    monkeypatch.setattr(plotter_mod, func_name, fake_fn)

    wrapper = getattr(p, method)
    result = wrapper(*extra_args, **extra_kwargs)

    cnp_arg, args, kwargs, out = result
    assert cnp_arg is dummy_cnp
    assert args == extra_args
    assert kwargs == extra_kwargs
    assert out is sentinel
