# CytoNormPy Public API

This is CytoNormPy's Public API reference.
Import cytonormpy with the following line:
```
import cytonormpy as cnp
```

<br/><br/>
Main tasks have been divided into the following classes:

```{eval-rst}

.. module:: cytonormpy
.. currentmodule:: cytonormpy

.. autosummary::
    :toctree: ../generated/
    :nosignatures:
    
    CytoNorm

```


```{eval-rst}

.. currentmodule:: cytonormpy

.. autosummary::
    :toctree: ../generated/
    :nosignatures:
    
    Plotter

```


<br/><br/>
Clustering can be achieved using one the four implemented clustering algorithms:

```{eval-rst}

.. currentmodule:: cytonormpy

.. autosummary::
    :toctree: ../generated/
    :nosignatures:
    
    FlowSOM
    KMeans
    MeanShift
    AffinityPropagation
```


<br/><br/>
Implemented transformations include Asinh, Log, Logicle and Hyperlog.

```{eval-rst}

.. currentmodule:: cytonormpy

.. autosummary::
    :toctree: ../generated/
    :nosignatures:
    
    AsinhTransformer
    LogTransformer
    LogicleTransformer
    HyperLogTransformer
```

<br/><br/>
In order to read the model, use the respective utility functions.

```{eval-rst}

.. currentmodule:: cytonormpy

.. autosummary::
    :toctree: ../generated/
    :nosignatures:

    read_model
```

