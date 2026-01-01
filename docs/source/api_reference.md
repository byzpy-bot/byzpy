# API Reference

The sections below group the most commonly used public modules. Each entry is
generated with Sphinx autodoc so it stays in sync with the codebase. Docstrings
follow the NumPy style and the signatures below use a color legend:

## Aggregators

```{eval-rst}
.. automodule:: byzpy.aggregators.base
   :members:
   :undoc-members:
   :show-inheritance:
```

```{eval-rst}
.. automodule:: byzpy.aggregators.coordinate_wise.median
   :members:
   :show-inheritance:
```

```{eval-rst}
.. automodule:: byzpy.aggregators.coordinate_wise.trimmed_mean
   :members:
   :show-inheritance:
```

```{eval-rst}
.. automodule:: byzpy.aggregators.coordinate_wise.mean_of_medians
   :members:
   :show-inheritance:
```

```{eval-rst}
.. automodule:: byzpy.aggregators.geometric_wise.geometric_median
   :members:
   :show-inheritance:
```

```{eval-rst}
.. automodule:: byzpy.aggregators.geometric_wise.krum
   :members:
   :show-inheritance:
```

```{eval-rst}
.. automodule:: byzpy.aggregators.geometric_wise.minimum_diameter_average
   :members:
   :show-inheritance:
```

```{eval-rst}
.. automodule:: byzpy.aggregators.geometric_wise.monna
   :members:
   :show-inheritance:
```

```{eval-rst}
.. automodule:: byzpy.aggregators.geometric_wise.smea
   :members:
   :show-inheritance:
```

```{eval-rst}
.. automodule:: byzpy.aggregators.norm_wise.center_clipping
   :members:
   :show-inheritance:
```

```{eval-rst}
.. automodule:: byzpy.aggregators.norm_wise.comparative_gradient_elimination
   :members:
   :show-inheritance:
```

```{eval-rst}
.. automodule:: byzpy.aggregators.norm_wise.caf
   :members:
   :show-inheritance:
```

## Attacks

```{eval-rst}
.. automodule:: byzpy.attacks.base
   :members:
   :undoc-members:
   :show-inheritance:
```

```{eval-rst}
.. automodule:: byzpy.attacks.empire
   :members:
   :show-inheritance:
```

```{eval-rst}
.. automodule:: byzpy.attacks.label_flip
   :members:
   :show-inheritance:
```

```{eval-rst}
.. automodule:: byzpy.attacks.little
   :members:
   :show-inheritance:
```

```{eval-rst}
.. automodule:: byzpy.attacks.sign_flip
   :members:
   :show-inheritance:
```

```{eval-rst}
.. automodule:: byzpy.attacks.gaussian
   :members:
   :show-inheritance:
```

```{eval-rst}
.. automodule:: byzpy.attacks.inf
   :members:
   :show-inheritance:
```

```{eval-rst}
.. automodule:: byzpy.attacks.mimic
   :members:
   :show-inheritance:
```

## Pre-Aggregators

```{eval-rst}
.. automodule:: byzpy.pre_aggregators.base
   :members:
   :undoc-members:
   :show-inheritance:
```

```{eval-rst}
.. automodule:: byzpy.pre_aggregators.bucketing
   :members:
   :show-inheritance:
```

```{eval-rst}
.. automodule:: byzpy.pre_aggregators.clipping
   :members:
   :show-inheritance:
```

```{eval-rst}
.. automodule:: byzpy.pre_aggregators.arc
   :members:
   :show-inheritance:
```

```{eval-rst}
.. automodule:: byzpy.pre_aggregators.nnm
   :members:
   :show-inheritance:
```

## Training Pipelines (PS & P2P)

```{eval-rst}
.. automodule:: byzpy.engine.parameter_server.ps
   :members:
   :undoc-members:
   :show-inheritance:
```

```{eval-rst}
.. automodule:: byzpy.engine.peer_to_peer.topology
   :members:
   :show-inheritance:
```

```{eval-rst}
.. automodule:: byzpy.engine.peer_to_peer.train
   :members:
   :show-inheritance:
```

## Engine APIs

### Node Applications

```{eval-rst}
.. automodule:: byzpy.engine.node.application
   :members:
   :undoc-members:
   :show-inheritance:
```

### Graph Scheduling

```{eval-rst}
.. automodule:: byzpy.engine.graph.scheduler
   :members:
   :undoc-members:
   :show-inheritance:
```

### Actor Pool

```{eval-rst}
.. automodule:: byzpy.engine.graph.pool
   :members:
   :undoc-members:
   :show-inheritance:
```

### Actor Factory

```{eval-rst}
.. automodule:: byzpy.engine.actor.factory
   :members:
   :undoc-members:
   :show-inheritance:
```
