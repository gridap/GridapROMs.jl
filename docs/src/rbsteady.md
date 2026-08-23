```@meta
CurrentModule = GridapROMs.RBSteady
```

# GridapROMs.RBSteady

This module provides the core infrastructure for Reduced Basis (RB) methods applied to steady-state parametric PDEs. It includes standard projection-based reduction techniques, operator evaluations, and solver wrappers for the offline and online phases.

## Neural Operators

In addition to classical linear ROMs, GridapROMs supports non-linear surrogate modeling via Neural Operators (DeepONet and NOMAD).
The integration relies on `NeuralOpStrategy` and `NeuralOpSolver` to manage the offline training phase using XLA-accelerated backends (Lux.jl and Reactant.jl), and provides support for Continual and Transfer Learning via model fine-tuning.

## Full API

```@autodocs
Modules = [RBSteady,]
```