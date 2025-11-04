# GridapROMs.jl 

Welcome to the documentation for GridapROMs. 

!!! note 
    The documentation is currently under construction.

## Introduction

This package provides a set of tools for the solution of parameterized partial differential equations (PDEs) with reduced order models (ROMs). The presence of parameters severely impacts the feasibility of running high-fidelity (HF) codes such as the finite element (FE) method, because typically the solution is required for many different values of the parameters. ROMs create surrogate models that approximate the solution manifold on a lower-dimensional manifold. (In linear ROMs, the manifold is a vector subspace, but more general nonlinear ROMs may be considered.) These surrogates provide accurate solutions in a much shorter time and with much fewer computational resources. Momentarily, the library supports the serial resolution of steady, transient, linear, nonlinear, single- and multi-field parameterized PDEs with the use of linear ROMs. 

## Future work

We envision two main developments for the library. Firstly, the development of a scalable, distributed-in-memory interface. Secondly, the extension to nonlinear ROMs, for example models with an autoencoder-like structure, which have alredy been proven to be effective in the literature.

## Manual

```@contents
Pages = [
  "utils.md",
  "dof_maps.md",
  "tproduct.md",
  "param_data_structures.md",
  "param_geometry.md",
  "param_fe_spaces.md",
  "param_steady.md",
  "param_odes.md",
  "param_extensions.md",
  "rbsteady.md",
  "rbtransient.md",
  ]
```