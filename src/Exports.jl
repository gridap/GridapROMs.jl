macro publish(mod,name)
  quote
    using GridapROMs.$mod: $name; export $name
  end
end

@publish Utils compute_speedup
@publish Utils compute_error
@publish Utils compute_relative_error
@publish Utils ∂₁
@publish Utils ∂₂
@publish Utils ∂₃
@publish Utils FEDomains
@publish Utils FEDomainOperator
@publish Utils LinearFEOperator
@publish Utils DomainOperator
@publish Utils LinearDomainOperator
@publish Utils GenericFEDomainOperator
@publish Utils GenericDomainOperator

@publish DofMaps OrderedFESpace
@publish DofMaps slow_index
@publish DofMaps fast_index
@publish DofMaps get_dof_map
@publish DofMaps get_dof_map_with_diri
@publish DofMaps get_sparse_dof_map
@publish DofMaps flatten

@publish TProduct TProductDiscreteModel
@publish TProduct TProductFESpace

@publish ParamGeometry PhysicalMap
@publish ParamGeometry DisplacementMap
@publish ParamGeometry mapped_grid

@publish ParamDataStructures Realization
@publish ParamDataStructures TransientRealization
@publish ParamDataStructures UniformSampling
@publish ParamDataStructures NormalSampling
@publish ParamDataStructures HaltonSampling
@publish ParamDataStructures ParamSpace
@publish ParamDataStructures TransientParamSpace
@publish ParamDataStructures ParamFunction
@publish ParamDataStructures TransientParamFunction
@publish ParamDataStructures realization
@publish ParamDataStructures parameterize

@publish ParamDataStructures ParamArray
@publish ParamDataStructures ConsecutiveParamArray
@publish ParamDataStructures ParamSparseMatrix
@publish ParamDataStructures BlockParamArray
@publish ParamDataStructures Snapshots
@publish ParamDataStructures select_snapshots

@publish ParamFESpaces TrialParamFESpace
@publish ParamFESpaces MultiFieldParamFESpace

@publish ParamSteady ParamTrialFESpace
@publish ParamSteady ParamOperator
@publish ParamSteady LinearParamOperator
@publish ParamSteady LinearNonlinearParamOperator
@publish ParamSteady ParamFEOperator
@publish ParamSteady LinearParamFEOperator
@publish ParamSteady LinearNonlinearParamFEOperator

@publish ParamODEs ODEParamOperator
@publish ParamODEs TransientLinearParamOperator
@publish ParamODEs TransientParamOperator
@publish ParamODEs LinearNonlinearTransientParamOperator
@publish ParamODEs TransientTrialParamFESpace
@publish ParamODEs TransientMultiFieldParamFESpace
@publish ParamODEs TransientParamFEOperator
@publish ParamODEs TransientLinearParamFEOperator
@publish ParamODEs LinearNonlinearTransientParamFEOperator
using GridapROMs.ParamODEs: ∂ₚt,∂ₚtt; export ∂ₚt,∂ₚtt

@publish Extensions DirectSumFESpace
@publish Extensions EmbeddedFESpace
@publish Extensions ExtensionOperator
@publish Extensions ExtensionParamOperator
@publish Extensions ExtensionLinearParamOperator
@publish Extensions ExtensionLinearNonlinearParamOperator
@publish Extensions TransientExtensionParamOperator
@publish Extensions TransientExtensionLinearParamOperator
@publish Extensions TransientExtensionLinearNonlinearParamOperator
@publish Extensions ZeroExtension
@publish Extensions MassExtension
@publish Extensions HarmonicExtension
@publish Extensions BlockExtension
@publish Extensions ExtensionSolver
@publish Extensions extend_solution
using GridapROMs.Extensions: ⊕; export ⊕

@publish Uncommon UncommonParamOperator
@publish Uncommon param_operator
@publish Uncommon allocate_batchvector
@publish Uncommon allocate_batchmatrix

@publish RBSteady Reduction
@publish RBSteady PODReduction
@publish RBSteady TTSVDReduction
@publish RBSteady SupremizerReduction
@publish RBSteady MDEIMHyperReduction
@publish RBSteady AdaptiveReduction

@publish RBSteady RBSolver
@publish RBSteady solution_snapshots
@publish RBSteady residual_snapshots
@publish RBSteady jacobian_snapshots

@publish RBSteady reduction
@publish RBSteady tpod
@publish RBSteady ttsvd
@publish RBSteady gram_schmidt
@publish RBSteady orth_projection

@publish RBSteady Projection
@publish RBSteady PODProjection
@publish RBSteady TTSVDProjection
@publish RBSteady NormedProjection
@publish RBSteady BlockProjection
@publish RBSteady ReducedProjection
@publish RBSteady projection
@publish RBSteady get_basis
@publish RBSteady get_cores
@publish RBSteady project
@publish RBSteady inv_project
@publish RBSteady galerkin_projection
@publish RBSteady union_bases
@publish RBSteady contraction

@publish RBSteady RBSpace
@publish RBSteady reduced_spaces

@publish RBSteady empirical_interpolation
@publish RBSteady reduced_jacobian
@publish RBSteady reduced_residual
@publish RBSteady reduced_weak_form

@publish RBSteady LocalProjection
@publish RBSteady compute_ncentroids

@publish RBSteady RBOperator
@publish RBSteady reduced_operator
@publish RBSteady change_operator

@publish RBSteady ROMPerformance
@publish RBSteady create_dir
@publish RBSteady eval_performance
@publish RBSteady plot_a_solution
@publish RBSteady load_snapshots
@publish RBSteady load_operator
@publish RBSteady load_results
@publish RBSteady load_stats

@publish RBTransient HighDimReduction
@publish RBTransient TransientReduction
@publish RBTransient HighDimHyperReduction
@publish RBTransient TransientHyperReduction
@publish RBTransient TransientProjection
@publish RBTransient TransientRBOperator
