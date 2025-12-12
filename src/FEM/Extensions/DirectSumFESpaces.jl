"""
    struct DirectSumFESpace{S<:SingleFieldFESpace,T<:SingleFieldFESpace} <: SingleFieldFESpace
      space::EmbeddedFESpace{S,T}
      complementary::EmbeddedFESpace
    end

FE space which essentially acts as a wrapper around an [`EmbeddedFESpace`](@ref)
`space`, but also stores its complementary space `complementary` obtained by calling
[`complementary_space`](@ref). This interface is useful for defining extension
operators from `space` onto `complementary`
"""
struct DirectSumFESpace{S<:SingleFieldFESpace,T<:SingleFieldFESpace} <: SingleFieldFESpace
  space::EmbeddedFESpace{S,T}
  complementary::EmbeddedFESpace
end

function DirectSumFESpace(bg_space::SingleFieldFESpace,act_space::SingleFieldFESpace)
  space = EmbeddedFESpace(act_space,bg_space)
  complementary = complementary_space(space)
  DirectSumFESpace(space,complementary)
end

FESpaces.ConstraintStyle(::Type{<:DirectSumFESpace{S}}) where S = ConstraintStyle(S)

FESpaces.get_free_dof_ids(f::DirectSumFESpace) = get_free_dof_ids(f.space)

FESpaces.get_triangulation(f::DirectSumFESpace) = get_triangulation(f.space)

FESpaces.get_dof_value_type(f::DirectSumFESpace) = get_dof_value_type(f.space)

FESpaces.get_cell_dof_ids(f::DirectSumFESpace) = get_cell_dof_ids(f.space)

FESpaces.get_fe_basis(f::DirectSumFESpace) = get_fe_basis(f.space)

FESpaces.get_trial_fe_basis(f::DirectSumFESpace) = get_trial_fe_basis(f.space)

FESpaces.get_fe_dof_basis(f::DirectSumFESpace) = get_fe_dof_basis(f.space)

FESpaces.get_cell_isconstrained(f::DirectSumFESpace) = get_cell_isconstrained(f.space)

FESpaces.get_cell_constraints(f::DirectSumFESpace) = get_cell_constraints(f.space)

FESpaces.get_dirichlet_dof_ids(f::DirectSumFESpace) = get_dirichlet_dof_ids(f.space)

FESpaces.get_cell_is_dirichlet(f::DirectSumFESpace) = get_cell_is_dirichlet(f.space)

FESpaces.num_dirichlet_dofs(f::DirectSumFESpace) = num_dirichlet_dofs(f.space)

FESpaces.num_dirichlet_tags(f::DirectSumFESpace) = num_dirichlet_tags(f.space)

FESpaces.get_dirichlet_dof_tag(f::DirectSumFESpace) = get_dirichlet_dof_tag(f.space)

FESpaces.get_vector_type(f::DirectSumFESpace) = get_vector_type(f.space)

FESpaces.get_dirichlet_dof_values(f::DirectSumFESpace) = get_dirichlet_dof_values(f.space)

function FESpaces.scatter_free_and_dirichlet_values(f::DirectSumFESpace,fv,dv)
  scatter_free_and_dirichlet_values(f.space,fv,dv)
end

function FESpaces.gather_free_and_dirichlet_values(f::DirectSumFESpace,cv)
  FESpaces.gather_free_and_dirichlet_values(f.space,cv)
end

function FESpaces.gather_free_and_dirichlet_values!(fv,dv,f::DirectSumFESpace,cv)
  gather_free_and_dirichlet_values!(fv,dv,f.space,cv)
end

function FESpaces.zero_free_values(f::DirectSumFESpace)
  zero_free_values(f.space)
end

function FESpaces.zero_dirichlet_values(f::DirectSumFESpace)
  zero_dirichlet_values(f.space)
end

function FESpaces.SparseMatrixAssembler(
  mat,
  vec,
  trial::DirectSumFESpace,
  test::DirectSumFESpace,
  strategy::AssemblyStrategy=FESpaces.DefaultAssemblyStrategy())

  ExtensionAssembler(mat,vec,trial,test,strategy)
end

function FESpaces.SparseMatrixAssembler(
  mat,
  vec,
  trial::DirectSumFESpace{<:SingleFieldParamFESpace,<:SingleFieldParamFESpace},
  test::DirectSumFESpace,
  strategy::AssemblyStrategy=FESpaces.DefaultAssemblyStrategy())

  trial′ = DirectSumFESpace(get_fe_space(trial.space.space),get_fe_space(trial.complementary.space))
  assem = SparseMatrixAssembler(mat,vec,trial′,test,strategy)
  parameterize(assem,param_length(trial.space.space))
end

# utils

get_emb_space(f::DirectSumFESpace) = get_emb_space(f.space)
get_act_space(f::DirectSumFESpace) = get_act_space(f.space)
get_bg_space(f::DirectSumFESpace) = get_bg_space(f.space)
get_bg_cell_dof_ids(f::DirectSumFESpace,args...) = get_bg_cell_dof_ids(f.space,args...)

for F in (:(DofMaps.get_dof_to_bg_dof),:(DofMaps.get_fdof_to_bg_fdof),:(DofMaps.get_ddof_to_bg_ddof),
          :(DofMaps.get_bg_dof_to_dof),:(DofMaps.get_bg_fdof_to_fdof),:(DofMaps.get_bg_ddof_to_ddof))
  @eval begin
    $F(f::DirectSumFESpace) = $F(f.space)
  end
end

get_space(f::DirectSumFESpace) = f.space
get_out_space(f::DirectSumFESpace) = f.complementary

for F in (:(ParamFESpaces.UnEvalTrialFESpace),:(ODEs.TransientTrialFESpace),:(FESpaces.TrialFESpace))
  @eval begin
    function $F(f::DirectSumFESpace,dirichlet::Union{Function,AbstractVector{<:Function}})
      DirectSumFESpace($F(f.space,dirichlet),$F(f.complementary,dirichlet))
    end
  end
end

"""
    (⊕)(uh::FEFunction,vh::FEFunction) -> FEFunction

Given a FEFunction `uh` defined on a FE space and a FEFunction `vh` defined on its
complementary (see [`complementary_space`](@ref) for more details), returns a
FEFunction defined as their direct sum. In practice, the output's values coincides
with those of `uh` on the DOFs associated with the space, and those of `vh` on the
DOFs associated with the complementary
"""
function (⊕)(uh::FEFunction,vh::FEFunction)
  space = get_fe_space(uh)
  complementary = get_fe_space(vh)
  @check _same_background_space(space,complementary)

  bg_fv = zero_bg_free_values(space)
  bg_dv = zero_bg_dirichlet_values(space)

  cfv = get_free_dof_values(vh)
  cdv = get_dirichlet_dof_values(vh)
  _bg_vals_from_vals!(bg_fv,bg_dv,complementary,cfv,cdv)

  fv = get_free_dof_values(uh)
  dv = get_dirichlet_dof_values(uh)
  _bg_vals_from_vals!(bg_fv,bg_dv,space,fv,dv)

  FEFunction(get_bg_space(space),bg_fv,bg_dv)
end

function _same_background_space(space::SingleFieldFESpace,complementary::SingleFieldFESpace)
  space==complementary
end

function _same_background_space(space::AbstractTrialFESpace,complementary::AbstractTrialFESpace)
  _same_background_space(get_fe_space(space),get_fe_space(complementary))
end

function _same_background_space(space::EmbeddedFESpace,complementary::EmbeddedFESpace)
  _same_background_space(get_bg_space(space),get_bg_space(complementary))
end

FESpaces.get_dirichlet_dof_values(uh::SingleFieldFEFunction) = uh.dirichlet_values
FESpaces.get_dirichlet_dof_values(uh::SingleFieldParamFEFunction) = uh.dirichlet_values

# dof map utils

function DofMaps.get_dof_map(f::DirectSumFESpace,args...)
  get_dof_map(get_bg_space(f),args...)
end

function DofMaps.get_dof_map_with_diri(f::DirectSumFESpace,args...)
  get_dof_map_with_diri(get_bg_space(f),args...)
end

function ParamSteady._assemble_matrix(f,V::DirectSumFESpace)
  ParamSteady._assemble_matrix(f,get_bg_space(V))
end

function ParamSteady._assemble_matrix(f,spaces::Vector{<:DirectSumFESpace})
  ParamSteady._assemble_matrix(f,get_bg_space.(spaces))
end

function DofMaps.get_sparsity(
  U::DirectSumFESpace{S,<:TProductFESpace},
  V::DirectSumFESpace{S,<:TProductFESpace},
  args...
  ) where S

  Ubg = get_bg_space(U)
  Vbg = get_bg_space(V)
  @check length(Ubg.spaces_1d) == length(Vbg.spaces_1d)
  sparsity = SparsityPattern(U,V,args...)
  sparsities_1d = map(1:length(Ubg.spaces_1d)) do d
    get_sparsity(Ubg.spaces_1d[d],Vbg.spaces_1d[d])
  end
  return TProductSparsity(sparsity,sparsities_1d)
end

function DofMaps.SparsityPattern(
  U::DirectSumFESpace,
  V::DirectSumFESpace,
  trian=DofMaps._get_common_domain(U,V)
  )

  a = ExtensionAssembler(U,V)
  m1 = nz_counter(FESpaces.get_matrix_builder(a),(FESpaces.get_rows(a),FESpaces.get_cols(a)))
  cellidsrows = get_bg_cell_dof_ids(V,trian)
  cellidscols = get_bg_cell_dof_ids(U,trian)
  DofMaps.trivial_symbolic_loop_matrix!(m1,cellidsrows,cellidscols)
  m2 = Algebra.nz_allocation(m1)
  DofMaps.trivial_symbolic_loop_matrix!(m2,cellidsrows,cellidscols)
  m3 = Algebra.create_from_nz(m2)
  SparsityPattern(m3)
end

# evaluations

const DirectSumTrialFESpace = DirectSumFESpace{<:AbstractTrialFESpace,<:AbstractTrialFESpace}

function Arrays.evaluate(f::DirectSumTrialFESpace,args...)
  space = evaluate(f.space,args...)
  complementary = evaluate(f.complementary,args...)
  DirectSumFESpace(space,complementary)
end

(f::DirectSumTrialFESpace)(μ) = evaluate(f,μ)
(f::DirectSumTrialFESpace)(μ,t) = evaluate(f,μ,t)

