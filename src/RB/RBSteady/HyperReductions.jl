"""
    abstract type HRProjection{A<:HyperReduction,B<:Projection} <: Projection end

Subtype of a [`Projection`](@ref) dedicated to the output of a hyper-reduction
procedure applied on residual/jacobians of a differential problem. This procedure
can be summarized in the following steps:

1. compute a snapshots tensor `T`
2. construct a `Projection` `Φ` by running the function `reduction` on `T`
3. implement an interpolation strategy

There are two types of interpolation strategies:

1. Empirical interpolation of the `Projection` `Φ`
2. Radial basis interpolation over the parameter space

We recall that a RB method requires the (Petrov-)Galerkin projection of the
operators (residuals/Jacobians) on the reduced subspace spanned by `Φ`:

- for residuals: `Φrb = test_basisᵀ Φ`
- for Jacobians: `Φrb = test_basisᵀ Φ trial_basis`

The output of this operation is a `ReducedProjection`. Therefore, a `HRProjection`
is completely characterized by the couple (`Φrb`,`i`), where `i` indicates the
chosen interpolation strategy.

Subtypes:
- [`GenericHRProjection`](@ref)
- [`BlockHRProjection`](@ref)
"""
abstract type HRProjection{A<:HyperReduction,B<:Projection} <: Projection end

const HRVecProjection{A<:HyperReduction} = HRProjection{A,<:ReducedVecProjection}
const HRMatProjection{A<:HyperReduction} = HRProjection{A,<:ReducedMatProjection}

HRProjection(::Reduction,args...) = @abstractmethod

"""
    get_interpolation(a::HRProjection) -> Interpolation

For a [`HRProjection`](@ref) `a` represented by the couple `(Φrb,i)`, returns `i`
"""
get_interpolation(a::HRProjection) = @abstractmethod

num_reduced_dofs(a::HRProjection) = num_reduced_dofs(get_basis(a))
num_reduced_dofs_left_projector(a::HRProjection) = num_reduced_dofs_left_projector(get_basis(a))
num_reduced_dofs_right_projector(a::HRProjection) = num_reduced_dofs_right_projector(get_basis(a))

function FESpaces.interpolate!(
  b̂::AbstractArray,
  coeff::AbstractArray,
  a::HRProjection,
  x::Any
  )

  o = one(eltype2(b̂))
  interpolate!(coeff,get_interpolation(a),x)
  mul!(b̂,a,coeff,o,o)
  return b̂
end

"""
"""
const NoHRProjection{A<:Projection} = HRProjection{NoHyperReduction,A}

function FESpaces.interpolate!(
  b̂::AbstractArray,
  coeff::AbstractArray,
  a::NoHRProjection,
  x::AbstractArray
  )

  o = one(eltype2(b̂))
  axpy!(o,coeff,b̂)
  return b̂
end

"""
"""
const AffineHRProjection{A<:Projection} = HRProjection{AffineHyperReduction,A}

function FESpaces.interpolate!(
  b̂::AbstractArray,
  coeff::AbstractArray,
  a::AffineHRProjection,
  x::Any
  )

  o = one(eltype2(b̂))
  L = param_length(b̂)
  ϕ = get_basis(get_basis(a))
  axpy!(o,parameterise(ϕ,L),b̂)
  return b̂
end

"""
"""
const DEIMProjection{A<:Projection} = HRProjection{<:DEIMHyperReduction,A}

"""
"""
const SOPTProjection{A<:Projection} = HRProjection{<:SOPTHyperReduction,A}

"""
"""
const RBFProjection{A<:Projection} = HRProjection{<:RBFHyperReduction,A}

"""
    struct GenericHRProjection{A<:HyperReduction,B<:Projection,C<:Interpolation} <: HRProjection{A,B}
      style::A
      basis::B
      interpolation::C
    end

Generic implementation of an [`HRProjection`](@ref) object
"""
struct GenericHRProjection{A<:HyperReduction,B<:Projection,C<:Interpolation} <: HRProjection{A,B}
  style::A
  basis::B
  interpolation::C
end

function HRProjection(basis::ReducedProjection,style::HyperReduction,interp::Interpolation)
  GenericHRProjection(style,basis,interp)
end

get_basis(a::GenericHRProjection) = a.basis
get_style(a::GenericHRProjection) = a.style
get_interpolation(a::GenericHRProjection) = a.interpolation

# empty cases

function EmptyHRProjection(red::Reduction,s,trian,test)
  T = get_dof_value_type(test)
  nrows = num_reduced_dofs(test)
  basis = ReducedProjection(zeros(T,nrows,1))
  interp = Interpolation(red)
  return HRProjection(basis,red,interp)
end

function EmptyHRProjection(red::Reduction,s,trian,trial,test)
  T = get_dof_value_type(trial)
  nrows = num_reduced_dofs(test)
  ncols = num_reduced_dofs(trial)
  basis = ReducedProjection(zeros(T,nrows,ncols,1))
  interp = Interpolation(red)
  return HRProjection(basis,red,interp)
end

function HRProjection(red::Reduction,s,trian,test)
  basis = projection(get_reduction(red),s)
  proj_basis = project(test,basis)
  interp = Interpolation(red,basis,trian,test)
  return HRProjection(proj_basis,red,interp)
end

function HRProjection(red::Reduction,s,trian,trial,test)
  basis = projection(get_reduction(red),s)
  proj_basis = project(test,basis,trial)
  interp = Interpolation(red,basis,trian,trial,test)
  return HRProjection(proj_basis,red,interp)
end

function HRProjection(red::NoHyperReduction,s,trian,test)
  T = get_dof_value_type(test)
  nrows = num_reduced_dofs(test)
  basis = ReducedProjection(zeros(T,nrows,1))
  interp = Interpolation(red,trian)
  return HRProjection(basis,red,interp)
end

function HRProjection(red::NoHyperReduction,s,trian,trial,test)
  T = get_dof_value_type(trial)
  nrows = num_reduced_dofs(test)
  ncols = num_reduced_dofs(trial)
  basis = ReducedProjection(zeros(T,nrows,ncols,1))
  interp = Interpolation(red,trian)
  return HRProjection(basis,red,interp)
end

function HRProjection(red::AffineHyperReduction,s,trian,test)
  basis = GalerkinProjectable(s)
  proj_basis = project(test,basis)
  interp = Interpolation(red)
  return HRProjection(proj_basis,red,interp)
end

function HRProjection(red::AffineHyperReduction,s,trian,trial,test)
  basis = GalerkinProjectable(s)
  proj_basis = project(test,basis,trial)
  interp = Interpolation(red)
  return HRProjection(proj_basis,red,interp)
end

function HRProjection(red::RBFHyperReduction,s,trian,test)
  basis = projection(get_reduction(red),s)
  proj_basis = project(test,basis)
  interp = Interpolation(red,basis,s)
  return HRProjection(proj_basis,red,interp)
end

function HRProjection(red::RBFHyperReduction,s,trian,trial,test)
  basis = projection(get_reduction(red),s)
  proj_basis = project(test,basis,trial)
  interp = Interpolation(red,basis,s)
  return HRProjection(proj_basis,red,interp)
end

function allocate_coefficient(a::HRProjection)
  T = projection_eltype(a)
  n = num_reduced_dofs(a)
  coeff = zeros(T,n)
  return coeff
end

function allocate_coefficient(a::NoHRProjection)
  allocate_hyper_reduction(a)
end

function allocate_hyper_reduction(a::HRVecProjection)
  T = projection_eltype(a)
  nrows = num_reduced_dofs_left_projector(a)
  hypred = zeros(T,nrows)
  return hypred
end

function allocate_hyper_reduction(a::HRMatProjection)
  T = projection_eltype(a)
  nrows = num_reduced_dofs_left_projector(a)
  ncols = num_reduced_dofs_right_projector(a)
  hypred = zeros(T,nrows,ncols)
  return hypred
end

for f in (:allocate_coefficient,:allocate_hyper_reduction)
  @eval $f(a::HRProjection,r::AbstractRealisation) = parameterise($f(a),num_params(r))
end

"""
    const AffineContribution{V<:HRProjection} = Contribution{V}

[`Contribution`](@ref) whose field `values` are Projections
"""
const AffineContribution{V<:HRProjection} = Contribution{V}

"""
"""
const NoHRContribution = AffineContribution{<:NoHRProjection}

"""
"""
const AffineHRContribution = AffineContribution{<:AffineHRProjection}

"""
"""
const DEIMContribution = AffineContribution{<:DEIMProjection}

"""
"""
const SOPTContribution = AffineContribution{<:SOPTProjection}

"""
"""
const RBFContribution = AffineContribution{<:RBFProjection}

function allocate_coefficient(a::AffineContribution,args...)
  contribution(get_domains(a)) do trian
    allocate_coefficient(a[trian],args...)
  end
end

function allocate_hyper_reduction(a::AffineContribution,args...)
  allocate_hyper_reduction(first(get_contributions(a)),args...)
end

function allocate_hypred_cache(a,args...)
  fecache = allocate_coefficient(a,args...)
  coeffs = allocate_coefficient(a,args...)
  hypred = allocate_hyper_reduction(a,args...)
  return HRParamArray(fecache,coeffs,hypred)
end

function FESpaces.interpolate!(
  hypred::AbstractArray,
  coeff::ArrayContribution,
  a::AffineContribution,
  b::ArrayContribution
  )

  @check length(coeff) == length(a) == length(b)
  fill!(hypred,zero(eltype(hypred)))
  for (aval,bval,cval) in zip(get_contributions(a),get_contributions(b),get_contributions(coeff))
    interpolate!(hypred,cval,aval,bval)
  end
  return hypred
end

function FESpaces.interpolate!(
  hypred::AbstractArray,
  coeff::ArrayContribution,
  a::AffineContribution,
  r::AbstractRealisation
  )

  @check length(coeff) == length(a)
  fill!(hypred,zero(eltype(hypred)))
  for (aval,cval) in zip(get_contributions(a),get_contributions(coeff))
    interpolate!(hypred,cval,aval,r)
  end
  return hypred
end

function reduced_form(red::Reduction,s,trian,args...)
  isempty(s) && return EmptyHRProjection(red,s,trian,args...)
  HRProjection(red,s,trian,args...)
end

function reduced_forms!(cache,red,op::PlainParamOperator,trians,red_test)
  contribution(trians) do trian
    res = _residual_for_trian!(cache,op,trian)
    reduced_form(red,res,trian,red_test)
  end
end

function reduced_forms!(cache,red,op::PlainParamOperator,trians,red_trial,red_test)
  contribution(trians) do trian
    jac = _jacobian_for_trian!(cache,op,trian)
    reduced_form(red,jac,trian,red_trial,red_test)
  end
end

function reduced_forms!(cache,red,op::BundleParamOperator,args...)
  cache_lin,cache_nlin = cache
  op_lin = get_linear_part(op)
  op_nlin = get_nonlinear_part(op)
  red_form_lin = reduced_forms!(cache_lin,red,op_lin,args...)
  red_form_nlin = reduced_forms!(cache_nlin,red,op_nlin,args...)
  return Bundle(red_form_lin,red_form_nlin)
end

"""
    reduced_residual(
      solver::RBSolver,
      op::ParamOperator,
      red_test::RBSpace,
      s::AbstractSnapshots
      ) -> AffineContribution

Reduces the residual contained in `op` via hyper-reduction. This function
first builds the residual snapshots, which are then reduced according to the strategy
`residual_reduction` specified in the reduced solver `solver`
"""
function reduced_residual(
  solver::RBSolver,
  op::ParamOperator,
  red_test::RBSpace,
  s::AbstractSnapshots
  )

  res_red = get_residual_reduction(solver)
  trians = get_domains_res(op)
  t = @timed begin
    resdata = _prepare_residual(solver,op,s)
    red_res = reduced_forms!(resdata,res_red,op,trians,red_test)
  end
  set_residual_tracker!(solver,t)
  return red_res
end

"""
    reduced_jacobian(
      solver::RBSolver,
      op::ParamOperator,
      red_trial::RBSpace,
      red_test::RBSpace,
      s::AbstractSnapshots
      ) -> Union{AffineContribution,AffineContributionTuple}

Reduces the Jacobian contained in `op` via hyper-reduction. This function
first builds the Jacobian snapshots, which are then reduced according to the strategy
`reduced_jacobian` specified in the reduced solver `solver`. In transient applications,
the output is a tuple of length equal to the number of Jacobians(i.e., equal to
the order of the ODE plus one)
"""
function reduced_jacobian(
  solver::RBSolver,
  op::ParamOperator,
  red_trial::RBSpace,
  red_test::RBSpace,
  s::AbstractSnapshots
  )

  jac_red = get_jacobian_reduction(solver)
  trians = get_domains_jac(op)
  t = @timed begin
    jacdata = _prepare_jacobian(solver,op,s)
    red_jac = reduced_forms!(jacdata,jac_red,op,trians,red_trial,red_test)
  end
  set_jacobian_tracker!(solver,t)
  return red_jac
end

"""
    reduced_weak_form(
      solver::RBSolver,
      op::ParamOperator,
      red_trial::RBSpace,
      red_test::RBSpace,
      s::AbstractSnapshots
      ) -> (AffineContribution,Union{AffineContribution,AffineContributionTuple})

Reduces the residual/Jacobian contained in `op` via hyper-reduction. Check the
functions [`reduced_residual`](@ref) and [`reduced_jacobian`](@ref) for more details
"""
function reduced_weak_form(
  solver::RBSolver,
  op::ParamOperator,
  red_trial::RBSpace,
  red_test::RBSpace,
  s::AbstractSnapshots
  )

  red_jac = reduced_jacobian(solver,op,red_trial,red_test,s)
  red_res = reduced_residual(solver,op,red_test,s)
  return red_jac,red_res
end

# multi field interface

"""
    struct BlockHRProjection{A<:HyperReduction,B<:Projection,T<:HRProjection{A,B},N} <: HRProjection{A,BlockProjection{T,N}}
      array::Array{T,N}
    end

Block container for HRProjection in a `MultiField` setting.
"""
struct BlockHRProjection{A<:HyperReduction,B<:Projection,T<:HRProjection{A,B},N} <: HRProjection{A,BlockProjection{T,N}}
  array::Array{T,N}
end

Base.ndims(a::BlockHRProjection) = ndims(a.array)
Base.size(a::BlockHRProjection,args...) = size(a.array,args...)
Base.axes(a::BlockHRProjection,args...) = axes(a.array,args...)
Base.length(a::BlockHRProjection) = length(a.array)
Base.eachindex(a::BlockHRProjection) = eachindex(a.array)

Base.getindex(a::BlockHRProjection,i...) = a.array[i...]
Base.setindex!(a::BlockHRProjection,v,i...) = (a.array[i...] = v)

Base.getindex(a::BlockHRProjection,i::Block) = getindex(a,i.n...)
Base.setindex!(a::BlockHRProjection,v,i::Block) = setindex!(a,v,i.n...)

Arrays.testitem(a::BlockHRProjection) = first(a.array)

function get_basis(a::BlockHRProjection)
  array = map(get_basis,a.array)
  return BlockProjection(array)
end

function get_interpolation(a::BlockHRProjection)
  array = map(get_interpolation,a.array)
  return BlockInterpolation(array)
end

get_style(a::BlockHRProjection) = get_style(first(a.array))
projection_eltype(a::BlockHRProjection) = promote_type(map(projection_eltype,a.array)...)

function FESpaces.interpolate!(
  hypred::BlockParamArray,
  coeff::AbstractArray{<:AbstractArray},
  a::BlockHRProjection,
  b::AbstractArray{<:AbstractArray}
  )

  for i in eachindex(a)
    interpolate!(blocks(hypred)[i],coeff[i],a.array[i],b[i])
  end
  return hypred
end

function FESpaces.interpolate!(
  hypred::BlockParamArray,
  coeff::AbstractArray{<:AbstractArray},
  a::BlockHRProjection,
  b::BlockParamArray
  )

  for i in eachindex(a)
    interpolate!(blocks(hypred)[i],coeff[i],a.array[i],blocks(b)[i])
  end
  return hypred
end

function FESpaces.interpolate!(
  hypred::BlockParamArray,
  coeff::AbstractArray{<:AbstractArray},
  a::BlockHRProjection,
  r::AbstractRealisation
  )

  for i in eachindex(a)
    interpolate!(blocks(hypred)[i],coeff[i],a.array[i],r)
  end
  return hypred
end

for T in (:AffineContribution,:BlockHRProjection)
  @eval begin
    function FESpaces.interpolate!(cache::HRParamArray,a::$T)
      interpolate!(cache.hypred,cache.coeff,a,cache.fecache)
    end

    function FESpaces.interpolate!(cache::HRParamArray,a::$T,r::AbstractRealisation)
      interpolate!(cache.hypred,cache.coeff,a,r)
    end
  end
end

function allocate_coefficient(a::BlockHRProjection)
  map(allocate_coefficient,a.array)
end

function allocate_coefficient(a::BlockHRProjection,r::AbstractRealisation)
  map(b -> allocate_coefficient(b,r),a.array)
end

function allocate_hyper_reduction(a::BlockHRProjection)
  map(allocate_hyper_reduction,a.array) |> mortar
end

function reduced_form(red::Reduction,s::BlockSnapshots,trian,test)
  @check length(s) == length(test)
  hyper_reds = map(eachindex(s)) do i
    reduced_form(red,s[i],trian,test[i])
  end
  BlockHRProjection(hyper_reds)
end

function reduced_form(red::Reduction,s::BlockSnapshots,trian,trial,test)
  @check size(s,1) == length(test)
  @check size(s,2) == length(trial)
  hyper_reds = map(Iterators.product(axes(s)...)) do (i,j)
    reduced_form(red,s[i,j],trian,trial[j],test[i])
  end
  BlockHRProjection(hyper_reds)
end

# utils

function _prepare_residual(
  solver::RBSolver,
  op::ParamOperator,
  s::AbstractSnapshots
  )

  sres = select_snapshots(s,res_params(solver))
  us_res = get_param_data(sres)
  r_res = get_realisation(sres)
  islinear(op) && fill!(us_res,zero(eltype2(us_res)))

  paramcache = allocate_paramcache(op,r_res)
  uh = EvaluationFunction(paramcache.trial,us_res)
  test = get_test(op)
  v = get_fe_basis(test)
  assem = get_param_assembler(op,r)

  dc = ParamSteady.get_eval_res(op)(r,uh,v)
  vecdata = collect_cell_vector(test,dc)
  b = allocate_vector(assem,vecdata)

  return b,dc,assem,r_res
end

function _prepare_residual(
  solver::RBSolver,
  op::BundleParamOperator,
  s::AbstractSnapshots
  )

  sres = select_snapshots(s,res_params(solver))
  us_res = get_param_data(sres)
  r_res = get_realisation(sres)
  islinear(op) && fill!(us_res,zero(eltype2(us_res)))

  paramcache = allocate_paramcache(op,r_res)
  uh = EvaluationFunction(paramcache.trial,us_res)
  test = get_test(op)
  v = get_fe_basis(test)
  assem = get_param_assembler(op,r)

  dc_lin = get_linear_part(get_res(op))(r,uh,v)
  dc_nlin = get_nonlinear_part(get_res(op))(r,uh,v)
  vecdata = collect_cell_vector(test,dc_nlin)
  b = allocate_vector(assem,vecdata)

  data_lin = b,dc_lin,assem,r_res
  data_nlin = b,dc_nlin,assem,r_res

  return data_lin,data_nlin
end

function _prepare_jacobian(
  solver::RBSolver,
  op::ParamOperator,
  s::AbstractSnapshots
  )

  sjac = select_snapshots(s,jac_params(solver))
  us_jac = get_param_data(sjac)
  r_jac = get_realisation(sjac)

  paramcache = allocate_paramcache(op,r_jac)
  uh = EvaluationFunction(paramcache.trial,us_jac)
  trial = get_trial(op)
  du = get_trial_fe_basis(trial)
  test = get_test(op)
  v = get_fe_basis(test)
  assem = get_param_assembler(op,r)

  dc = get_jac(op)(r,uh,du,v)
  matdata = collect_cell_matrix(trial,test,dc)
  A = allocate_matrix(assem,matdata)

  return A,dc,assem,r_jac
end

function _prepare_jacobian(
  solver::RBSolver,
  op::ParamOperator,
  s::AbstractSnapshots
  )

  sjac = select_snapshots(s,jac_params(solver))
  us_jac = get_param_data(sjac)
  r_jac = get_realisation(sjac)

  paramcache = allocate_paramcache(op,r_jac)
  uh = EvaluationFunction(paramcache.trial,us_jac)
  trial = get_trial(op)
  du = get_trial_fe_basis(trial)
  test = get_test(op)
  v = get_fe_basis(test)
  assem = get_param_assembler(op,r)

  dc_lin = get_linear_part(get_jac(op))(r,uh,du,v)
  dc_nlin = get_nonlinear_part(get_jac(op))(r,uh,du,v)
  matdata = collect_cell_matrix(trial,test,dc_nlin)
  A = allocate_matrix(assem,matdata)

  data_lin = A,dc_lin,assem,r_jac
  data_nlin = A,dc_nlin,assem,r_jac

  return data_lin,data_nlin
end

function _residual_for_trian!(resdata,op,trian)
  b,dc,assem,r = resdata
  test = get_test(op)
  vecdata = collect_cell_vector_for_trian(test,dc,trian)
  assemble_vector!(b,assem,vecdata)
  bmap = get_dof_map(op,b)
  Snapshots(b,bmap,r)
end

function _jacobian_for_trian!(jacdata,op,trian)
  A,dc,assem,r = jacdata
  trial = get_trial(op)
  test = get_test(op)
  matdata = collect_cell_matrix_for_trian(trial,test,dc,trian)
  assemble_matrix!(A,assem,matdata)
  Amap = get_dof_map(op,A)
  Snapshots(A,Amap,r)
end

islinear(::ParamOperator) = false
islinear(::ParamOperator{LinearParamEq}) = true