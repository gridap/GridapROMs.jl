"""
    abstract type HRProjection{A<:Projection,B<:HyperReduction} <: Projection end

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
abstract type HRProjection{A<:Projection,B<:HyperReduction} <: Projection end

const HRVecProjection{B<:HyperReduction} = HRProjection{<:ReducedVecProjection,B}
const HRMatProjection{B<:HyperReduction} = HRProjection{<:ReducedMatProjection,B}

HRProjection(::Reduction,args...) = @abstractmethod

"""
    get_interpolation(a::HRProjection) -> Interpolation

For a [`HRProjection`](@ref) `a` represented by the couple `(Φrb,i)`, returns `i`
"""
get_interpolation(a::HRProjection) = @abstractmethod

get_integration_domain(a::HRProjection) = get_integration_domain(get_interpolation(a))

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
    reduced_triangulation(trian::Triangulation,a::HRProjection)

Returns the triangulation view of `trian` on the integration cells contained in `a`
"""
function reduced_triangulation(trian::Triangulation,a::HRProjection)
  reduced_triangulation(trian,get_interpolation(a))
end

function move_interpolation(a::HRProjection,args...)
  move_interpolation(get_interpolation(a),args...)
end

"""
"""
const MDEIMProjection{A<:ReducedProjection} = HRProjection{A,<:MDEIMHyperReduction}

"""
"""
const SOPTProjection{A<:ReducedProjection} = HRProjection{A,<:SOPTHyperReduction}

"""
"""
const RBFProjection{A<:ReducedProjection} = HRProjection{A,<:RBFHyperReduction}

"""
    struct GenericHRProjection{A,B} <: HRProjection{A,B}
      basis::A
      style::B
      interpolation::Interpolation
    end

Generic implementation of an [`HRProjection`](@ref) object
"""
struct GenericHRProjection{A,B} <: HRProjection{A,B}
  basis::A
  style::B
  interpolation::Interpolation
end

function HRProjection(basis::ReducedProjection,style::HyperReduction,interp::Interpolation)
  GenericHRProjection(basis,style,interp)
end

get_basis(a::GenericHRProjection) = a.basis
get_interpolation(a::GenericHRProjection) = a.interpolation
projection_eltype(a::GenericHRProjection) = projection_eltype(get_basis(a))

function HRProjection(
  red::Reduction,
  s::Nothing,
  trian::Triangulation,
  test::RBSpace
  )

  T = get_dof_value_type(test)
  nrows = num_reduced_dofs(test)
  basis = ReducedProjection(zeros(T,nrows,1))
  interp = Interpolation(red)
  return HRProjection(basis,red,interp)
end

function HRProjection(
  red::Reduction,
  s::Nothing,
  trian::Triangulation,
  trial::RBSpace,
  test::RBSpace
  )

  T = get_dof_value_type(trial)
  nrows = num_reduced_dofs(test)
  ncols = num_reduced_dofs(trial)
  basis = ReducedProjection(zeros(T,nrows,1,ncols))
  interp = Interpolation(red)
  return HRProjection(basis,red,interp)
end

function HRProjection(
  red::Reduction,
  s::Snapshots,
  trian::Triangulation,
  test::RBSpace
  )

  basis = projection(get_reduction(red),s)
  proj_basis = project(test,basis)
  interp = Interpolation(red,basis,trian,test)
  return HRProjection(proj_basis,red,interp)
end

function HRProjection(
  red::Reduction,
  s::Snapshots,
  trian::Triangulation,
  trial::RBSpace,
  test::RBSpace
  )

  basis = projection(get_reduction(red),s)
  proj_basis = project(test,basis,trial)
  interp = Interpolation(red,basis,trian,trial,test)
  return HRProjection(proj_basis,red,interp)
end

function HRProjection(
  red::RBFHyperReduction,
  s::Snapshots,
  trian::Triangulation,
  test::RBSpace
  )

  basis = projection(get_reduction(red),s)
  proj_basis = project(test,basis)
  interp = Interpolation(red,basis,s)
  return HRProjection(proj_basis,red,interp)
end

function HRProjection(
  red::RBFHyperReduction,
  s::Snapshots,
  trian::Triangulation,
  trial::RBSpace,
  test::RBSpace
  )

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

function allocate_hyper_reduction(a::HRVecProjection)
  T = projection_eltype(a)
  nrows = num_reduced_dofs_left_projector(a)
  hypred = zeros(T,nrows)
  fill!(hypred,zero(eltype(hypred)))
  return hypred
end

function allocate_hyper_reduction(a::HRMatProjection)
  T = projection_eltype(a)
  nrows = num_reduced_dofs_left_projector(a)
  ncols = num_reduced_dofs_right_projector(a)
  hypred = zeros(T,nrows,ncols)
  fill!(hypred,zero(eltype(hypred)))
  return hypred
end

for f in (:allocate_coefficient,:allocate_hyper_reduction)
  @eval $f(a::HRProjection,r::AbstractRealisation) = parameterise($f(a),num_params(r))
end

"""
    const AffineContribution{V<:Projection} = Contribution{V}

[`Contribution`](@ref) whose field `values` are Projections
"""
const AffineContribution{V<:Projection} = Contribution{V}

"""
"""
const MDEIMContribution = AffineContribution{<:MDEIMProjection}

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
  return hr_array(fecache,coeffs,hypred)
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

function reduced_form(red::Reduction,s,trian::Triangulation,args...)
  hyper_red = HRProjection(red,s,trian,args...)
  red_trian = reduced_triangulation(trian,hyper_red)
  return hyper_red,red_trian
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

  res = residual_snapshots(solver,op,s)
  res_red = get_residual_reduction(solver)
  t = @timed red_res = reduced_residual(res_red,red_test,res)
  println(CostTracker(t,name="Residual hyper-reduction"))
  return red_res
end

function reduced_residual(red::Reduction,test::RBSpace,c::ArrayContribution)
  a,trians = map(get_domains(c),get_contributions(c)) do trian,values
    reduced_form(red,values,trian,test)
  end |> tuple_of_arrays
  return Contribution(a,trians)
end

function get_background_trian(f::FESpace)
  model = get_background_model(get_triangulation(f))
  Triangulation(model)
end

function reduced_residual(red::Reduction,test::RBSpace,r::Snapshots)
  trian = get_background_trian(test)
  reduced_residual(red,test,Contribution(r,trian))
end

"""
    reduced_jacobian(
      solver::RBSolver,
      op::ParamOperator,
      red_trial::RBSpace,
      red_test::RBSpace,
      s::AbstractSnapshots
      ) -> Union{AffineContribution,TupOfAffineContribution}

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

  jac = jacobian_snapshots(solver,op,s)
  jac_red = get_jacobian_reduction(solver)
  t = @timed red_jac = reduced_jacobian(jac_red,red_trial,red_test,jac)
  println(CostTracker(t,name="Jacobian hyper-reduction"))
  return red_jac
end

function reduced_jacobian(red::Reduction,trial::RBSpace,test::RBSpace,c::ArrayContribution)
  a,trians = map(get_domains(c),get_contributions(c)) do trian,values
    reduced_form(red,values,trian,trial,test)
  end |> tuple_of_arrays
  return Contribution(a,trians)
end

function reduced_jacobian(red::Reduction,trial::RBSpace,test::RBSpace,j::Snapshots)
  trian = get_background_trian(test)
  reduced_jacobian(red,trial,test,Contribution(j,trian))
end

"""
    reduced_weak_form(
      solver::RBSolver,
      op::ParamOperator,
      red_trial::RBSpace,
      red_test::RBSpace,
      s::AbstractSnapshots
      ) -> (AffineContribution,Union{AffineContribution,TupOfAffineContribution})

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
    struct BlockHRProjection{N,A,B} <: HRProjection{BlockProjection{A,N},B}
      array::Array{<:HRProjection{A,B},N}
    end

Block container for HRProjection of type `H` in a `MultiField` setting.
"""
struct BlockHRProjection{N,A,B} <: HRProjection{BlockProjection{A,N},B}
  array::Array{<:HRProjection{A,B},N}
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
Arrays.testitem(a::BlockHRProjection) = a.array[1]

function get_basis(a::BlockHRProjection{N}) where N
  touched = fill(true,size(a))
  ArrayBlock(a.array,touched)
end

function get_interpolation(a::BlockHRProjection{N}) where N
  array = map(get_interpolation,a.array)
  touched = fill(true,size(a))
  ArrayBlock(array,touched)
end

function FESpaces.interpolate!(
  hypred::Union{BlockParamArray,BlockArray},
  coeff::ArrayBlock,
  a::BlockHRProjection,
  b::ArrayBlock
  )

  for i in eachindex(a)
    interpolate!(blocks(hypred)[i],coeff.array[i],a.array[i],b.array[i])
  end
  return hypred
end

function FESpaces.interpolate!(
  hypred::Union{BlockParamArray,BlockArray},
  coeff::ArrayBlock,
  a::BlockHRProjection,
  r::AbstractRealisation
  )

  for i in eachindex(a)
    interpolate!(blocks(hypred)[i],coeff.array[i],a.array[i],r)
  end
  return hypred
end

for T in (:AffineContribution,:BlockHRProjection)
  @eval begin
    function FESpaces.interpolate!(cache::AbstractHRArray,a::$T)
      interpolate!(cache.hypred,cache.coeff,a,cache.fecache)
    end

    function FESpaces.interpolate!(cache::AbstractHRArray,a::$T,r::AbstractRealisation)
      interpolate!(cache.hypred,cache.coeff,a,r)
    end
  end
end

for f in (:allocate_coefficient,:allocate_hyper_reduction)
  @eval begin
    function $f(a::BlockHRProjection)
      array = map($f,a.array)
      touched = fill(true,size(a))
      ArrayBlock(array,touched)
    end
  end
end

function reduced_form(
  red::Reduction,
  s::BlockSnapshots,
  trian::Triangulation,
  test::MultiFieldRBSpace
  )

  hyper_reds = map(eachindex(s)) do i
    hyper_red, = reduced_form(red,s[i],trian,test[i])
    hyper_red
  end

  hyper_red = BlockHRProjection(hyper_reds)
  red_trian = reduced_triangulation(trian,hyper_red)

  return hyper_red,red_trian
end

function reduced_form(
  red::Reduction,
  s::BlockSnapshots,
  trian::Triangulation,
  trial::MultiFieldRBSpace,
  test::MultiFieldRBSpace
  )

  hyper_reds = map(Iterators.product(axes(s)...)) do (i,j)
    hyper_red, = reduced_form(red,s[i,j],trian,trial[j],test[i])
    hyper_red
  end

  hyper_red = BlockHRProjection(hyper_reds)
  red_trian = reduced_triangulation(trian,hyper_red)

  return hyper_red,red_trian
end
