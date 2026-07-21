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
const NoHRProjection{A<:Projection} = HRProjection{A,<:NoHyperReduction}

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
const AffineHRProjection{A<:Projection} = HRProjection{A,<:AffineHyperReduction}

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
const MDEIMProjection{A<:Projection} = HRProjection{A,<:MDEIMHyperReduction}

"""
"""
const SOPTProjection{A<:Projection} = HRProjection{A,<:SOPTHyperReduction}

"""
"""
const RBFProjection{A<:Projection} = HRProjection{A,<:RBFHyperReduction}

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
get_style(a::GenericHRProjection) = a.style
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
  red::NoHyperReduction,
  s::Snapshots,
  trian::Triangulation,
  test::RBSpace
  )

  T = get_dof_value_type(test)
  nrows = num_reduced_dofs(test)
  basis = ReducedProjection(zeros(T,nrows,1))
  interp = Interpolation(red,trian)
  return HRProjection(basis,red,interp)
end

function HRProjection(
  red::NoHyperReduction,
  s::Snapshots,
  trian::Triangulation,
  trial::RBSpace,
  test::RBSpace
  )

  T = get_dof_value_type(trial)
  nrows = num_reduced_dofs(test)
  ncols = num_reduced_dofs(trial)
  basis = ReducedProjection(zeros(T,nrows,1,ncols))
  interp = Interpolation(red,trian)
  return HRProjection(basis,red,interp)
end

function HRProjection(
  red::AffineHyperReduction,
  s::Snapshots,
  trian::Triangulation,
  test::RBSpace
  )

  basis = GalerkinProjectable(s)
  proj_basis = project(test,basis)
  interp = Interpolation(red)
  return HRProjection(proj_basis,red,interp)
end

function HRProjection(
  red::AffineHyperReduction,
  s::Snapshots,
  trian::Triangulation,
  trial::RBSpace,
  test::RBSpace
  )

  basis = GalerkinProjectable(s)
  proj_basis = project(test,basis,trial)
  interp = Interpolation(red)
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

const NNHRProjection{A<:Projection,B<:AbstractNNHyperReduction} = HRProjection{A,B}

function FESpaces.interpolate!(
  b̂::AbstractArray,
  cache,
  a::NNHRProjection{<:Projection,<:NNHyperReduction},
  r::AbstractRealisation
  )

  o = one(eltype2(b̂))
  x = matrix_of_params(r)
  i = get_interpolation(a)
  coeff = evaluate!(cache,i.interpolation,x)
  mul!(b̂,a,coeff,o,o)
  return b̂
end

struct NNOperator{A,B} <: NNHRProjection{B,NNOperatorReduction}
  model::A
  bias::B
end

function NNOperator(model::NeuralNetwork,test::RBSpace) 
  T = get_dof_value_type(test)
  nrows = num_reduced_dofs(test)
  basis = ReducedProjection(zeros(T,nrows,1))
  NNOperator(model,basis)
end

function NNOperator(model::NeuralNetwork,trial::RBSpace,test::RBSpace) 
  T = get_dof_value_type(trial)
  nrows = num_reduced_dofs(test)
  ncols = num_reduced_dofs(trial)
  basis = ReducedProjection(zeros(T,nrows,1,ncols))
  NNOperator(model,basis)
end

get_basis(a::NNOperator) = a.bias
get_interpolation(a::NNOperator) = EmptyInterpolation()
projection_eltype(a::NNOperator) = eltype(get_weights(a.model))

num_reduced_dofs(a::NNOperator) = 1
num_reduced_dofs_left_projector(a::NNOperator) = first(a.reduced_sizes)
num_reduced_dofs_right_projector(a::NNOperator) = last(a.reduced_sizes)

function FESpaces.interpolate!(
  b̂::AbstractArray,
  cache,
  a::NNOperator,
  r::AbstractRealisation
  )

  b̂r = evaluate!(cache,a.model,matrix_of_params(r))
  o = one(eltype2(b̂))
  _axpy!(o,b̂r,b̂)
  return b̂
end

function HRProjection(
  red::NNOperatorReduction,
  s::Snapshots,
  trian::Triangulation,
  test::RBSpace
  )

  r = get_realisation(s)
  b = GalerkinProjectable(s)
  y = galerkin_projection(test,b)
  ϕ = get_basis(y)
  model = TrainedNeuralNetwork(get_strategy(red),r,ϕ)
  return NNOperator(model,test)
end

function HRProjection(
  red::NNOperatorReduction,
  s::Snapshots,
  trian::Triangulation,
  trial::RBSpace,
  test::RBSpace
  )

  r = get_realisation(s)
  A = GalerkinProjectable(s)
  y = galerkin_projection(test,A,trial)
  ϕ = permutedims(get_basis(y),(1,3,2))
  model = TrainedNeuralNetwork(get_strategy(red),r,ϕ)
  return NNOperator(model,trial,test)
end

function HRProjection(
  red::NNHyperReduction,
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
  red::NNHyperReduction,
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

function allocate_coefficient(a::NNOperator,r::AbstractRealisation)
  x = matrix_of_params(r)
  return_cache(a.model,x)
end

function allocate_coefficient(a::NNHRProjection{<:Projection,<:NNHyperReduction},r::AbstractRealisation)
  x = matrix_of_params(r)
  i = get_interpolation(a)
  return_cache(i.interpolation,x)
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
const MDEIMContribution = AffineContribution{<:MDEIMProjection}

"""
"""
const SOPTContribution = AffineContribution{<:SOPTProjection}

"""
"""
const RBFContribution = AffineContribution{<:RBFProjection}

"""
"""
const NNContribution = AffineContribution{<:NNHRProjection}

function allocate_coefficient(a::AffineContribution,args...)
  contribution(get_domains(a)) do trian
    allocate_coefficient(a[trian],args...)
  end
end

function allocate_coefficient(a::NNContribution,args...)
  allocate_coefficient(first(get_contributions(a)),args...)
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

function allocate_hypred_cache(a::NNContribution,args...)
  fecache = allocate_coefficient(a,args...)
  coeffs = fecache
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

function FESpaces.interpolate!(
  hypred::AbstractArray,
  cache,
  a::NNContribution,
  r::AbstractRealisation
  )

  fill!(hypred,zero(eltype(hypred)))
  for aval in get_contributions(a)
    interpolate!(hypred,cache,aval,r)
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
      touched::Array{Bool,N}
    end

Block container for HRProjection in a `MultiField` setting. This
type is conceived similarly to `ArrayBlock` in [`Gridap`](@ref)
"""
struct BlockHRProjection{N,A,B} <: HRProjection{BlockProjection{A,N},B}
  array::Array{<:HRProjection{A,B},N}
  touched::Array{Bool,N}

  function BlockHRProjection(
    array::Array{<:HRProjection{A,B},N},
    touched::Array{Bool,N}
    ) where {A,B,N}

    @check size(array) == size(touched)
    new{N,A,B}(array,touched)
  end
end

Base.ndims(a::BlockHRProjection) = ndims(a.touched)
Base.size(a::BlockHRProjection,args...) = size(a.touched,args...)
Base.axes(a::BlockHRProjection,args...) = axes(a.touched,args...)
Base.length(a::BlockHRProjection) = length(a.touched)
Base.eachindex(a::BlockHRProjection) = eachindex(a.touched)

function Base.getindex(a::BlockHRProjection,i...)
  if !a.touched[i...]
    return nothing
  end
  a.array[i...]
end

function Base.setindex!(a::BlockHRProjection,v,i...)
  @check a.touched[i...] "Only touched entries can be set"
  a.array[i...] = v
end

Base.getindex(a::BlockHRProjection,i::Block) = getindex(a,i.n...)
Base.setindex!(a::BlockHRProjection,v,i::Block) = setindex!(a,v,i.n...)

function Arrays.testitem(a::BlockHRProjection)
  i = findfirst(a.touched)
  @notimplementedif isnothing(i)
  a.array[i]
end

function get_basis(a::BlockHRProjection{N}) where N
  A = eltype(a.array)
  block_cache = Array{A,N}(undef,size(a))
  for i in eachindex(a)
    if a.touched[i]
      block_cache[i] = get_basis(a[i])
    end
  end
  return ArrayBlock(block_cache,a.touched)
end

function get_interpolation(a::BlockHRProjection)
  array = Array{Interpolation,ndims(a)}(undef,size(a))
  for i in eachindex(a)
    if a.touched[i]
      array[i] = get_interpolation(a.array[i])
    end
  end
  return BlockInterpolation(array,a.touched)
end

get_style(a::BlockHRProjection) = get_style(first(a.array))
projection_eltype(a::BlockHRProjection) = promote_type(map(projection_eltype,a.array)...)

function FESpaces.interpolate!(
  hypred::Union{BlockParamArray,BlockArray},
  coeff::ArrayBlock,
  a::BlockHRProjection,
  b::ArrayBlock
  )

  for i in eachindex(a)
    if a.touched[i]
      interpolate!(blocks(hypred)[i],coeff.array[i],a.array[i],b.array[i])
    end
  end
  return hypred
end

function FESpaces.interpolate!(
  hypred::Union{BlockParamArray,BlockArray},
  coeff::ArrayBlock,
  a::BlockHRProjection,
  b::Union{BlockParamArray,BlockArray}
  )

  for i in eachindex(a)
    if a.touched[i]
      interpolate!(blocks(hypred)[i],coeff.array[i],a.array[i],blocks(b)[i])
    end
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
    if a.touched[i]
      interpolate!(blocks(hypred)[i],coeff.array[i],a.array[i],r)
    end
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

function allocate_coefficient(a::BlockHRProjection{N}) where N
  A = typeof(allocate_coefficient(first(a.array)))
  block_cache = Array{A,N}(undef,size(a))
  for i in eachindex(a)
    if a.touched[i]
      block_cache[i] = allocate_coefficient(a[i])
    end
  end
  return ArrayBlock(block_cache,a.touched)
end

function allocate_hyper_reduction(a::BlockHRProjection{N}) where N
  A = typeof(allocate_hyper_reduction(first(a.array)))
  block_cache = Array{A,N}(undef,size(a))
  for i in eachindex(a)
    block_cache[i] = allocate_hyper_reduction(a.array[i])
  end
  return mortar(block_cache)
end

function allocate_coefficient(
  a::BlockHRProjection{N,<:Any,<:AbstractNNHyperReduction},
  r::AbstractRealisation
  ) where N

  i0 = findfirst(a.touched)
  A = typeof(allocate_coefficient(a.array[i0],r))
  block_cache = Array{A,N}(undef,size(a))
  for i in eachindex(a)
    if a.touched[i]
      block_cache[i] = allocate_coefficient(a.array[i],r)
    end
  end
  return ArrayBlock(block_cache,a.touched)
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

  hyper_red = BlockHRProjection(hyper_reds,s.touched)
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

  hyper_red = BlockHRProjection(hyper_reds,s.touched)
  red_trian = reduced_triangulation(trian,hyper_red)

  return hyper_red,red_trian
end


# utils 

_axpy!(α,a,b) = @abstractmethod 

function _axpy!(α,a::AbstractMatrix,b::AbstractParamVector) 
  axpy!(α,a,get_all_data(b))
end

function _axpy!(α,a::AbstractMatrix,b::AbstractParamMatrix)
  nrows,ncols = innersize(b)
  k = param_length(b)
  a′ = reshape(a,nrows,ncols,k)
  axpy!(α,a′,get_all_data(b))
end

