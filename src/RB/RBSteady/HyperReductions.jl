"""
    abstract type HyperReduction{
      A<:Reduction,
      B<:ReducedProjection,
      C<:IntegrationDomain
      } <: Projection end

Subtype of a [`Projection`](@ref) dedicated to the outputd of a hyper-reduction
(e.g. an empirical interpolation method (EIM)) procedure applied on residual/jacobians
of a differential problem. This procedure can be summarized in the following steps:

1. compute a snapshots tensor `T`
2. construct a `Projection` `Φ` by running the function `reduction` on `T`
3. find the EIM quantities `i`,`Φi` by running the function `empirical_interpolation`
  on `Φ`

The triplet (`Φ`, `Φi`,`i`) represents the minimum information needed to run the
online phase of the hyper-reduction. However, we recall that a RB method requires
the (Petrov-)Galerkin projection of residuals/Jacobianson a reduced subspace
built from solution snapshots, instead of providing the projection `Φ` we return
the reduced projection `Φrb`, where

- for residuals: `Φrb = test_basis' * Φ`
- for Jacobians: `Φrb = test_basis' * Φ * trial_basis`

The output of this operation is a ReducedProjection. Therefore, a HyperReduction
is completely characterized by the triplet `(Φrb,Φi,i)`.
Subtypes:
- [`TrivialHyperReduction`](@ref)
- [`MDEIM`](@ref)
"""
abstract type HyperReduction{A<:Reduction,B<:ReducedProjection,C<:IntegrationDomain} <: Projection end

HyperReduction(::Reduction,args...) = @abstractmethod

"""
    get_interpolation(a::HyperReduction) -> Factorization

For a [`HyperReduction`](@ref) `a` represented by the triplet `(Φrb,Φi,i)`,
returns `Φi`, usually stored as a Factorization
"""
get_interpolation(a::HyperReduction) = @abstractmethod

"""
    get_integration_domain(a::HyperReduction) -> IntegrationDomain

For a [`HyperReduction`](@ref) `a` represented by the triplet `(Φrb,Φi,i)`,
returns `i`
"""
get_integration_domain(a::HyperReduction) = @abstractmethod

get_integration_cells(a::HyperReduction) = get_integration_cells(get_integration_domain(a))
get_cellids_rows(a::HyperReduction) = get_cellids_rows(get_integration_domain(a))
get_cellids_cols(a::HyperReduction) = get_cellids_cols(get_integration_domain(a))
get_owned_icells(a::HyperReduction,cells) =

num_reduced_dofs(a::HyperReduction) = num_reduced_dofs(get_basis(a))
num_reduced_dofs_left_projector(a::HyperReduction) = num_reduced_dofs_left_projector(get_basis(a))
num_reduced_dofs_right_projector(a::HyperReduction) = num_reduced_dofs_right_projector(get_basis(a))

function inv_project!(
  b̂::AbstractParamArray,
  coeff::AbstractParamArray,
  a::HyperReduction,
  b::AbstractParamArray)

  o = one(eltype2(b̂))
  interp = get_interpolation(a)
  ldiv!(coeff,interp,b)
  mul!(b̂,a,coeff,o,o)
  return b̂
end

"""
    struct TrivialHyperReduction{A,B} <: HyperReduction{A,B,IntegrationDomain}
      reduction::A
      basis::B
    end

Trivial hyper-reduction returned whenever the residual/Jacobian is zero
"""
struct TrivialHyperReduction{A,B} <: HyperReduction{A,B,IntegrationDomain}
  reduction::A
  basis::B
end

get_basis(a::TrivialHyperReduction) = a.basis
get_interpolation(a::TrivialHyperReduction) = @notimplemented
get_integration_domain(a::TrivialHyperReduction) = @notimplemented

function HyperReduction(red::Reduction,s::Nothing,test::RBSpace)
  red = get_reduction(red)
  nrows = num_free_dofs(test)
  basis = ReducedProjection(zeros(nrows,1))
  return TrivialHyperReduction(red,basis)
end

function HyperReduction(red::Reduction,s::Nothing,trial::RBSpace,test::RBSpace)
  red = get_reduction(red)
  nrows = num_free_dofs(test)
  ncols = num_free_dofs(trial)
  basis = ReducedProjection(zeros(nrows,1,ncols))
  return TrivialHyperReduction(red,basis)
end

"""
    struct MDEIM{A,B,C} <: HyperReduction{A,B,C}
      reduction::A
      basis::B
      interpolation::Factorization
      domain::C
    end

[`HyperReduction`](@ref) returned by a matrix-based empirical interpolation method
"""
struct MDEIM{A,B,C} <: HyperReduction{A,B,C}
  reduction::A
  basis::B
  interpolation::Factorization
  domain::C
end

get_basis(a::MDEIM) = a.basis
get_interpolation(a::MDEIM) = a.interpolation
get_integration_domain(a::MDEIM) = a.domain

function HyperReduction(
  red::AbstractMDEIMReduction,
  s::Snapshots,
  trian::Triangulation,
  test::RBSpace)

  red = get_reduction(red)
  basis = projection(red,s)
  proj_basis = project(test,basis)
  indices,interp = empirical_interpolation(basis)
  factor = lu(interp)
  domain = vector_domain(trian,test,indices)
  return MDEIM(red,proj_basis,factor,domain)
end

function HyperReduction(
  red::AbstractMDEIMReduction,
  s::Snapshots,
  trian::Triangulation,
  trial::RBSpace,
  test::RBSpace)

  red = get_reduction(red)
  basis = projection(red,s)
  proj_basis = project(test,basis,trial)
  indices,interp = empirical_interpolation(basis)
  factor = lu(interp)
  domain = matrix_domain(trian,trial,test,indices...)
  return MDEIM(red,proj_basis,factor,domain)
end

function reduced_triangulation(trian::Triangulation,a::TrivialHyperReduction)
  red_trian = view(trian,[])
  return red_trian
end

"""
    reduced_triangulation(trian::Triangulation,a::HyperReduction)

Returns the triangulation view of `trian` on the integration cells contained in `a`
"""
function reduced_triangulation(trian::Triangulation,a::HyperReduction)
  i = get_integration_domain(a)
  cells = get_integration_cells(i)
  red_trian = view(trian,cells)
  return red_trian
end

function allocate_coefficient(a::HyperReduction,r::AbstractRealization)
  n = num_reduced_dofs(a)
  np = num_params(r)
  coeffvec = zeros(n)
  coeff = global_parameterize(coeffvec,np)
  return coeff
end

function allocate_hyper_reduction(
  a::HyperReduction{<:Reduction,<:ReducedVecProjection},
  r::AbstractRealization)

  nrows = num_reduced_dofs_left_projector(a)
  np = num_params(r)
  b = zeros(nrows)
  hypred = global_parameterize(b,np)
  fill!(hypred,zero(eltype(hypred)))
  return hypred
end

function allocate_hyper_reduction(
  a::HyperReduction{<:Reduction,<:ReducedMatProjection},
  r::AbstractRealization)

  nrows = num_reduced_dofs_left_projector(a)
  ncols = num_reduced_dofs_right_projector(a)
  np = num_params(r)
  M = zeros(nrows,ncols)
  hypred = global_parameterize(M,np)
  fill!(hypred,zero(eltype(hypred)))
  return hypred
end

function Utils.Contribution(v::Tuple{Vararg{HyperReduction}},t::Tuple{Vararg{Triangulation}})
  AffineContribution(v,t)
end

"""
    struct AffineContribution{A,V,K} <: Contribution

Contribution whose `values` assume one of the following types:

- [`HyperReduction`](@ref) for single field problems
- [`BlockProjection`](@ref) for multi field problems
"""
struct AffineContribution{A<:Projection,V,K} <: Contribution
  values::V
  trians::K
  function AffineContribution(
    values::V,
    trians::K
    ) where {A,V<:Tuple{Vararg{A}},K<:Tuple{Vararg{Triangulation}}}

    @check length(values) == length(trians)
    @check !any([t === first(trians) for t = trians[2:end]])
    new{A,V,K}(values,trians)
  end
end

function allocate_coefficient(a::AffineContribution,r::AbstractRealization)
  contribution(get_domains(a)) do trian
    allocate_coefficient(a[trian],r)
  end
end

function allocate_hyper_reduction(a::AffineContribution,r::AbstractRealization)
  allocate_hyper_reduction(first(get_contributions(a)),r)
end

"""
"""
function allocate_hypred_cache(a::AffineContribution,r::AbstractRealization)
  fecache = allocate_coefficient(a,r)
  coeffs = allocate_coefficient(a,r)
  hypred = allocate_hyper_reduction(a,r)
  return HRParamArray(fecache,coeffs,hypred)
end

function inv_project!(
  hypred::AbstractParamArray,
  coeff::ArrayContribution,
  a::AffineContribution,
  b::ArrayContribution)

  @check length(coeff) == length(a) == length(b)
  fill!(hypred,zero(eltype(hypred)))
  for (aval,bval,cval) in zip(get_contributions(a),get_contributions(b),get_contributions(coeff))
    inv_project!(hypred,cval,aval,bval)
  end
  return hypred
end

function reduced_form(red::Reduction,s,trian::Triangulation,args...)
  hyper_red = HyperReduction(red,s,trian,args...)
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
  s::AbstractSnapshots)

  res = residual_snapshots(solver,op,s)
  res_red = get_residual_reduction(solver)
  reduced_residual(res_red,red_test,res)
end

function reduced_residual(red::Reduction,test::RBSpace,c::ArrayContribution)
  t = @timed begin
    a,trians = map(get_domains(c),get_contributions(c)) do trian,values
      reduced_form(red,values,trian,test)
    end |> tuple_of_arrays
  end
  println(CostTracker(t,name="Residual hyper-reduction"))
  return Contribution(a,trians)
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
  s::AbstractSnapshots)

  jac = jacobian_snapshots(solver,op,s)
  jac_red = get_jacobian_reduction(solver)
  reduced_jacobian(jac_red,red_trial,red_test,jac)
end

function reduced_jacobian(red::Reduction,trial::RBSpace,test::RBSpace,c::ArrayContribution)
  t = @timed begin
    a,trians = map(get_domains(c),get_contributions(c)) do trian,values
      reduced_form(red,values,trian,trial,test)
    end |> tuple_of_arrays
  end
  println(CostTracker(t,name="Jacobian hyper-reduction"))
  return Contribution(a,trians)
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
  s::AbstractSnapshots)

  red_jac = reduced_jacobian(solver,op,red_trial,red_test,s)
  red_res = reduced_residual(solver,op,red_test,s)
  return red_jac,red_res
end

# multi field interface

"""
    const BlockHyperReduction{A<:HyperReduction,N} = BlockProjection{A,N}
"""
const BlockHyperReduction{A<:HyperReduction,N} = BlockProjection{A,N}

function Utils.Contribution(
  v::Tuple{Vararg{BlockHyperReduction}},
  t::Tuple{Vararg{Triangulation}})

  AffineContribution(v,t)
end

for f in (:get_basis,:get_interpolation)
  @eval begin
    function Arrays.return_cache(::typeof($f),a::BlockHyperReduction)
      cache = $f(testitem(a))
      block_cache = Array{typeof(cache),ndims(a)}(undef,size(a))
      return block_cache
    end

    function $f(a::BlockHyperReduction)
      cache = return_cache($f,a)
      for i in eachindex(a)
        if a.touched[i]
          cache[i] = $f(a[i])
        end
      end
      return ArrayBlock(cache,a.touched)
    end
  end
end

function Arrays.return_cache(::typeof(get_integration_cells),a::BlockHyperReduction)
  ntouched = length(findall(a.touched))
  cache = get_integration_cells(testitem(a))
  block_cache = Vector{typeof(cache)}(undef,ntouched)
  return block_cache
end

function get_integration_cells(a::BlockHyperReduction)
  cache = return_cache(get_integration_cells,a)
  count = 0
  for i in eachindex(a)
    if a.touched[i]
      count += 1
      cache[count] = get_integration_cells(a[i])
    end
  end
  return union(cache...)
end

function get_owned_icells(a::BlockHyperReduction)
  cells = get_integration_cells(a)
  get_owned_icells(a,cells)
end

function Arrays.return_cache(::typeof(get_owned_icells),a::BlockHyperReduction,cells)
  cache = get_owned_icells(testitem(a),cells)
  block_cache = Array{typeof(cache),ndims(a)}(undef,size(a))
  return block_cache
end

function get_owned_icells(a::BlockHyperReduction,cells::AbstractVector)
  cache = return_cache(get_owned_icells,a,cells)
  for i in eachindex(a)
    if a.touched[i]
      cache[i] = get_owned_icells(a[i],cells)
    end
  end
end

function inv_project!(
  hypred::BlockParamArray,
  coeff::ArrayBlock,
  a::BlockHyperReduction,
  b::ArrayBlock)

  for i in eachindex(a)
    if a.touched[i]
      inv_project!(blocks(hypred)[i],coeff[i],a[i],b[i])
    end
  end
  return hypred
end

for T in (:AffineContribution,:BlockHyperReduction)
  @eval begin
    function inv_project!(cache::HRParamArray,a::$T)
      inv_project!(cache.hypred,cache.coeff,a,cache.fecache)
    end
  end
end

function Arrays.return_cache(
  ::typeof(allocate_coefficient),
  a::HyperReduction,
  r::AbstractRealization)

  coeffvec = testvalue(Vector{Float64})
  global_parameterize(coeffvec,num_params(r))
end

function Arrays.return_cache(
  ::typeof(allocate_coefficient),
  a::BlockHyperReduction,
  r::AbstractRealization)

  i = findfirst(a.touched)
  @notimplementedif isnothing(i)
  coeff = return_cache(allocate_coefficient,a[i],r)
  block_coeff = Array{typeof(coeff),ndims(a)}(undef,size(a))
  return block_coeff
end

function allocate_coefficient(a::BlockHyperReduction,r::AbstractRealization)
  coeff = return_cache(allocate_coefficient,a,r)
  for i in eachindex(a)
    if a.touched[i]
      coeff[i] = allocate_coefficient(a[i],r)
    end
  end
  return ArrayBlock(coeff,a.touched)
end

function Arrays.return_cache(
  ::typeof(allocate_hyper_reduction),
  a::HyperReduction{<:Reduction,<:ReducedVecProjection},
  r::AbstractRealization)

  hypvec = testvalue(Vector{Float64})
  global_parameterize(hypvec,num_params(r))
end

function Arrays.return_cache(
  ::typeof(allocate_hyper_reduction),
  a::HyperReduction{<:Reduction,<:ReducedMatProjection},
  r::AbstractRealization)

  hypvec = testvalue(Matrix{Float64})
  global_parameterize(hypvec,num_params(r))
end

function Arrays.return_cache(
  ::typeof(allocate_hyper_reduction),
  a::BlockHyperReduction,
  r::AbstractRealization)

  i = findfirst(a.touched)
  @notimplementedif isnothing(i)
  hypred = return_cache(allocate_hyper_reduction,a[i],r)
  block_hypred = Array{typeof(hypred),ndims(a)}(undef,size(a))
  return block_hypred
end

function allocate_hyper_reduction(a::BlockHyperReduction,r::AbstractRealization)
  hypred = return_cache(allocate_hyper_reduction,a,r)
  for i in eachindex(a)
    hypred[i] = allocate_hyper_reduction(a.array[i],r)
  end
  return mortar(hypred)
end

function reduced_triangulation(trian::Triangulation,b::BlockHyperReduction)
  red_trian = Triangulation[]
  for i in eachindex(b)
    if b.touched[i]
      push!(red_trian,reduced_triangulation(trian,b[i]))
    end
  end
  return red_trian
end

function reduced_form(
  red::Reduction,
  s::BlockSnapshots,
  trian::Triangulation,
  test::MultiFieldRBSpace)

  hyper_reds = Vector{HyperReduction}(undef,size(s))
  for i in eachindex(s)
    hyper_red, = reduced_form(red,s[i],trian,test[i])
    hyper_reds[i] = hyper_red
  end

  hyper_red = BlockProjection(hyper_reds,s.touched)
  red_cells = get_integration_cells(hyper_red)
  red_trian = view(trian,red_cells)

  return hyper_red,red_trian
end

function reduced_form(
  red::Reduction,
  s::BlockSnapshots,
  trian::Triangulation,
  trial::MultiFieldRBSpace,
  test::MultiFieldRBSpace)

  hyper_reds = Matrix{HyperReduction}(undef,size(s))
  for (i,j) in Iterators.product(axes(s)...)
    hyper_red, = reduced_form(red,s[i,j],trian,trial[j],test[i])
    hyper_reds[i,j] = hyper_red
  end

  hyper_red = BlockProjection(hyper_reds,s.touched)
  red_cells = get_integration_cells(hyper_red)
  red_trian = view(trian,red_cells)

  return hyper_red,red_trian
end
