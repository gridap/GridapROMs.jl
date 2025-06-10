"""
    abstract type HRProjection{A<:ReducedProjection} <: Projection end

Subtype of a [`Projection`](@ref) dedicated to the outputd of a hyper-reduction
(e.g. an empirical interpolation method (EIM)) procedure applied on residual/jacobians
of a differential problem. This procedure can be summarized in the following steps:

1. compute a snapshots tensor `T`
2. construct a `Projection` `Φ` by running the function `reduction` on `T`
3. find the EIM quantities `(Φi,i)`, by running the function `empirical_interpolation`
  on `Φ`

The triplet `(Φ,Φi,i)` represents the minimum information needed to run the
online phase of the hyper-reduction. However, we recall that a RB method requires
the (Petrov-)Galerkin projection of residuals/Jacobianson a reduced subspace
built from solution snapshots, instead of providing the projection `Φ` we return
the reduced projection `Φrb`, where

- for residuals: `Φrb = test_basisᵀ Φ`
- for Jacobians: `Φrb = test_basisᵀ Φ trial_basis`

The output of this operation is a ReducedProjection. Therefore, a HRProjection
is completely characterized by the triplet `(Φrb,Φi,i)`.
Subtypes:
- [`TrivialHRProjection`](@ref)
- [`MDEIM`](@ref)
"""
abstract type HRProjection{A<:ReducedProjection} <: Projection end

const AbstractHRProjection = Union{HRProjection,BlockProjection{<:HRProjection}}

const HRVecProjection = HRProjection{<:ReducedVecProjection}
const HRMatProjection = HRProjection{<:ReducedMatProjection}

HRProjection(::Reduction,args...) = @abstractmethod

"""
    get_interpolation(a::HRProjection) -> Factorization

For a [`HRProjection`](@ref) `a` represented by the triplet `(Φrb,Φi,i)`,
returns `Φi`, usually stored as a Factorization
"""
get_interpolation(a::HRProjection) = @abstractmethod

"""
    get_integration_domain(a::HRProjection) -> IntegrationDomain

For a [`HRProjection`](@ref) `a` represented by the triplet `(Φrb,Φi,i)`,
returns `i`
"""
get_integration_domain(a::HRProjection) = @abstractmethod

get_integration_cells(a::HRProjection,args...) = get_integration_cells(get_integration_domain(a),args...)
get_cellids_rows(a::HRProjection) = get_cellids_rows(get_integration_domain(a))
get_cellids_cols(a::HRProjection) = get_cellids_cols(get_integration_domain(a))
get_owned_icells(a::HRProjection,args...) = get_owned_icells(a,get_integration_cells(a,args...))
get_owned_icells(a::HRProjection,cells::AbstractVector) = get_owned_icells(get_integration_domain(a),cells)

num_reduced_dofs(a::HRProjection) = num_reduced_dofs(get_basis(a))
num_reduced_dofs_left_projector(a::HRProjection) = num_reduced_dofs_left_projector(get_basis(a))
num_reduced_dofs_right_projector(a::HRProjection) = num_reduced_dofs_right_projector(get_basis(a))

function inv_project!(
  b̂::AbstractArray,
  coeff::AbstractArray,
  a::HRProjection,
  b::AbstractArray)

  o = one(eltype2(b̂))
  interp = get_interpolation(a)
  ldiv!(coeff,interp,b)
  mul!(b̂,a,coeff,o,o)
  return b̂
end

"""
    struct TrivialHRProjection{A} <: HRProjection{A}
      basis::A
    end

Trivial hyper-reduction returned whenever the residual/Jacobian is zero
"""
struct TrivialHRProjection{A} <: HRProjection{A}
  basis::A
end

get_basis(a::TrivialHRProjection) = a.basis
get_interpolation(a::TrivialHRProjection) = @notimplemented
get_integration_domain(a::TrivialHRProjection) = @notimplemented

function HRProjection(red::Reduction,s::Nothing,trian::Triangulation,test::RBSpace)
  nrows = num_reduced_dofs(test)
  basis = ReducedProjection(zeros(nrows,1))
  return TrivialHRProjection(basis)
end

function HRProjection(red::Reduction,s::Nothing,trian::Triangulation,trial::RBSpace,test::RBSpace)
  nrows = num_reduced_dofs(test)
  ncols = num_reduced_dofs(trial)
  basis = ReducedProjection(zeros(nrows,1,ncols))
  return TrivialHRProjection(basis)
end

"""
    struct MDEIM{A} <: HRProjection{A}
      basis::A
      interpolation::Factorization
      domain::IntegrationDomain
    end

[`HRProjection`](@ref) returned by a matrix-based empirical interpolation method
"""
struct MDEIM{A} <: HRProjection{A}
  basis::A
  interpolation::Factorization
  domain::IntegrationDomain
end

get_basis(a::MDEIM) = a.basis
get_interpolation(a::MDEIM) = a.interpolation
get_integration_domain(a::MDEIM) = a.domain

function HRProjection(
  red::MDEIMReduction,
  s::Snapshots,
  trian::Triangulation,
  test::RBSpace)

  red = get_reduction(red)
  basis = projection(red,s)
  proj_basis = project(test,basis)
  rows,interp = empirical_interpolation(basis)
  factor = lu(interp)
  domain = vector_domain(trian,test,rows)
  return MDEIM(proj_basis,factor,domain)
end

function HRProjection(
  red::MDEIMReduction,
  s::Snapshots,
  trian::Triangulation,
  trial::RBSpace,
  test::RBSpace)

  red = get_reduction(red)
  basis = projection(red,s)
  proj_basis = project(test,basis,trial)
  (rows,cols),interp = empirical_interpolation(basis)
  factor = lu(interp)
  domain = matrix_domain(trian,trial,test,rows,cols)
  return MDEIM(proj_basis,factor,domain)
end

"""
    struct InterpHRProjection{A} <: HRProjection{A}
      a::HRProjection{A}
      interp::Interpolator
    end

Hyper-reduction with pre-trained online coefficients. During the online phase,
for any new parameter, we simply need to evaluate the interpolator `interp`, and
the result will be the new reduced coefficient
"""
struct InterpHRProjection{A} <: HRProjection{A}
  basis::A
  interpolation::Interpolator
end

get_basis(a::InterpHRProjection) = a.basis
get_interpolation(a::InterpHRProjection) = a.interpolation

function HRProjection(
  hred::InterpHyperReduction,
  s::Snapshots,
  trian::Triangulation,
  args...)

  HRProjection(hred,s,args...)
end

function HRProjection(
  hred::InterpHyperReduction,
  s::Snapshots,
  test::RBSpace)

  red = get_reduction(hred)
  basis = projection(red,s)
  proj_basis = project(test,basis)
  coeff_interp = get_interpolator(hred,basis,s)
  return InterpHRProjection(proj_basis,coeff_interp)
end

function HRProjection(
  hred::InterpHyperReduction,
  s::Snapshots,
  trial::RBSpace,
  test::RBSpace)

  red = get_reduction(hred)
  basis = projection(red,s)
  proj_basis = project(test,basis,trial)
  coeff_interp = get_interpolator(hred,basis,s)
  return InterpHRProjection(proj_basis,coeff_interp)
end

function get_interpolator(red::InterpHyperReduction,a::Projection,s::Snapshots)
  get_interpolator(interp_strategy(red),a,s)
end

function get_interpolator(strategy,a::Projection,s::Snapshots)
  @notimplemented "Only implemented an interpolator with radial bases"
end

function get_interpolator(strategy::AbstractRadialBasis,a::Projection,s::Snapshots)
  inds,interp = empirical_interpolation(a)
  factor = lu(interp)
  r = get_realization(s)
  red_data = get_at_domain(s,inds)
  coeff = allocate_coefficient(a,r)
  ldiv!(coeff,factor,red_data)
  Interpolator(r,coeff,strategy)
end

function FESpaces.interpolate(a::InterpHRProjection,r)
  cache = allocate_hypred_cache(a,r)
  inv_project!(cache,a,r)
  return cache
end

function reduced_triangulation(trian::Triangulation,a::TrivialHRProjection)
  red_trian = view(trian,Int[])
  return red_trian
end

"""
    reduced_triangulation(trian::Triangulation,a::HRProjection)

Returns the triangulation view of `trian` on the integration cells contained in `a`
"""
function reduced_triangulation(trian::Triangulation,a::HRProjection)
  red_cells = get_integration_cells(a)
  red_trian = view(trian,red_cells)
  return red_trian
end

function reduced_triangulation(trian::Triangulation,a::InterpHRProjection)
  red_trian = trian
  return red_trian
end

function allocate_coefficient(a::Projection)
  n = num_reduced_dofs(a)
  coeff = zeros(n)
  return coeff
end

function allocate_hyper_reduction(a::HRVecProjection)
  nrows = num_reduced_dofs_left_projector(a)
  hypred = zeros(nrows)
  fill!(hypred,zero(eltype(hypred)))
  return hypred
end

function allocate_hyper_reduction(a::HRMatProjection)
  nrows = num_reduced_dofs_left_projector(a)
  ncols = num_reduced_dofs_right_projector(a)
  hypred = zeros(nrows,ncols)
  fill!(hypred,zero(eltype(hypred)))
  return hypred
end

for f in (:allocate_coefficient,:allocate_hyper_reduction)
  @eval $f(a::Projection,r::AbstractRealization) = global_parameterize($f(a),num_params(r))
end

"""
    const AffineContribution{V<:Projection} = Contribution{V}

[`Contribution`](@ref) whose field `values` are Projections
"""
const AffineContribution{V<:Projection} = Contribution{V}

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

function inv_project!(
  hypred::AbstractArray,
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

for T in (:AbstractRealization,:AbstractArray)
  @eval begin
    function inv_project!(
      b̂::AbstractArray,
      coeff::AbstractArray,
      a::InterpHRProjection,
      r::$T)

      o = one(eltype2(b̂))
      interp = get_interpolation(a)
      interpolate!(coeff,interp,r)
      mul!(b̂,a,coeff,o,o)
      return b̂
    end

    function inv_project!(cache::AbstractHRArray,a::InterpHRProjection,r::$T)
      inv_project!(cache.hypred,cache.coeff,a,r)
    end
  end
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
  s::AbstractSnapshots)

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

function reduced_residual(red::Reduction,test::RBSpace,s::Snapshots)
  reduced_form(red,s,test)
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

function reduced_jacobian(red::Reduction,trial::RBSpace,test::RBSpace,s::Snapshots)
  reduced_form(red,s,trial,test)
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
    const BlockHRProjection{A<:HRProjection,N} = BlockProjection{A,N}
"""
const BlockHRProjection{A<:HRProjection,N} = BlockProjection{A,N}

for f in (:get_basis,:get_interpolation)
  @eval begin
    function Arrays.return_cache(::typeof($f),a::BlockHRProjection)
      cache = $f(testitem(a))
      block_cache = Array{typeof(cache),ndims(a)}(undef,size(a))
      return block_cache
    end

    function $f(a::BlockHRProjection)
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

for f in (:get_cellids_rows,:get_cellids_cols)
  @eval begin
    function Arrays.return_cache(::typeof($f),a::BlockHRProjection)
      block_cache = Array{Table,ndims(a)}(undef,size(a))
      return block_cache
    end

    function $f(a::BlockHRProjection)
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

function Arrays.return_cache(::typeof(get_integration_cells),a::BlockHRProjection,args...)
  ntouched = length(findall(a.touched))
  cache = get_integration_cells(testitem(a),args...)
  block_cache = Vector{typeof(cache)}(undef,ntouched)
  return block_cache
end

function get_integration_cells(a::BlockHRProjection,args...)
  _union(a) = a
  _union(a,b) = union(a,b)
  _union(a::AppendedArray,b::AppendedArray) = lazy_append(union(a.a,b.a),union(a.b,b.b))
  _union(a,b,c...) = _union(_union(a,b),c...)

  cache = return_cache(get_integration_cells,a,args...)
  count = 0
  for i in eachindex(a)
    if a.touched[i]
      count += 1
      cache[count] = get_integration_cells(a[i],args...)
    end
  end
  return _union(cache...)
end

function get_owned_icells(a::BlockHRProjection,args...)
  cells = get_integration_cells(a,args...)
  get_owned_icells(a,cells)
end

function Arrays.return_cache(::typeof(get_owned_icells),a::BlockHRProjection,cells)
  cache = get_owned_icells(testitem(a),cells)
  block_cache = Array{typeof(cache),ndims(a)}(undef,size(a))
  return block_cache
end

function get_owned_icells(a::BlockHRProjection,cells::AbstractVector)
  cache = return_cache(get_owned_icells,a,cells)
  for i in eachindex(a)
    if a.touched[i]
      cache[i] = get_owned_icells(a[i],cells)
    end
  end
  return ArrayBlock(cache,a.touched)
end

function inv_project!(
  hypred::Union{BlockParamArray,BlockArray},
  coeff::ArrayBlock,
  a::BlockHRProjection,
  b::ArrayBlock)

  for i in eachindex(a)
    if a.touched[i]
      inv_project!(blocks(hypred)[i],coeff[i],a[i],b[i])
    end
  end
  return hypred
end

for T in (:AffineContribution,:BlockHRProjection)
  @eval begin
    function inv_project!(cache::AbstractHRArray,a::$T)
      inv_project!(cache.hypred,cache.coeff,a,cache.fecache)
    end
  end
end

function reduced_triangulation(trian::Triangulation,a::BlockHRProjection)
  red_cells = get_integration_cells(a)
  red_trian = view(trian,red_cells)
  return red_trian
end

function Arrays.return_cache(
  ::typeof(allocate_coefficient),
  a::HRProjection,
  r::AbstractRealization)

  coeffvec = testvalue(Vector{Float64})
  global_parameterize(coeffvec,num_params(r))
end

function Arrays.return_cache(
  ::typeof(allocate_coefficient),
  a::BlockHRProjection,
  args...)

  i = findfirst(a.touched)
  @notimplementedif isnothing(i)
  coeff = return_cache(allocate_coefficient,a[i],r)
  block_coeff = Array{typeof(coeff),ndims(a)}(undef,size(a))
  return block_coeff
end

function allocate_coefficient(a::BlockHRProjection,args...)
  coeff = return_cache(allocate_coefficient,a,args...)
  for i in eachindex(a)
    if a.touched[i]
      coeff[i] = allocate_coefficient(a[i],args...)
    end
  end
  return ArrayBlock(coeff,a.touched)
end

function Arrays.return_cache(
  ::typeof(allocate_hyper_reduction),
  a::HRVecProjection)

  testvalue(Vector{Float64})
end

function Arrays.return_cache(
  ::typeof(allocate_hyper_reduction),
  a::HRMatProjection)

  testvalue(Matrix{Float64})
end

function Arrays.return_cache(
  ::typeof(allocate_hyper_reduction),
  a::HRProjection,
  r::AbstractRealization)

  hypvec = return_cache(allocate_hyper_reduction,a)
  global_parameterize(hypvec,num_params(r))
end

function Arrays.return_cache(
  ::typeof(allocate_hyper_reduction),
  a::BlockHRProjection,
  args...)

  i = findfirst(a.touched)
  @notimplementedif isnothing(i)
  hypred = return_cache(allocate_hyper_reduction,a[i],args...)
  block_hypred = Array{typeof(hypred),ndims(a)}(undef,size(a))
  return block_hypred
end

function allocate_hyper_reduction(a::BlockHRProjection,args...)
  hypred = return_cache(allocate_hyper_reduction,a,args...)
  for i in eachindex(a)
    hypred[i] = allocate_hyper_reduction(a.array[i],args...)
  end
  return mortar(hypred)
end

function reduced_form(
  red::Reduction,
  s::BlockSnapshots,
  trian::Triangulation,
  test::MultiFieldRBSpace)

  hyper_reds = Vector{HRProjection}(undef,size(s))
  for i in eachindex(s)
    hyper_red, = reduced_form(red,s[i],trian,test[i])
    hyper_reds[i] = hyper_red
  end

  hyper_red = BlockProjection(hyper_reds,s.touched)
  red_trian = reduced_triangulation(trian,hyper_red)

  return hyper_red,red_trian
end

function reduced_form(
  red::Reduction,
  s::BlockSnapshots,
  trian::Triangulation,
  trial::MultiFieldRBSpace,
  test::MultiFieldRBSpace)

  hyper_reds = Matrix{HRProjection}(undef,size(s))
  for (i,j) in Iterators.product(axes(s)...)
    hyper_red, = reduced_form(red,s[i,j],trian,trial[j],test[i])
    hyper_reds[i,j] = hyper_red
  end

  hyper_red = BlockProjection(hyper_reds,s.touched)
  red_trian = reduced_triangulation(trian,hyper_red)

  return hyper_red,red_trian
end

# interpolation utils

function get_at_domain(s::Snapshots,rows::AbstractVector{<:Integer})
  data = get_all_data(s)
  datav = view(data,rows,:)
  ConsecutiveParamArray(datav)
end

function get_at_domain(s::SparseSnapshots,rowscols::Tuple)
  rows,cols = rowscols
  sparsity = get_sparsity(get_dof_map(s))
  inds = sparsify_split_indices(rows,cols,sparsity)
  data = get_all_data(s)
  datav = view(data,inds,:)
  ConsecutiveParamArray(datav)
end

function RadialBasisFunctions.Interpolator(
  x::Realization,
  y::ConsecutiveParamArray,
  basis::B=PHS()
  ) where B<:AbstractRadialBasis

  dim = length(first(x))
  k = param_length(x)
  npoly = binomial(dim+basis.poly_deg,basis.poly_deg)
  n = k + npoly
  mon = MonomialBasis(dim,basis.poly_deg)
  data_type = promote_type(eltype(first(x.params)),eltype2(y))
  A = Symmetric(zeros(data_type,n,n))
  RadialBasisFunctions._build_collocation_matrix!(A,x.params,basis,mon,k)
  factor = factorize(A)
  l = innerlength(y)
  w = zeros(data_type,n,l)
  b = zeros(data_type,n,l)
  z = zero(data_type)
  for j in 1:l
    for i in 1:n
      b[i,j] = i < k ? y.data[j,i] : z
    end
  end
  ldiv!(w,factor,b)
  return Interpolator(x,y,view(w,1:k,:),view(w,1+k:n,:),basis,mon)
end

(rbfi::Interpolator)(x::AbstractRealization) = interpolate(rbfi,x)

function FESpaces.interpolate(rbfi::Interpolator,x::AbstractRealization)
  k′ = param_length(x)
  l = size(rbfi.rbf_weights,2)
  cache = ConsecutiveParamArray(zeros(l,k′))
  interpolate!(cache,rbfi,x)
  return cache
end

function FESpaces.interpolate!(cache::ConsecutiveParamVector,rbfi::Interpolator,x::AbstractRealization)
  k′ = param_length(x)
  l = size(rbfi.rbf_weights,2)

  for j in 1:l
    for i in 1:k′
      rbfji = 0.0
      for q in axes(rbfi.rbf_weights,1)
        rbfji += rbfi.rbf_weights[q,j]*rbfi.rbf_basis(x.params[i],rbfi.x.params[q])
      end

      if !isempty(rbfi.monomial_weights)
        val_poly = rbfi.monomial_basis(x.params[i])
        for (q,val) in enumerate(val_poly)
          rbfji += rbfi.monomial_weights[q,j]*val
        end
      end

      cache.data[j,i] = rbfji
    end
  end
end
