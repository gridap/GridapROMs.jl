function RBSteady.check_interpolation(snaps,interp::DistributedInterpolation,_fecache)
  msg = "fecache mismatch at interpolation points"
  fecache = sreduce(map(get_all_data,local_views(_fecache)))
  sdofs = get_at_domain(snaps,interp)
  @check isapprox(fecache,get_all_data(sdofs);rtol=1e-8) msg
  return true
end

const DOFMAP_LABEL = "dofmap"
const HRPROJECTION_LABEL = "hrprojection"
const NORM_MATRIX_LABEL = "norm"
const BLOCK_LABEL = "block"
const TRIAN_LABEL = "trian"

function DrWatson.save(dir,s::DistributedSnapshots;label="")
  _psave(dir,SNAPSHOTS_LABEL,s.snaps;label)
end

function DrWatson.save(dir,s::DistributedBlockSnapshots;label="")
  for i in eachindex(blocks(s))
    save(dir,blocks(s)[i];label=_plabel(label,BLOCK_LABEL*"$i"))
  end
end

function RBSteady.load_snapshots(dir,ranks::AbstractArray;label="")
  if _haspart(dir,SNAPSHOTS_LABEL,ranks;label=_plabel(label,BLOCK_LABEL*"1"))
    nblocks = 0
    while _haspart(dir,SNAPSHOTS_LABEL,ranks;label=_plabel(label,BLOCK_LABEL*"$(nblocks+1)"))
      nblocks += 1
    end
    array = map(1:nblocks) do i
      _pload(dir,SNAPSHOTS_LABEL,ranks;label=_plabel(label,BLOCK_LABEL*"$i"))
    end
    param_data = mortar(map(get_param_data,array))
    BlockSnapshots(array,param_data)
  else
    snaps = _pload(dir,SNAPSHOTS_LABEL,ranks;label)
    DistributedSnapshots(snaps)
  end
end

function DrWatson.save(dir,a::DistributedProjection;label="")
  _psave(dir,PROJECTION_LABEL,a.basis;label)
  _psave(dir,DOFMAP_LABEL,a.dof_map;label)
end

function DrWatson.save(dir,a::DistributedNormedProjection;label="")
  save(dir,a.projection;label)
  _psave(dir,NORM_MATRIX_LABEL,a.norm_matrix;label)
end

function DrWatson.save(dir,a::DistributedKroneckerProjection;label="")
  save(dir,a.projection_space;label=_get_label(label,"space"))
  save(dir,a.projection_time;label=_get_label(label,"time"))
end

for T in (:DistributedProjection,:DistributedNormedProjection,:DistributedKroneckerProjection)
  @eval begin
    function DrWatson.save(dir,a::BlockProjection{<:$T};label="")
      for i in eachindex(a)
        save(dir,a[i];label=_plabel(label,BLOCK_LABEL*"$i"))
      end
    end
  end
end

function RBSteady.load_projection(dir,ranks::AbstractArray;label="")
  if _haspart(dir,PROJECTION_LABEL,ranks;label=_plabel(label,BLOCK_LABEL*"1"))
    # 1) block projection
    nblocks = 0
    while _haspart(dir,PROJECTION_LABEL,ranks;label=_plabel(label,BLOCK_LABEL*"$(nblocks+1)"))
      nblocks += 1
    end
    block_basis = map(1:nblocks) do i
      load_projection(dir,ranks;label=_plabel(label,BLOCK_LABEL*"$i"))
    end
    return BlockProjection(block_basis)
  elseif _haspart(dir,PROJECTION_LABEL,ranks;label=_get_label(label,"space"))
    # 2) kronecker projection: the space part is distributed, the time part is not
    projection_space = load_projection(dir,ranks;label=_get_label(label,"space"))
    projection_time = load_projection(dir;label=_get_label(label,"time"))
    return KroneckerProjection(projection_space,projection_time)
  else
    basis = _pload(dir,PROJECTION_LABEL,ranks;label)
    dof_map = _pload(dir,DOFMAP_LABEL,ranks;label)
    proj = Projection(basis,dof_map)
    if _haspart(dir,NORM_MATRIX_LABEL,ranks;label)
      # 3) normed projection
      X = _pload(dir,NORM_MATRIX_LABEL,ranks;label)
      return NormedProjection(proj,X)
    end
    # 4) generic projection
    return proj
  end
end

function DrWatson.save(dir,a::DistributedHRProjection;label="")
  map(local_views(a),linear_indices(local_views(a))) do a,p
    serialize(_part_filename(dir,HRPROJECTION_LABEL,label,p),a)
  end
end

function DrWatson.save(dir,a::BlockHRProjection{<:HyperReduction,<:Projection,<:DistributedHRProjection};label="")
  for i in eachindex(a)
    save(dir,a[i];label=_plabel(label,BLOCK_LABEL*"$i"))
  end
end

function RBSteady.load_subspace(dir,f::DistributedSingleFieldFESpace,ranks::AbstractArray;label="")
  basis = load_projection(dir,ranks;label)
  reduced_subspace(f,basis)
end

function RBSteady.load_subspace(dir,f::DistributedMultiFieldFESpace,ranks::AbstractArray;label="")
  basis = load_projection(dir,ranks;label)
  reduced_subspace(f,basis)
end

# `ranks::Vector`: reassemble a serial (non-distributed) reduced subspace, over
# an ordinary (non-distributed) FE space `f` matching the one the distributed
# problem was solved on.
function RBSteady.load_subspace(dir,f::FESpace,ranks::Vector;label="")
  basis = load_projection(dir,ranks;label)
  reduced_subspace(f,basis)
end

function DrWatson.save(dir,contrib::Contribution{V,T};label="") where {V,T<:DistributedTriangulation}
  for (i,v) in enumerate(get_contributions(contrib))
    save(dir,v;label=_plabel(label,"$(TRIAN_LABEL)$i"))
  end
end

function RBSteady.load_contribution(dir,trian::Tuple{Vararg{DistributedTriangulation}},ranks::AbstractArray;label="")
  vals = ntuple(length(trian)) do i
    _load_distributed_hr(dir,ranks;label=_plabel(label,"$(TRIAN_LABEL)$i"))
  end
  RBSteady._setup_contribution(vals,trian)
end

# `ranks::Vector`: reassemble a serial (non-distributed) contribution, over
# ordinary (non-distributed) triangulations `trian` matching the ones the
# distributed problem was solved on.
function RBSteady.load_contribution(dir,trian::Tuple,ranks::Vector;label="")
  vals = ntuple(length(trian)) do i
    _load_distributed_hr(dir,ranks;label=_plabel(label,"$(TRIAN_LABEL)$i"))
  end
  RBSteady._setup_contribution(vals,trian)
end

function RBSteady.load_operator(dir,feop::ParamOperator,ranks::AbstractArray;label="")
  test = load_subspace(dir,get_test(feop),ranks;label=_plabel(label,RBSteady.TEST_LABEL))
  trial = load_subspace(dir,get_trial(feop),ranks;label=_plabel(label,RBSteady.TRIAL_LABEL))
  trian_res = get_domains_res(feop)
  trian_jac = get_domains_jac(feop)
  red_rhs = load_contribution(dir,trian_res,ranks;label=_plabel(label,RBSteady.RHS_LABEL))
  red_lhs = load_contribution(dir,trian_jac,ranks;label=_plabel(label,RBSteady.LHS_LABEL))
  ROMOperator(feop,trial,test,red_lhs,red_rhs)
end

function RBSteady.load_operator(dir,feop::ODEParamOperator,ranks::AbstractArray;label="")
  test = load_subspace(dir,get_test(feop),ranks;label=_plabel(label,RBSteady.TEST_LABEL))
  trial = load_subspace(dir,get_trial(feop),ranks;label=_plabel(label,RBSteady.TRIAL_LABEL))
  trian_res = get_domains_res(feop)
  trian_jacs = get_domains_jac(feop)
  red_rhs = load_contribution(dir,trian_res,ranks;label=_plabel(label,RBSteady.RHS_LABEL))
  red_lhs = ntuple(length(trian_jacs)) do i
    load_contribution(dir,trian_jacs[i],ranks;label=_plabel(label,RBSteady.LHS_LABEL,i))
  end
  ROMOperator(feop,trial,test,red_lhs,red_rhs)
end

function RBSteady.load_operator(dir,feop::LinearNonlinearODEParamOperator,ranks::AbstractArray;label="")
  feop_lin = get_linear_operator(feop)
  feop_nlin = get_nonlinear_operator(feop)
  # test and trial are the same for both the linear and nonlinear operators
  test = load_subspace(dir,get_test(feop_lin),ranks;label=_plabel(label,RBSteady.TEST_LABEL))
  trial = load_subspace(dir,get_trial(feop_lin),ranks;label=_plabel(label,RBSteady.TRIAL_LABEL))
  trian_res_lin = get_domains_res(feop_lin)
  trian_jacs_lin = get_domains_jac(feop_lin)
  red_rhs_lin = load_contribution(dir,trian_res_lin,ranks;label=_plabel(label,RBSteady.LINEAR_LABEL,RBSteady.RHS_LABEL))
  red_lhs_lin = ntuple(length(trian_jacs_lin)) do i
    load_contribution(dir,trian_jacs_lin[i],ranks;label=_plabel(label,RBSteady.LINEAR_LABEL,RBSteady.LHS_LABEL,i))
  end
  trian_res_nlin = get_domains_res(feop_nlin)
  trian_jacs_nlin = get_domains_jac(feop_nlin)
  red_rhs_nlin = load_contribution(dir,trian_res_nlin,ranks;label=_plabel(label,RBSteady.NONLINEAR_LABEL,RBSteady.RHS_LABEL))
  red_lhs_nlin = ntuple(length(trian_jacs_nlin)) do i
    load_contribution(dir,trian_jacs_nlin[i],ranks;label=_plabel(label,RBSteady.NONLINEAR_LABEL,RBSteady.LHS_LABEL,i))
  end
  op_lin = ROMOperator(feop_lin,trial,test,red_lhs_lin,red_rhs_lin)
  op_nlin = ROMOperator(feop_nlin,trial,test,red_lhs_nlin,red_rhs_nlin)
  LinearNonlinearROMOperator(op_lin,op_nlin)
end

# utils

_plabel(name,label...) = foldl(RBSteady._get_label,label;init=name)
_part_name(name,label,part) = _plabel(name,label,"part$part")
_part_filename(dir,name,label,part) = joinpath(dir,_part_name(name,label,part)*".jld")
_haspart(dir,name,ranks;label="") = isfile(_part_filename(dir,name,label,getany(ranks)))

function _psave(dir,name,x::Union{GenericPArray,PVector};label="")
  map(partition(x),partition(axes(x,1))) do xloc,ind
    serialize(_part_filename(dir,name,label,part_id(ind)),(xloc,ind))
  end
end

function _psave(dir,name,x::PSparseMatrix;label="")
  map(partition(x),partition(axes(x,1)),partition(axes(x,2))) do xloc,rind,cind
    serialize(_part_filename(dir,name,label,part_id(rind)),(xloc,rind,cind))
  end
end

function _pload(dir,name,ranks;label="")
  data,inds... = map(ranks) do p
    deserialize(_part_filename(dir,name,label,p))
  end |> tuple_of_arrays
  _allocate(ranks,data,inds...)
end

_allocate(ranks::Vector,d,i...) = _sallocate(d,i...)
_allocate(ranks,d,i...) = _pallocate(d,i...)

function _pallocate(d,i...)
  @abstractmethod
end

function _pallocate(
  d::AbstractArray{<:AbstractVector},
  r::AbstractVector{<:AbstractVector}
  )

  PVector(d,r)
end

function _pallocate(
  d::AbstractArray{<:AbstractArray},
  r::AbstractVector{<:AbstractVector}
  )

  GenericPArray(d,r)
end

function _pallocate(
  d::AbstractArray{<:AbstractMatrix},
  r::AbstractVector{<:AbstractVector},
  c::AbstractVector{<:AbstractVector}
  )

  PSparseMatrix(d,r,c)
end

function _sallocate(d,i...)
  @abstractmethod
end

function _sallocate(
  d::AbstractVector{<:AbstractArray},
  r::AbstractVector{<:AbstractLocalIndices}
  )

  n = maximum(maximum,map(own_to_global,r))
  T = eltype(eltype(d))
  tail = size(first(d))[2:end]
  v = zeros(T,n,tail...)
  for (dl,rl) in zip(d,r)
    o2l = own_to_local(rl)
    o2g = own_to_global(rl)
    selectdim(v,1,o2g) .= selectdim(dl,1,o2l)
  end
  v
end

function _sallocate(
  d::AbstractVector{<:AbstractMatrix},
  r::AbstractVector{<:AbstractLocalIndices},
  c::AbstractVector{<:AbstractLocalIndices}
  )

  nr = maximum(maximum,map(own_to_global,r))
  nc = maximum(maximum,map(own_to_global,c))
  T = eltype(eltype(d))
  I,J,V = Int[],Int[],T[]
  for (dl,rl,cl) in zip(d,r,c)
    o2l_r = own_to_local(rl)
    o2g_r = own_to_global(rl)
    l2g_c = local_to_global(cl)
    is,js,vs = findnz(sparse(dl[o2l_r,:]))
    append!(I,o2g_r[is])
    append!(J,l2g_c[js])
    append!(V,vs)
  end
  sparse(I,J,V,nr,nc)
end

function _load_distributed_hrprojection(dir,ranks;label="")
  basis,style,interps = map(ranks) do p
    a = deserialize(_part_filename(dir,HRPROJECTION_LABEL,label,p))
    get_basis(a),get_style(a),get_interpolation(a)
  end |> tuple_of_arrays
  i = getany(interps)
  interp = if i isa TransientInterpolation
    tstyle = get_interpolation_style(i)
    indices_time = get_indices_time(i)
    interp_spaces = map(i -> i.interp_space,interps)
    TransientInterpolation(tstyle,DistributedInterpolation(interp_spaces),indices_time)
  else
    DistributedInterpolation(interps)
  end
  HRProjection(getany(basis),getany(style),interp)
end

function _load_distributed_hr(dir,ranks;label="")
  if _haspart(dir,HRPROJECTION_LABEL,ranks;label=_plabel(label,BLOCK_LABEL*"1"))
    nblocks = 0
    while _haspart(dir,HRPROJECTION_LABEL,ranks;label=_plabel(label,BLOCK_LABEL*"$(nblocks+1)"))
      nblocks += 1
    end
    array = map(1:nblocks) do i
      _load_distributed_hrprojection(dir,ranks;label=_plabel(label,BLOCK_LABEL*"$i"))
    end
    return BlockHRProjection(array)
  end
  _load_distributed_hrprojection(dir,ranks;label)
end