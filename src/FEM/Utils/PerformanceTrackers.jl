abstract type PerformanceTracker end

mutable struct CostTracker <: PerformanceTracker
  name::String
  time::Float64
  nallocs::Float64
  nruns::Int
end

get_name(c::CostTracker) = c.name
get_time(c::CostTracker) = c.time
get_nallocs(c::CostTracker) = c.nallocs
get_nruns(c::CostTracker) = c.nruns

function CostTracker(;name="",time=0.0,nallocs=0.0,nruns=1)
  CostTracker(name,time,nallocs,nruns)
end

function CostTracker(stats::NamedTuple;nruns=1,name="")
  time = stats[:time]
  nallocs = stats[:bytes] / 1e6
  CostTracker(name,time,nallocs,nruns)
end

function Base.show(io::IO,k::MIME"text/plain",t::CostTracker)
  show_mega = t.nallocs < 1e3
  println(io," -------------------------------------------------------------")
  println(io," > CostTracker($(t.name)) across $(t.nruns) runs:")
  println(io," > computational time (s): $(t.time)")
  if show_mega
    println(io," > memory footprint (Mb): $(t.nallocs)")
  else
    println(io," > memory footprint (Gb): $(t.nallocs/1e3)")
  end
  println(io," -------------------------------------------------------------")
end

function Base.show(io::IO,t::CostTracker)
  show(io,MIME"text/plain"(),t)
end

function mean(cs::AbstractVector{<:CostTracker})
  name = get_name(first(cs))
  @check all(name == get_name(c) for c in cs)
  time = sum(map(get_time,cs))
  nallocs = sum(map(get_nallocs,cs))
  nruns = sum(map(get_nruns,cs))
  CostTracker(name,time,nallocs,nruns)
end

function reset_tracker!(t::CostTracker)
  t.time = 0.0
  t.nallocs = 0.0
  t.nruns = 0
end

function update_tracker!(t::CostTracker,stats::NamedTuple;msg="")
  time = stats[:time]
  nallocs = stats[:bytes] / 1e6
  t.time += time
  t.nallocs += nallocs
  if !isempty(msg)
    println(msg)
    show(stdout,MIME"text/plain"(),t)
  end
end

function get_stats(t::CostTracker)
  avg_time = t.time / t.nruns
  avg_nallocs = t.nallocs / t.nruns
  return avg_time,avg_nallocs
end

struct Speedup <: PerformanceTracker
  name::String
  speedup_time::Float64
  speedup_memory::Float64
end

get_name(su::Speedup) = su.name
get_speedup_time(su::Speedup) = su.speedup_time
get_speedup_memory(su::Speedup) = su.speedup_memory

function Base.show(io::IO,k::MIME"text/plain",su::Speedup)
  println(io," -------------------- Speedup($(su.name)) -------------------------")
  println(io," > speedup in time: $(su.speedup_time)")
  println(io," > speedup in memory: $(su.speedup_memory)")
  println(io," -------------------------------------------------------------")
end

function Base.show(io::IO,su::Speedup)
  show(io,MIME"text/plain"(),su)
end

function mean(sus::AbstractVector{<:Speedup})
  name = get_name(first(sus))
  @check all(name == get_name(su) for su in sus)
  mean_sut = mean(map(get_speedup_time,sus))
  mean_sum = mean(map(get_speedup_memory,sus))
  Speedup(name,mean_sut,mean_sum)
end

"""
    compute_speedup(t1::CostTracker,t2::CostTracker) -> Speedup

Computes the speedup the tracker `t2` achieves with respect to `t1`, in time and
in memory footprint
"""
function compute_speedup(t1::CostTracker,t2::CostTracker)
  name = "$(t1.name) / $(t2.name)"
  avg_time1,avg_nallocs1 = get_stats(t1)
  avg_time2,avg_nallocs2 = get_stats(t2)
  speedup_time = avg_time1 / avg_time2
  speedup_memory = avg_nallocs1 / avg_nallocs2
  return Speedup(name,speedup_time,speedup_memory)
end

induced_norm(v::AbstractVector) = norm(v)
induced_norm(v::AbstractVector,norm_matrix::AbstractMatrix) = sqrt(v'*(norm_matrix*v))

induced_norm(A::AbstractMatrix) = sqrt(sum(diag(A'*A)))
induced_norm(A::AbstractMatrix,norm_matrix::AbstractMatrix) = sqrt(sum(diag(A'*(norm_matrix*A))))

"""
    compute_error(
      sol::AbstractArray{T,N},
      sol_approx::AbstractArray{T,N},
      args...
      ) where {T,N} -> Number

Computes the error between `sol` and `sol_approx`, by default in the Euclidean
norm. A different norm (usually represented by a sparse matrix) can be provided
as an argument.
"""
function compute_error(
  sol::AbstractArray{T,N},
  sol_approx::AbstractArray{T,N},
  args...
  ) where {T,N}

  @check size(sol) == size(sol_approx)
  errors = size(sol,N)
  @inbounds for i = 1:nruns
    soli = selectdim(sol,N,i)
    soli_approx = selectdim(sol_approx,N,i)
    errors[i] = compute_relative_error(soli,soli_approx,args...)
  end
  return mean(errors)
end

"""
    compute_relative_error(
      sol::AbstractArray{T,N},
      sol_approx::AbstractArray{T,N},
      args...
      ) where {T,N} -> Number

Computes the relative error between `sol` and `sol_approx`, by default in the Euclidean
norm. A different norm (usually represented by a sparse matrix) can be provided
as an argument.
"""
function compute_relative_error(sol::AbstractArray,sol_approx::AbstractArray,args...)
  err_norm = induced_norm(sol-sol_approx,args...)
  sol_norm = induced_norm(sol,args...)
  rel_norm = err_norm / sol_norm
  return rel_norm
end
