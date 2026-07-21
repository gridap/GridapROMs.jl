function Algebra.residual!(
  b::HRParamArray,
  op::GenericRBOperator{O,T,A,<:NNContribution},
  r::Realisation,
  u::AbstractVector,
  paramcache
  ) where {O,T,A}

  fill!(b.hypred,zero(eltype(b.hypred)))
  interpolate!(b,op.rhs,r)
end

function Algebra.jacobian!(
  A::HRParamArray,
  op::GenericRBOperator{O,T,<:NNContribution,B},
  r::Realisation,
  u::AbstractVector,
  paramcache
  ) where {O,T,B}

  fill!(A.hypred,zero(eltype(A.hypred)))
  interpolate!(A,op.lhs,r)
end