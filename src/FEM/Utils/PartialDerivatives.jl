"""
    struct PartialDerivative{N} <: Function end

`Gridap` Map that implements a partial derivative
"""
struct PartialDerivative{N} <: Function end

PartialDerivative{N}(f) where N = Operation(Component{N}())(∇(f))

"""
    const ∂₁ = PartialDerivative{1}
"""
const ∂₁ = PartialDerivative{1}

"""
    const ∂₂ = PartialDerivative{2}
"""
const ∂₂ = PartialDerivative{2}

"""
    const ∂₃ = PartialDerivative{3}
"""
const ∂₃ = PartialDerivative{3}

const Divergence = Union{PartialDerivative{(1,2)},PartialDerivative{(1,2,3)}}

function Arrays.evaluate!(cache,::Broadcasting{<:PartialDerivative{N}},f) where N
  Broadcasting(Operation(Component{N}()))(Broadcasting(∇)(f))
end

function PartialDerivative{N}(f::Function,x::Point,fx) where N
  Component{N}(gradient(f,x,fx))
end

struct Component{N} <: Function end

(f::Component{N})(x...) where N = evaluate(f,x...)
(f::Component{N})(a::CellField) where N = Operation(f)(a)
(f::Component{N})(a::Field...) where N = Operation(f)(a...)

(f::Component{N})(v::MultiValue) where N = Component{N}(v)
Component{N}(v::MultiValue) where N = @notimplemented
Component{N}(v::MultiValue{Tuple{D}}) where {N,D} = v[N]
