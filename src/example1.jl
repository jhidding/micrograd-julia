# ~\~ language=Julia filename=src/example1.jl
# ~\~ begin <<README.md|src/example1.jl>>[init]
using Printf: @printf

# ~\~ begin <<README.md|value>>[init]
mutable struct Value{T}
    value :: T
    grad :: T
    children :: Vector{Value}
    operator :: Symbol
    label :: Union{String, Nothing}
end
# ~\~ end
# ~\~ begin <<README.md|value>>[1]
function Base.:+(a :: Value{T}, b :: Value{T}) where T
    Value{T}(a.value + b.value, zero(T), [a, b], :+, nothing)
end

function Base.:*(a :: Value{T}, b :: Value{T}) where T
    Value{T}(a.value * b.value, zero(T), [a, b], :*, nothing)
end
# ~\~ end
# ~\~ begin <<README.md|value>>[2]
function literal(value :: T) where T
    Value{T}(value, zero(T), [], :const, nothing)
end
# ~\~ end
# ~\~ begin <<README.md|value>>[3]
function label(l :: String)
    v -> begin v.label = l; v end
end
# ~\~ end
# ~\~ begin <<README.md|this-and-others>>[init]
struct ThisAndOthers{T}
    elems :: Vector{T}
end

function Base.iterate(a :: ThisAndOthers{T}) where T
    if isempty(a.elems)
        return nothing
    end
    ((a.elems[1], a.elems[2:end]), 1)
end

function Base.iterate(a :: ThisAndOthers{T}, it) where T
    if length(a.elems) == it
        return nothing
    end
    it += 1
    ((a.elems[it], vcat(a.elems[1:it], a.elems[it+1:end])), it)
end

function this_and_others(v :: Vector{T}) where T
    ThisAndOthers(v)
end
# ~\~ end
# ~\~ begin <<README.md|backpropagate>>[init]
const derivatives = IdDict(
    :* => (grad, others) -> reduce(*, others; init = grad),
    :+ => (grad, _) -> grad
)
# ~\~ end
# ~\~ begin <<README.md|backpropagate>>[1]
function backpropagate(v :: Value{T}) where T
    v.grad = one(T)
    stack = [v]
    while !isempty(stack)
        local v = pop!(stack)
        for (c, others) in this_and_others(v.children)
            c.grad += derivatives[v.operator](v.grad, map(x -> x.value, others))
        end
        append!(stack, v.children)
    end
end
# ~\~ end

function main()
    a = literal(2.0) |> label("a")
    b = literal(3.0) |> label("b")
    c = literal(10.0) |> label("c")
    e = a * b |> label("e")
    d = e + c |> label("d")
    d1 = d.value
    @printf "%s = %f\n" d.label d.value
    backpropagate(d)
    @printf "%s' = %f\n" a.label a.grad
    a.value = 2.01
    e.value = a.value * b.value
    d.value = e.value + c.value
    @printf "numerical %s' = %f\n" a.label ((d.value - d1) / 0.01)
end

main()
# ~\~ end
