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
# ~\~ begin <<README.md|value>>[4]
function Base.tanh(v::Value{T}) where T
    Value{T}(tanh(v.value), zero(T), [v], :tanh, nothing)
end
# ~\~ end
# ~\~ begin <<README.md|this-and-others>>[init]
function this_and_others(v :: Vector{T}) where T
    Channel() do chan
        for (idx, x) in enumerate(v)
            put!(chan, (x, [v[1:idx-1];v[idx+1:end]]))
        end
    end
end
# ~\~ end
# ~\~ begin <<README.md|topo-sort>>[init]
function topo_sort(tree, children = t -> t.children)
    visited = [tree]
    stack = [tree]
    while !isempty(stack)
        t = pop!(stack)
        for c in children(t)
            if c ∉ visited
                push!(stack, c)
                push!(visited, c)
            end
        end
    end
    visited
end
# ~\~ end
# ~\~ begin <<README.md|backpropagate>>[init]
const derivatives = IdDict(
    :* => (_, others) -> reduce(*, others),
    :+ => (_, _) -> 1.0,
    # ~\~ begin <<README.md|derivatives>>[init]
    :tanh => (value, _) -> Base.Math.sech(value)^2
    # ~\~ end
)
# ~\~ end
# ~\~ begin <<README.md|backpropagate>>[1]
function backpropagate(v :: Value{T}) where T
    v.grad = one(T)
    for n in topo_sort(v)
        for (c, others) in this_and_others(n.children)
            c.grad += n.grad * derivatives[n.operator](c.value, map(x -> x.value, others))
        end
    end
end
# ~\~ end

function main()
    # ~\~ begin <<README.md|example-1>>[init]
    a = literal(2.0) |> label("a")
    b = literal(3.0) |> label("b")
    c = literal(10.0) |> label("c")
    d = a * b + c * a |> label("d")
    # ~\~ end
    @printf "%s = %f\n" d.label d.value
    backpropagate(d)
    @printf "∂_%s d = %f\n" a.label a.grad
    print(collect(topo_sort(d)) .|> x -> x.label)
end

main()
# ~\~ end
