# ~\~ language=Julia filename=src/viz_example2.jl
# ~\~ begin <<README.md|src/viz_example2.jl>>[init]
using Printf: @sprintf
include("Graphviz.jl")
using .Graphviz: Graph, digraph, add_node, add_edge, add_attr

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
# ~\~ begin <<README.md|visualize>>[init]
function visualize(
        v::Value{T},
        g::Union{Graph,Nothing}=nothing,
        done::Union{Set{Value{T}},Nothing}=nothing) where T
    if isnothing(g)
        g = digraph(; rankdir="LR")
    end
    if isnothing(done)
        done = Set([v])
    end
    objid = repr(hash(v))
    g |> add_node("dat_" * objid; shape="record",
        label=(@sprintf "{ %s | data: %0.2f | grad: %0.2f }" (isnothing(v.label) ? "" : v.label) v.value v.grad))
    if (v.operator !== :const)
        g |> add_node("op_" * objid; label=String(v.operator)) |>
             add_edge("op_" * objid, "dat_" * objid)
    end
    for c in v.children
        childid = repr(hash(c))
        if !(c in done)
            visualize(c, g, done)
        end
        g |> add_edge("dat_" * childid, (v.operator !== :const ? "op_" : "dat_") * objid)
    end
    g
end
# ~\~ end

function main()
    # ~\~ begin <<README.md|example-2>>[init]
    x1 = literal(2.0) |> label("x1")
    x2 = literal(0.0) |> label("x2")
    w1 = literal(-3.0) |> label("w1")
    w2 = literal(1.0) |> label("w2")
    b = literal(6.8813735870195432) |> label("b")
    n = x1*w1 + x2*w2 + b |> label("n")
    o = tanh(n) |> label("out")
    backpropagate(o)
    # ~\~ end
    print(visualize(o))
end

main()
# ~\~ end
