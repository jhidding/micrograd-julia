# ~\~ language=Julia filename=src/viz_example1.jl
# ~\~ begin <<README.md|src/viz_example1.jl>>[init]
using Printf: @sprintf
include("Graphviz.jl")
using .Graphviz: Graph, digraph, add_node, add_edge, add_attr, c_graph

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
    ((a.elems[it], vcat(a.elems[1:it-1], a.elems[it+1:end])), it)
end

Base.length(a :: ThisAndOthers{T}) where T = length(a.elems)

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
# ~\~ begin <<README.md|visualize>>[init]
function visualize(
        v::Value{T},
        g::Union{Graph,Nothing}=nothing,
        done::Union{Set{Value{T}},Nothing}=nothing) where T
    if isnothing(g)
        g = digraph() |> add_attr(c_graph; rankdir="LR")
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
    # ~\~ begin <<README.md|example-1>>[init]
    a = literal(2.0) |> label("a")
    b = literal(3.0) |> label("b")
    c = literal(10.0) |> label("c")
    d = a * b + c * a |> label("d")
    # ~\~ end
    backpropagate(d)
    print(visualize(d))
end

main()
# ~\~ end
