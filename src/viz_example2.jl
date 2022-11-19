# ~\~ language=Julia filename=src/viz_example2.jl
# ~\~ begin <<README.md|src/viz_example2.jl>>[init]
using Printf: @sprintf
using MLStyle: @match
include("Graphviz.jl")
using .Graphviz: Graph, digraph, add_node, add_edge, add_attr

# ~\~ begin <<README.md|value>>[init]
mutable struct Value{T}
    value :: T
    operator :: Union{Symbol,Expr}
    children :: Vector{Value{T}}
    grad :: T
    label :: Union{String, Nothing}
end

Value{T}(value::T, operator::Union{Expr,Symbol}, children::Vector{Value{T}}) where T = 
    Value(value, operator, children, zero(T), nothing)
# ~\~ end
# ~\~ begin <<README.md|value>>[1]
function Base.:+(a :: Value{T}, b :: Value{T}) where T
    Value{T}(a.value + b.value, :+, [(a.operator == :+ ? a.children : a);
                                     (b.operator == :+ ? b.children : b)])
end

function Base.:*(a :: Value{T}, b :: Value{T}) where T
    Value{T}(a.value * b.value, :*, [(a.operator == :* ? a.children : a);
                                     (b.operator == :* ? b.children : b)])
end
# ~\~ end
# ~\~ begin <<README.md|value>>[2]
function literal(value :: T) where T
    Value{T}(value, :const, Value{T}[])
end
# ~\~ end
# ~\~ begin <<README.md|value>>[3]
function label(l :: String)
    v -> begin v.label = l; v end
end
# ~\~ end
# ~\~ begin <<README.md|value>>[4]
function Base.tanh(v::Value{T}) where T
    Value{T}(tanh(v.value), :tanh, [v])
end
# ~\~ end
# ~\~ begin <<README.md|value>>[5]
function Base.convert(::Type{Value{T}}, x :: T) where T
   literal(x)
end

function vmap(f, operator::Symbol, value::Value{T}) where T
    Value{T}(f(value.value), operator, [value])
end

Base.:*(s::U, a::Value{T}) where {T, U <: Number} = convert(Value{T}, convert(T,s)) * a
Base.inv(a::Value{T}) where T = vmap(inv, :inv, a)
Base.:/(a::Value{T}, b::Value{T}) where T = a * inv(b)
Base.exp(a::Value{T}) where T = vmap(exp, :exp, a)
negate(a::Value{T}) where T = vmap(-, :negate, a)
Base.:-(a::Value{T}) where T = negate(a)
Base.:-(a::Value{T}, b::Value{T}) where T = a + negate(b)
Base.:-(a::Value{T}, b::U) where {T, U <: Number} = a - literal(convert(T,b))
Base.:+(a::Value{T}, b::U) where {T, U <: Number} = a + literal(convert(T,b))
Base.:^(a::Value{T}, b::U) where {T, U <: Integer} = Value{T}(a.value^b, :(x^$(b)), [a])
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
function topo_sort(node, children = n -> n.children, visited = nothing)
    visited = isnothing(visited) ? [] : visited
    Channel() do chan
        if node ∉ visited
            push!(visited, node)
            for c in children(node)
                foreach(n->put!(chan, n), topo_sort(c, children, visited))
            end
            put!(chan, node)
        end
    end
end
# ~\~ end
# ~\~ begin <<README.md|backpropagate>>[init]
function derive(symb :: Symbol)
    derivatives[symb]
end

const derivatives = IdDict(
    :* => (_, others) -> reduce(*, others),
    :+ => (_, _) -> 1.0,
    # ~\~ begin <<README.md|derivatives>>[init]
    :tanh => (value, _) -> Base.Math.sech(value)^2,
    # ~\~ end
    # ~\~ begin <<README.md|derivatives>>[1]
    :inv => (x, _) -> -1/x^2,
    :log => (x, _) -> 1/x,
    :exp => (x, _) -> exp(x),
    :negate => (_, _) -> -1.0,
    # ~\~ end
)
# ~\~ end
# ~\~ begin <<README.md|backpropagate>>[1]
# ~\~ begin <<README.md|this-and-others>>[init]
function this_and_others(v :: Vector{T}) where T
    Channel() do chan
        for (idx, x) in enumerate(v)
            put!(chan, (x, [v[1:idx-1];v[idx+1:end]]))
        end
    end
end
# ~\~ end

function backpropagate(v :: Value{T}) where T
    v.grad = one(T)
    for n in Iterators.reverse(collect(topo_sort(v)))
        for (c, others) in this_and_others(n.children)
            c.grad += n.grad * derive(n.operator)(c.value, map(x -> x.value, others))
        end
    end
end
# ~\~ end
# ~\~ begin <<README.md|backpropagate>>[2]
function derive(expr :: Expr)
    (x, _) -> 2 * x
    #    @match expr begin
#        Expr(:call, :^, :x, n) => (x, _) -> n * x^(n-1)
#        Expr(expr_type, _...)  => error("Unknown expression $(expr) of type $(expr_type)")
#    end
end
# ~\~ end
# ~\~ begin <<README.md|visualize>>[init]
function visualize(v::Value{T}) where T
    g = digraph(; rankdir="LR")
    for n in topo_sort(v)
        objid = repr(hash(n))
        objlabel = (isnothing(n.label) ? "" : n.label)
        reclabel = @sprintf "{ %s | data: %0.2f | grad: %0.2f }"  objlabel n.value n.grad
        g |> add_node("dat_" * objid; shape="record", label=reclabel)
        if (n.operator !== :const)
            g |> add_node("op_" * objid; label=String(n.operator)) |>
                 add_edge("op_" * objid, "dat_" * objid)
        end
        for c in n.children
            childid = repr(hash(c))
            g |> add_edge("dat_" * childid, (n.operator !== :const ? "op_" : "dat_") * objid)
        end
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
