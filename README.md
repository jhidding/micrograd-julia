# Micrograd in Julia
A literate Julia translation of Andrej Karpathy's `micrograd`, following his video lecture.

## Derivatives
The goal of this exercise is to compute derivatives across a neural network. The idea is that we compute a value of some very complicated function in a forward pass, and then, traversing backward through the tree, we can infer (cheaply) the gradient of the output with respect to input variables.

We start by learning about derivatives, usually defined as the rate of increment in a function in the limit where some step-size goes to zero:

$$\partial_x f(x) := \lim_{h \to 0} {f(x+h) - f(x) \over h}.$$

For functions of a single parameter, we may also write $\partial_x f(x) = f'(x)$, allowing us to sometimes drop the gratuitous $x$ from our notation (it's a dummy free variable). From this definition we can compute the derivative of composite functions analytically.

### Sum rule
Given $f = u + v$, we may see from the linearity in the definition of the derivative that,

:::result
$$(u+v)' = u' + v'.$$
:::

### Product rule
Given $f = uv$, things get a bit more complicated. First of all, we need to see that inside the limit we can write,

$$\lim_{h \to 0} \Big[f'(x) = {f(x+h) - f(x) \over h}\Big],$$

therefore,
$$\lim_{h \to 0} \Big[f(x+h) = f(x) + h f'(x)\Big].$$

Then if we follow the definition and write out $f'$ for $f=uv$,

$$f'(x) = \lim_{h \to 0} {u(x+h)v(x+h) - u(x)v(x) \over h},$$

and taking $v(x+h) \approx v(x) + h v'(x)$,

$$\begin{align}f'(x) &= \lim_{h \to 0} {u(x+h) (v(x) + h v'(x)) - u(x) v(x) \over h}\\
                     &= \lim_{h \to 0} {u(x+h) v(x) + u(x+h) h v'(x) - u(x) v(x) \over h}\\
                     &= \lim_{h \to 0} \Big[ v(x) {u(x+h) - u(x) \over h} + u(x+h) v'(x) \Big]\\
                     &= v(x)u'(x) + u(x) v'(x).\end{align}$$

In the last step, using that in the limit $u(x+h) = u(x)$. In short,

:::result
$$(uv)' = u'v + uv'.$$
:::

Ok, with that out of the way, we can implement the first tiny version of an automatic differentating back propagation.

## Computation
We define the a data structure that traces a computation.

``` {.julia #value}
mutable struct Value{T}
    value :: T
    grad :: T
    children :: Vector{Value}
    operator :: Symbol
    label :: Union{String, Nothing}
end
```

Now we add methods to perform addition and multiplication on `Value`s.

``` {.julia #value}
function Base.:+(a :: Value{T}, b :: Value{T}) where T
    Value{T}(a.value + b.value, zero(T), [a, b], :+, nothing)
end

function Base.:*(a :: Value{T}, b :: Value{T}) where T
    Value{T}(a.value * b.value, zero(T), [a, b], :*, nothing)
end
```

To create a literal value, say an input:

``` {.julia #value}
function literal(value :: T) where T
    Value{T}(value, zero(T), [], :const, nothing)
end
```

To add a label to a value, we'll have a nice `|> label("x")` syntax.

``` {.julia #value}
function label(l :: String)
    v -> begin v.label = l; v end
end
```

### Special iterator: `this_and_others`
I supposed that, from some generality concerns, we could have combinators with more than two children. In that case, we'd like to iterate over each child, together with all their siblings (excluding the child). This is why I made an iterator that does just that `this_and_others`. Given a `Vector{T}` it yields pairs of an element and a vector containing the other elements.

``` {.julia #this-and-others}
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
```

### Derivatives
We previously derived the sum and product rules for differentiation. When written in this form, they become rather obvious. What was all the fuss about?

``` {.julia #backpropagate}
const derivatives = IdDict(
    :* => (grad, others) -> reduce(*, others; init = grad),
    :+ => (grad, _) -> grad
)
```

Now, it is a matter of walking the evaluation tree backward. I use a stack, pushing children and popping them off, until no children remain. This is the stuff of nightmares.

``` {.julia #backpropagate}
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
```

Note, that a value may be used in several subexpressions, creating a diamond dependency diagram. In such a case, we want to add all contributions from different branches. This is why we find `c.grad += ...` there.

### First example

``` {.julia #example-1}
a = literal(2.0) |> label("a")
b = literal(3.0) |> label("b")
c = literal(10.0) |> label("c")
d = a * b + c * a |> label("d")
```

``` {.julia file=src/example1.jl}
using Printf: @printf

<<value>>
<<this-and-others>>
<<backpropagate>>

function main()
    <<example-1>>
    @printf "%s = %f\n" d.label d.value
    backpropagate(d)
    @printf "∂_%s d = %f\n" a.label a.grad
end

main()
```

Giving

```
d = 26.000000
∂_a d = 13.000000
```

## Plotting tree in `graphviz`
Julia has a module for interaction with Graphviz, but it requires input in dot language, so this module is next to useless. We can do better.


``` {.julia #visualize}
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
```

``` {.julia file=src/viz_example1.jl}
using Printf: @sprintf
include("Graphviz.jl")
using .Graphviz: Graph, digraph, add_node, add_edge, add_attr, c_graph

<<value>>
<<this-and-others>>
<<backpropagate>>
<<visualize>>

function main()
    <<example-1>>
    backpropagate(d)
    print(visualize(d))
end

main()
```

``` {.make .figure target=fig/example1.svg}
$(target): src/viz_example1.jl
> julia $< | dot -Tsvg > $@
```

## Appendix: Graphviz module
For those interested, here's the source for `Graphviz.jl`.

``` {.julia file=src/Graphviz.jl}
module Graphviz

@enum GraphComponent c_graph c_node c_edge

const component_name = IdDict(
    c_graph => "graph",
    c_node => "node",
    c_edge => "edge"
)

Base.show(io :: IO, c :: GraphComponent) = print(io, component_name[c])

@enum CompassPt c_n c_ne c_e c_se c_s c_sw c_w c_nw c_c c_empty

const compass_symb = IdDict(
    c_n => "n", c_ne => "ne", c_e => "e", c_se => "se", c_s => "s",
    c_sw => "sw", c_w => "w", c_nw => "nw", c_c => "c", c_empty => "_"
)

Base.show(io :: IO, c :: CompassPt) = print(io, compass_symb[c])

abstract type Statement end

struct Subgraph <: Statement
    name :: Union{String, Nothing}
    stmt_lst :: Vector{Statement}
end

struct AList
    content :: IdDict{Symbol,String}
end

function Base.show(io :: IO, alst :: AList)
    for (k, v) in pairs(alst.content)
        print(io, k, "=\"", v, "\";")
    end
end

struct NodeId
    name :: String
    port :: Union{String,Nothing}
end

struct NodeStmt <: Statement
    id :: NodeId
    attr_list :: Vector{AList}
end

NodeOrSubgraph = Union{NodeId,Subgraph}

function Base.show(io :: IO, n :: NodeId)
    print(io, "\"", n.name, "\"")
    if !isnothing(n.port)
        print(io, n.port)
    end
end

function Base.show(io :: IO, n :: NodeStmt)
    print(io, n.id)
    for a in n.attr_list
        print(io, "[", a, "]")
    end
end

struct EdgeStmt <: Statement
    is_directed :: Bool
    from :: Union{NodeId,Subgraph}
    to :: Vector{Union{NodeId,Subgraph}}
    attr_list :: Vector{AList}
end

function Base.show(io :: IO, e :: EdgeStmt)
    print(io, e.from)
    for n in e.to
        print(io, e.is_directed ? "->" : "--", n)
    end
    for a in e.attr_list
        print(io, "[", a, "]")
    end
end

struct AttrStmt <: Statement
    component :: GraphComponent
    attr_list :: Vector{AList}
end

function Base.show(io :: IO, a :: AttrStmt)
    print(io, component_name[a.component])
    for attr in a.attr_list
        print(io, "[", attr, "]")
    end
end

struct IdentityStmt <: Statement
    first :: String
    second :: String
end

function Base.show(io :: IO, i :: IdentityStmt)
    print(io, i.first, "=\"", i.second, "\"")
end

function Base.show(io :: IO, s :: Subgraph)
    print(io, "subgraph ")
    if !isnothing(s.name)
        print(io, s.name, " ")
    end
    print(io, "{\n")
    for s in s.stmt_lst
        print(io, "  ", s, ";\n")
    end
    print(io, "}\n")
end

mutable struct Graph
    is_strict :: Bool
    is_directed :: Bool
    name :: Union{String, Nothing}
    stmt_list :: Vector{Statement}
end

function Base.show(io :: IO, g :: Graph)
    print(io, g.is_strict ? "strict " : "",
              g.is_directed ? "digraph " : "graph ",
              !isnothing(g.name) ? name * " " : "",
              "{\n")
    for s in g.stmt_list
        print(io, "  ", s, ";\n")
    end
    print(io, "}\n")
end

function graph(name = nothing)
    Graph(false, false, name, [])
end

function digraph(name = nothing)
    Graph(false, true , name, [])
end

strict(g :: Graph) = begin g.is_strict = True; g end

function add_node(id :: String, port :: Union{String, Nothing} = nothing; kwargs ...)
    g -> begin push!(g.stmt_list, NodeStmt(NodeId(id, port), [AList(kwargs)])); g end
end

function add_edge(from :: String, to :: String ...; kwargs ...)
    add_edge(NodeId(from, nothing), NodeOrSubgraph[NodeId(n, nothing) for n in to]; kwargs ...)
end

function add_edge(from::NodeOrSubgraph, to::Vector{NodeOrSubgraph}; kwargs ...)
    g -> begin push!(g.stmt_list, EdgeStmt(g.is_directed, from, to, [AList(kwargs)])); g end
end

function add_attr(comp :: GraphComponent; attrs ...)
    g -> begin push!(g.stmt_list, AttrStmt(comp,[AList(IdDict(attrs))])); g end
end

end
```
