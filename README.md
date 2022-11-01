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
    ((a.elems[it], vcat(a.elems[1:it], a.elems[it+1:end])), it)
end

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

``` {.julia file=src/example1.jl}
using Printf: @printf

<<value>>
<<this-and-others>>
<<backpropagate>>

function main()
    a = literal(2.0) |> label("a")
    b = literal(3.0) |> label("b")
    c = literal(10.0) |> label("c")
    e = a * b |> label("e")
    d = e + c |> label("d")
    d1 = d.value
    @printf "%s = %f\n" d.label d.value
    backpropagate(d)
    @printf "∂_%s d = %f\n" a.label a.grad
    a.value = 2.01
    e.value = a.value * b.value
    d.value = e.value + c.value
    @printf "numerical ∂_%s d = %f\n" a.label ((d.value - d1) / 0.01)
end

main()
```

Giving

```
d = 16.000000
∂_a d = 3.000000
numerical ∂_a d = 3.000000
```
