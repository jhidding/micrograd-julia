# ~\~ language=Julia filename=src/example1.jl
# ~\~ begin <<README.md|src/example1.jl>>[init]
include("MicroGrad.jl")
using .MicroGrad: literal, label, backpropagate

function main()
    # ~\~ begin <<README.md|example-1>>[init]
    a = literal(2.0) |> label("a")
    b = literal(3.0) |> label("b")
    c = literal(10.0) |> label("c")
    d = a * b + c * a |> label("d")
    println("$(d.label) = $(d.value)")
    backpropagate(d)
    println("∂_$(a.label) d = $(a.grad)")
    # ~\~ end
end
main()
# ~\~ end
