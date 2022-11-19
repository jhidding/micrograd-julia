# ~\~ language=Julia filename=src/viz_example1.jl
# ~\~ begin <<README.md|src/viz_example1.jl>>[init]
# ~\~ begin <<README.md|prelude>>[init]
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
# ~\~ end

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
    backpropagate(d)
    print(visualize(d))
end

main()
# ~\~ end
