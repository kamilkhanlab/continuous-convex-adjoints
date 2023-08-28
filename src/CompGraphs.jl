#=
module CompGraphs
=================
The CompGraph type in this module is intended to hold the computational graph of a 
single composite function with vector-valued inputs and outputs. This graph 
expresses the composite function as a recipe of elemental operations. Computational 
graphs are required for implementing certain numerical methods, such as: 

- the standard reverse/adjoint mode of automatic differentiation (AD)

- the "branch-locking" method for efficient reverse-AD-like generalized
  differentiation by Khan (2018)
  https://doi.org/10.1080/10556788.2017.1341506

- the "cone-squashing" method for generalized differentiation 
  by Khan and Barton (2012, 2013)
  https://doi.org/10.1145/2491491.2491493

- the "reverse McCormick" convex relaxations incorporating constraint propagation, 
  by Wechsung et al. (2015)
  https://doi.org/10.1007/s10898-015-0303-6

"load_function!" in this module uses operator overloading to construct the 
computational graph of a finite composition of supported operations. Each 
node and the graph overall can also hold additional user-specified data, 
intended for use in methods like the reverse AD mode.

Written by Kamil Khan on February 10, 2022
=#

##############################
# Last modified by Yulan Zhang on May 12, 2023
# Inlcude parameters into this graph computation
##############################

module CompGraphs

using Printf

export CompGraph, GraphNode, CompGraphList

export load_function!, is_function_loaded

# structs; T and P are for smuggling in any sort of application-dependent data,
# but could easily be Any or Nothing for simplicity.

struct GraphNode{P}
    operation::Symbol
    parentIndices::Vector{Int}       # identify operands of "operation"
    constValue::Union{Float64, Int}  # only used when "operation" = :const or :^
    data::P                          # hold extra node-specific data
end

struct CompGraph{T, P}
    nodeList::Vector{GraphNode{P}}
    domainDim::Int              # domain dimension of graphed function
    parmDim::Int                # parameter dimension of graphed function 
    rangeDim::Int               # range dimension of graphed function 
    index::Int                  # index of function
    data::T                     # hold extra data not specific to any one node
end

struct CompGraphList{T, P}
    GraphList::Vector{CompGraph{T,P}}               
end

struct GraphBuilder{T,P}
    index::Int
    graph::CompGraph{T,P}
end

struct GraphBuilder_p{T,P}
    index::Int
    graph::CompGraph{T,P}
end

# constructors
GraphNode{P}(op::Symbol, i::Vector{Int}, p::P) where P = GraphNode{P}(op, i, 0.0, p)

# the following constructor requires a constructor T() with no arguments
function CompGraph{T, P}(n::Int, m::Int, k::Int, t::Int) where {T, P}
    return CompGraph{T, P}(GraphNode{P}[], n, m, k, t, T())
end

function CompGraphList{T, P}() where {T, P}
    return CompGraphList{T, P}(CompGraph{T,P}[])
end

# A GraphNode.operation can be any Symbol from the following lists
unaryOpList = [:-, :inv, :exp, :log, :sin, :cos, :abs]
binaryOpList = [:+, :-, :*, :/, :^, :max, :min, :hypot]
customOpList = [:input, :output, :parm, :const]   # input, output, parameter and Float64 constant

# print graph or individual nodes
opStringDict = Dict(
    :- => "neg",
    :inv => "inv",
    :exp => "exp",
    :log => "log",
    :sin => "sin",
    :cos => "cos",
    :abs => "abs",
    :+ => " + ",
    :- => " - ",
    :* => " * ",
    :/ => " / ",
    :^ => " ^ ",
    :max => "max",
    :min => "min",
    :hypot => "hyp",
    :input => "inp",
    :output => "out",
    :parm => " p ",
    :const => "con",
)

function Base.show(io::IO, node::GraphNode)
    parents = node.parentIndices
    nParents = length(parents)

    if (node.operation == :^) && (length(parents) == 1)
        opString = @sprintf " ^%1d" node.constValue
    else
        opString = opStringDict[node.operation]
    end

    if nParents <= 2
        oneParent(i::Int) = (nParents < i) ? "   " : @sprintf "%-3d" parents[i]
        parentString = oneParent(1) * "  " * oneParent(2)
    else
        parentString = string(parents)
    end
    
    return print(io, opString, " | ", parentString)
end


function Base.show(io::IO, graphList::CompGraphList)
    return begin
        for (j, graph) in enumerate(graphList.GraphList)
            println(io, " Computational graph of function No." * string(j) * "\n")
            println(" index | op  | parents")
            println(" ---------------------")
            for (i, node) in enumerate(graph.nodeList)
                @printf "   %3d | " i
                println(node)
            end
            println("\n")
        end
    end
end


# load in a function using operator overloading, and store its computational graph
function load_function!(
    f::Function,
    graph::CompGraph{T, P},
    initP::P
) where {T, P}
    
    empty!(graph.nodeList)

    # push new nodes for function inputs
    xGB = [GraphBuilder{T,P}(i,graph) for i=1:(graph.domainDim)]
    for xComp in xGB
        inputData = deepcopy(initP)
        inputNode = GraphNode{P}(:input, Int[], 0.0, inputData)
        push!(graph.nodeList, inputNode)
    end
    
    # push new nodes for function parameters
    pGB = [GraphBuilder_p{T,P}(i,graph) for i=1:(graph.parmDim)]
    for pComp in pGB
        parmData = deepcopy(initP)
        parmNode = GraphNode{P}(:parm, Int[], 0.0, parmData)
        push!(graph.nodeList, parmNode)
    end

    # push new nodes for all intermediate operations, using operator overloading
    yGB_i = graph.index
    yGB = f(xGB, pGB, yGB_i)
    if !(yGB isa Vector)
        yGB = [yGB]
    end

    # push new nodes for function outputs
    for yComp in yGB
        outputData = deepcopy(initP)
        outputNode = GraphNode{P}(:output, [yComp.index], 0.0, outputData)
        push!(graph.nodeList, outputNode)
    end
end

is_function_loaded(graphList::CompGraphList) = !isempty(graphList.GraphList)

## let GraphBuilder construct a nodeList by operator overloading

# overload unary operations in unaryOpList
macro define_GraphBuilder_unary_op_rule(op)
    opEval = eval(op)
    return quote
        function Base.$opEval(u::GraphBuilder{T,P}) where {T,P}
            parentGraph = u.graph
            prevNodes = parentGraph.nodeList
            newNodeData = deepcopy(prevNodes[u.index].data)
            newNode = GraphNode{P}($op, [u.index], 0.0, newNodeData)
            push!(prevNodes, newNode)                
            return GraphBuilder{T,P}(length(prevNodes), parentGraph)
        end

        function Base.$opEval(u::GraphBuilder_p{T,P}) where {T,P}
            parentGraph = u.graph
            prevNodes = parentGraph.nodeList
            newNodeData = deepcopy(prevNodes[u.index + parentGraph.domainDim].data)
            newNode = GraphNode{P}($op, [u.index + parentGraph.domainDim], 0.0, newNodeData)
            push!(prevNodes, newNode)                
            return GraphBuilder{T,P}(length(prevNodes), parentGraph)
        end

    end
end

for i in 1:length(unaryOpList)
    @eval @define_GraphBuilder_unary_op_rule $unaryOpList[$i]
end


# overload binary operations in binaryOpList.
# Uses nontrivial "promote" rules to push Float64 constants to the parent graph
# Constvalue
function Base.promote(uA::GraphBuilder{T,P}, uB::Float64) where {T,P}
    parentGraph = uA.graph
    prevNodes = parentGraph.nodeList

    newNodeData = deepcopy(prevNodes[uA.index].data)
    newNode = GraphNode{P}(:const, Int[], uB, newNodeData)
    push!(prevNodes, newNode)

    return (uA, GraphBuilder{T,P}(length(prevNodes), parentGraph))
end
function Base.promote(uA::Float64, uB::GraphBuilder{T,P}) where {T,P}
    return reverse(promote(uB, uA))
end

function Base.promote(uA::GraphBuilder_p{T,P}, uB::Float64) where {T,P}
    parentGraph = uA.graph
    prevNodes = parentGraph.nodeList

    newNodeData = deepcopy(prevNodes[uA.index].data)
    newNode = GraphNode{P}(:const, Int[], uB, newNodeData)
    push!(prevNodes, newNode)

    return (uA, GraphBuilder{T,P}(length(prevNodes), parentGraph))
end
function Base.promote(uA::Float64, uB::GraphBuilder_p{T,P}) where {T,P}
    return reverse(promote(uB, uA))
end

# Parameter
Base.promote(uA::GraphBuilder{T,P}, uB::GraphBuilder{T,P}) where {T,P} = (uA, uB)
Base.promote(uA::GraphBuilder{T,P}, uB::GraphBuilder_p{T,P}) where {T,P} = (uA, uB)
Base.promote(uA::GraphBuilder_p{T,P}, uB::GraphBuilder{T,P}) where {T,P} = (uA, uB)
Base.promote(uA::GraphBuilder_p{T,P}, uB::GraphBuilder_p{T,P}) where {T,P} = (uA, uB)


macro define_GraphBuilder_binary_op_rule(op)
    opEval = eval(op)
    return quote
        function Base.$opEval(uA::GraphBuilder{T,P}, uB::GraphBuilder{T,P}) where {T,P}
            parentGraph = uA.graph
            prevNodes = parentGraph.nodeList
            
            newNodeData = deepcopy(prevNodes[uA.index].data)
            newNode = GraphNode{P}($op, [uA.index, uB.index], 0.0, newNodeData)
            push!(prevNodes, newNode)
            
            return GraphBuilder{T,P}(length(prevNodes), parentGraph)
        end
        
        function Base.$opEval(uA::GraphBuilder{T,P}, uB::GraphBuilder_p{T,P}) where {T,P}
            parentGraph = uA.graph
            prevNodes = parentGraph.nodeList
            
            newNodeData = deepcopy(prevNodes[uA.index].data)
            newNode = GraphNode{P}($op, [uA.index, uB.index + parentGraph.domainDim], 0.0, newNodeData)
            push!(prevNodes, newNode)
            
            return GraphBuilder{T,P}(length(prevNodes), parentGraph)
        end
        
        function Base.$opEval(uA::GraphBuilder_p{T,P}, uB::GraphBuilder{T,P}) where {T,P}
            parentGraph = uA.graph
            prevNodes = parentGraph.nodeList
            
            newNodeData = deepcopy(prevNodes[uA.index].data)
            newNode = GraphNode{P}($op, [uA.index + parentGraph.domainDim, uB.index], 0.0, newNodeData)
            push!(prevNodes, newNode)
            
            return GraphBuilder{T,P}(length(prevNodes), parentGraph)
        end
        
        function Base.$opEval(uA::GraphBuilder_p{T,P}, uB::GraphBuilder_p{T,P}) where {T,P}
            parentGraph = uA.graph
            prevNodes = parentGraph.nodeList
            
            newNodeData = deepcopy(prevNodes[uA.index].data)
            newNode = GraphNode{P}($op, [uA.index + parentGraph.domainDim, uB.index + parentGraph.domainDim], 0.0, newNodeData)
            push!(prevNodes, newNode)
            
            return GraphBuilder{T,P}(length(prevNodes), parentGraph)
        end

        function Base.$opEval(uA::GraphBuilder{T,P}, uB::Float64) where {T,P}
            return $opEval(promote(uA, uB)...)
        end

        function Base.$opEval(uA::Float64, uB::GraphBuilder{T,P}) where {T,P}
            return $opEval(promote(uA, uB)...)
        end
        
        function Base.$opEval(uA::GraphBuilder_p{T,P}, uB::Float64) where {T,P}
            return $opEval(promote(uA, uB)...)
        end

        function Base.$opEval(uA::Float64, uB::GraphBuilder_p{T,P}) where {T,P}
            return $opEval(promote(uA, uB)...)
        end
    end
end


for i in 1:length(binaryOpList)
    @eval @define_GraphBuilder_binary_op_rule $binaryOpList[$i]
end

# constant integer exponents are supported
function Base.:^(uA::GraphBuilder{T,P}, uB::Int) where {T,P}
    parentGraph = uA.graph
    prevNodes = parentGraph.nodeList
    
    newNodeData = deepcopy(prevNodes[uA.index].data)
    newNode = GraphNode{P}(:^, [uA.index], uB, newNodeData)
    push!(prevNodes, newNode)
    
    return GraphBuilder{T,P}(length(prevNodes), parentGraph)
end

function Base.:^(uA::GraphBuilder_p{T,P}, uB::Int) where {T,P}
    parentGraph = uA.graph
    prevNodes = parentGraph.nodeList
    
    newNodeData = deepcopy(prevNodes[uA.index].data)
    newNode = GraphNode{P}(:^, [uA.index + parentGraph.domainDim], uB, newNodeData)
    push!(prevNodes, newNode)
    
    return GraphBuilder{T,P}(length(prevNodes), parentGraph)
end

end # module

