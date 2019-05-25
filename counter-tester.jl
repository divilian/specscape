include("max-num-generator.jl")

cc = Channel(producer);

for i in 1:10
    println(take!(cc))
end
