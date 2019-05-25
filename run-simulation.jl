include("Sugarscape.jl")
include("Agent.jl")
include("Institution.jl")
include("max-num-generator.jl")

using Statistics
using Random
using Distributions
using CSV
using DataFrames
# using RCall

function set_up_environment(scape_side, scape_carry_cap, scape_growth_rate,
                            pop_density, metab_range_tpl, vision_range_tpl, suglvl_range_tpl)
    """
    Arguments:
    scape_side
    scape_carry_cap
    scape_growth_rate
    pop_density
    metab_range_tpl
    vision_range_tpl
    suglvl_range_tpl

    Returns: dictionary {sugscape object =>, arr_agents => }
    """
    ## Generate an empty sugarscape
    sugscape_obj = generate_sugarscape(scape_side, scape_growth_rate, scape_carry_cap, 3);
    stats = get_sugarscape_stats(sugscape_obj);


    no_agents = Int(ceil(pop_density * scape_side^2));

    metabol_distrib =  DiscreteUniform(metab_range_tpl[1], metab_range_tpl[2]);
    vision_distrib = DiscreteUniform(vision_range_tpl[1], vision_range_tpl[2]);
    suglvl_distrib = DiscreteUniform(suglvl_range_tpl[1], suglvl_range_tpl[2]);

    arr_poss_locations = sample([(x,y) for x in 1:scape_side, y in 1:scape_side],
                                no_agents, replace=false)    

    arr_agents = [Agent(agg_id,
                        arr_poss_locations[agg_id][1],
                        arr_poss_locations[agg_id][2],
                        rand(vision_distrib),
                        rand(metabol_distrib),
                        rand(suglvl_distrib), true, -1)
                  for agg_id in 1:no_agents]

    ## mark as occupied the cells in sugarscape corresponding to the agents' locs
    for loc in arr_poss_locations
        sugscape_obj[loc[1], loc[2]].occupied = true
    end
    # println("Created a sugarscape of size: ", 
    #         string(size(sugscape_obj)[1] * size(sugscape_obj)[2]))
    # println("Created ", string(length(arr_agents)), " agents.")
    return(Dict("sugscape_obj" => sugscape_obj,
                "arr_agents" => arr_agents)) 
end ## end of set_up_environment()


function compute_Gini(collection_obj, type)
    # println("Computing gini for type: ", type)
    arr_suglevels = [singobj.sugar_level for singobj in
                     collection_obj if singobj.sugar_level >= 0] 
    # println(arr_suglevels)
    # println("Enter enter")
    # readline()
    # @assert(size(arr_suglevels)[1] > 0)
    # R"library(ineq)"
    # gini = R"ineq($arr_suglevels, type='Gini')"[1]
    # try
    #     @assert gini >= 0.0 && gini <= 1.0
    # catch
    #     # println("Came across a nonsensical Gini value: ", string(gini))
    #     # readline()
    # end
    # # println(gini)
    # return(gini)    
    n = size(arr_suglevels)[1]
    iss = 1:n
    if (n > 0)
        # sort(arr_suglevels, dims=1)
        g = (2 * sum(iss .* sort(arr_suglevels)))/(n * sum(arr_suglevels))
        return(g - ((n+1)/n))
    else
        return(NaN)
    end
end

function animate_sim(sugscape_obj, arr_agents, time_periods, 
                     birth_rate, inbound_rate, outbound_rate,
                     vision_range_tpl, metab_range_tpl, suglvl_range_tpl,
                     threshold)
    """
    Performs the various operations on the sugarscape and agent population
    to 'animate' them.
    Returns a single row, consisting of all of the params + gini values
    of sugar across all the time periods.    
    """
    metabol_distrib =  DiscreteUniform(metab_range_tpl[1], metab_range_tpl[2]);
    vision_distrib = DiscreteUniform(vision_range_tpl[1], vision_range_tpl[2]);
    suglvl_distrib = DiscreteUniform(suglvl_range_tpl[1], suglvl_range_tpl[2]); 

    arr_agent_ginis = zeros(time_periods)
    arr_scape_ginis = zeros(time_periods)

    @assert !(arr_scape_ginis === arr_agent_ginis)
    ## the following is a hack because creating an empty array of Array{Institution, 1}
    ## and adding Institution objects via push! is resulting in errors.
    ## So to add type-checking on arr_institutions, we're going to initialize it with
    ## a dummy Institution object
    # arr_institutions = Array{Institution, 1}
    arr_institutions = [Institution(-1, -1, false, [-1], [Transaction(-1, -1, "", -1)])]
    
    for period in 1:time_periods 
        for ind in shuffle(1:length(arr_agents))
            locate_move_feed!(arr_agents[ind], sugscape_obj, arr_agents, arr_institutions, period)
        end 
        regenerate_sugar!(sugscape_obj)
        perform_birth_inbound_outbound!(arr_agents, sugscape_obj, birth_rate, 
                                        inbound_rate, outbound_rate, 
                                        vision_distrib, metabol_distrib,
                                        suglvl_distrib) 
        form_possible_institutions!(arr_agents, threshold, sugscape_obj, 
                                    arr_institutions, period)

        arr_agents = life_check!(arr_agents)
        @assert all([aggobj.alive for aggobj in arr_agents])
        # println("HERHEREHERE")
        # readline()
        update_occupied_status!(arr_agents, sugscape_obj)
        update_institution_statuses!(arr_institutions, period)

        arr_agent_ginis[period] = compute_Gini(arr_agents)
        arr_scape_ginis[period] = compute_Gini(sugscape_obj)
        # println("No. of institutions: ", length(arr_institutions))
        # println("No. of agents: ", string(length(arr_agents)))
        # println("Finished time-step: ", string(period), "\n\n") 
        ## println("Enter enter")
        ## readline()
    end## end of time_periods for loop
    return((arr_agent_ginis, arr_scape_ginis))
end ## end animate_sim()

function run_sim(givenseed)
    Random.seed!(givenseed)
    params_df = CSV.read("parameter-ranges-testing.csv")
    outfile_name = "outputs-new.csv"
    time_periods = 100

    # temp_out = DataFrame(zeros(nrow(params_df), time_periods))
    # names!(temp_out, Symbol.(["prd_"*string(i) for i in 1:time_periods]))

    temp_out_agents = DataFrame(zeros(nrow(params_df), time_periods))
    names!(temp_out_agents, Symbol.(["prd_"*string(i) for i in 1:time_periods]))

    temp_out_scape = DataFrame(zeros(nrow(params_df), time_periods))
    names!(temp_out_scape, Symbol.(["prd_"*string(i) for i in 1:time_periods]))

    # out_df = DataFrame()
    out_df_agents = DataFrame()
    out_df_scape = DataFrame()
    # for colname in names(params_df)
    #     out_df[Symbol(colname)] = params_df[Symbol(colname)]
    # end

    # for colname in names(temp_out)
    #     out_df[Symbol(colname)] = temp_out[Symbol(colname)]
    # end

    for colname in names(params_df)
        out_df_agents[Symbol(colname)] = params_df[Symbol(colname)]
    end

    for colname in names(temp_out_agents)
        out_df_agents[Symbol(colname)] = temp_out_agents[Symbol(colname)]
    end

    for colname in names(params_df)
        out_df_scape[Symbol(colname)] = params_df[Symbol(colname)]
    end

    for colname in names(temp_out_scape)
        out_df_scape[Symbol(colname)] = temp_out_scape[Symbol(colname)]
    end
    
    for rownum in 1:nrow(params_df)
        scape_side = params_df[rownum, :Side]
        scape_carry_cap = params_df[rownum, :Capacity]
        scape_growth_rate = params_df[rownum, :RegRate]
        metab_range_tpl = (1, params_df[rownum, :MtblRate])
        vision_range_tpl = (1, params_df[rownum, :VsnRng])
        suglvl_range_tpl = (1, params_df[rownum, :InitSgLvl])
        pop_density = params_df[rownum, :Adensity]
        birth_rate = params_df[rownum, :Birthrate]
        inbound_rate = params_df[rownum, :InbndRt]
        outbound_rate = params_df[rownum, :OtbndRt]
        threshold = params_df[rownum, :Threshold]
        
            
        dict_objs = set_up_environment(scape_side, scape_carry_cap,
                                       scape_growth_rate, pop_density,
                                       metab_range_tpl, vision_range_tpl,
                                       suglvl_range_tpl)
        sugscape_obj = dict_objs["sugscape_obj"]
        arr_agents = dict_objs["arr_agents"]
        
        ## println(get_sugarscape_stats(sugscape_obj))
        ## println("\n\n")
        # plot_sugar_concentrations!(sugscape_obj)

        ## next, animate the simulation - move the agents, have them consume sugar,
        ## reduce the sugar in sugscape cells, regrow the sugar....and collect the
        ## array of gini coeffs
        arr_agent_ginis, arr_scape_ginis = animate_sim(sugscape_obj, arr_agents, time_periods, 
                                                       birth_rate, inbound_rate, outbound_rate,
                                                       vision_range_tpl, metab_range_tpl, 
                                                       suglvl_range_tpl, threshold)


        # for colnum in ncol(params_df)+1 : ncol(out_df)
        #     out_df[rownum, colnum] = arr_agent_ginis[colnum - ncol(params_df)]
        # end

        for colnum in ncol(params_df)+1 : ncol(out_df_agents)
            out_df_agents[rownum, colnum] = arr_agent_ginis[colnum - ncol(params_df)]
        end

        for colnum in ncol(params_df)+1 : ncol(out_df_scape)
            out_df_scape[rownum, colnum] = arr_scape_ginis[colnum - ncol(params_df)]
        end
        
        
        println("Finished combination $rownum")
        # println("Here's the out_df")
        # println(out_df)
        # readline()
    end #end iterate over param rows 

    # return(out_df)    
    return((out_df_agents, out_df_scape))
end ## run_sim



# run_sim() |> CSV.write("output.csv")
arr_seeds = [10, 80085, 4545, 4543543535, 87787765, 63542, 34983, 596895, 2152, 434];
for seednum in arr_seeds 
    outdf_agents, outdf_scape = run_sim(seednum)
    fname = "outputfile-agents-" * string(seednum) * ".csv"
    outdf_agents |> CSV.write(fname)
    fname = "outputfile-scape-" * string(seednum) * ".csv"
    outdf_scape |> CSV.write(fname)
    println("Finished processing seed: ", string(seednum))
end
