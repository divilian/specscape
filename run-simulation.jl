include("Sugarscape.jl")
include("Agent.jl")
using Statistics
using Random
using Distributions
using CSV
using DataFrames
using RCall

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
    
    return(Dict("sugscape_obj" => sugscape_obj,
                "arr_agents" => arr_agents)) 
end ## end of set_up_environment()

function animate_sim(sugscape_obj, arr_agents, time_periods, 
                     birth_rate, inbound_rate, outbound_rate,
                     vision_range_tpl, metab_range_tpl, suglvl_range_tpl)
    """
    Performs the various operations on the sugarscape and agent population
    to 'animate' them.
    Returns a single row, consisting of all of the params + gini values
    of sugar across all the time periods
    
    """
    metabol_distrib =  DiscreteUniform(metab_range_tpl[1], metab_range_tpl[2]);
    vision_distrib = DiscreteUniform(vision_range_tpl[1], vision_range_tpl[2]);
    suglvl_distrib = DiscreteUniform(suglvl_range_tpl[1], suglvl_range_tpl[2]); 
    arr_ginis = zeros(time_periods)
    for period in 1:time_periods 
        for ind in shuffle(1:length(arr_agents))
            locate_move_feed!(arr_agents[ind], sugscape_obj, arr_agents)            
        end 
        regenerate_sugar!(sugscape_obj) 
        perform_birth_inbound_outbound!(arr_agents, sugscape_obj, birth_rate, 
                                        inbound_rate, outbound_rate, 
                                        vision_distrib, metabol_distrib,
                                        suglvl_distrib) 
        life_check!(arr_agents) 
        arr_ginis[period] = compute_Gini(arr_agents)
        ## println("Finished time-step: ", string(period), "\n\n") 
    end## end of time_periods for loop
    return(arr_ginis)
end ## end animate_sim()

function run_sim()
    ## Random.seed!(13990);
    params_df = CSV.read("parameter-ranges-testing-new.csv")
    outfile_name = "outputs-new.csv"
    time_periods = 100

    temp_out = DataFrame(zeros(nrow(params_df), time_periods))
    names!(temp_out, Symbol.(["prd_"*string(i) for i in 1:time_periods]))

    out_df = DataFrame()
    for colname in names(params_df)
        out_df[Symbol(colname)] = params_df[Symbol(colname)]
    end

    for colname in names(temp_out)
        out_df[Symbol(colname)] = temp_out[Symbol(colname)]
    end
    
    for rownum in 1:nrow(params_df)
        scape_side = params_df[2, :Side]
        scape_carry_cap = params_df[2, :Capacity]
        scape_growth_rate = params_df[2, :RegRate]
        metab_range_tpl = (1, params_df[2, :MtblRate])
        vision_range_tpl = (1, params_df[2, :VsnRng])
        suglvl_range_tpl = (1, params_df[2, :InitSgLvl])
        pop_density = params_df[2, :Adensity]
        birth_rate = params_df[2, :Birthrate]
        inbound_rate = params_df[2, :InbndRt]
        outbound_rate = params_df[2, :OtbndRt]
        
            
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
        arr_ginis = animate_sim(sugscape_obj, arr_agents, time_periods, 
                                birth_rate, inbound_rate, outbound_rate,
                                vision_range_tpl, metab_range_tpl, 
                                suglvl_range_tpl)

        # for colname in names(params_df)
        #     out_df[rownum, Symbol(colname)] = params_df[rownum, Symbol(colname)]
        # end

        for colnum in ncol(params_df)+1 : ncol(out_df)
            out_df[rownum, colnum] = arr_ginis[colnum - ncol(params_df)]
        end
        
        
        ## create a row
        println("Finished combination $rownum")
        # println("Here's the out_df")
        # println(out_df)
        # readline()
    end #end iterate over param rows 

    ## return the output df 
    return(out_df)
    
end ## run_sim

run_sim() |> CSV.write("output.csv")
