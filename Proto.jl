using Random

mutable struct Transaction
    transaction_amount::Float64
    period::Int32
    type::String ## deposit or withdrawal or closure
    agent_id::Int64    
end

mutable struct Proto
    proto_id::Int64
    sugar_level::Float64
    alive::Bool
    arr_member_ids::Array{Int64, 1}
    ledger_transactions::Array{Transaction, 1}
end

function fetch_specific_proto_obj(arr_protos, proto_id)
    """
    Return an proto object from the given array of proto objects
    that has the given proto id.
    """
    return [probj for probj in arr_protos 
            if probj.proto_id == proto_id][1] 
end

function fetch_neighbors(agobj, arr_agents, sugscape_obj, threshold)
    """
    For a given agent, identify agents that are in the n-s-e-w cells
    whose sugar_level > threshold and return them.
    """

    neighbor_locs = [(agobj.location_x - 1, agobj.location_y), 
                     (agobj.location_x + 1, agobj.location_y),
                     (agobj.location_x, agobj.location_y - 1), 
                     (agobj.location_x, agobj.location_y + 1)]

    neighbor_locs = [loc for loc in neighbor_locs if 
                     loc[1] >= 1 & loc[1] <= size(sugscape_obj)[1] &
                     loc[2] >= 1 & loc[2] <= size(sugscape_obj)[2]]

    neighbor_agents = [nagobj for nagobj in arr_agents 
                       if nagobj.sugar_level > threshold &&
                       (nagobj.location_x, nagobj.location_y) 
                       in neighbor_locs]
    ## println("Entered fetch_neighbors")
    ## readline()
    try
        @assert size(neighbor_agents)[1] in 0:4
    catch except
        println("********************************************************************" )
        println("")
        println("No. of neighbor agents: ", length(neighbor_agents))
        println("********************************************************************" )
    end 
    shuffle!(neighbor_agents)
    return(neighbor_agents)
end ## end fetch_neighbors

function form_possible_protos!(arr_agents, threshold, sugscape_obj, 
                                     arr_protos, timeperiod)
    """
    Called during each time period to enact association among agents to form
    protos. The following conditions exist:
    (a) none of the focal agent's neighbors are part of an proto
    (b) one of the focal agent's neighbors is part of an proto
    (c) two or more of the focal agent's neighbors are part of an proto.
    
    In the case of (a), a randomly chosen neighbor and the focal agent join to
    form an proto.
    In the case of (b), the focal agent joins the proto of the neighbor.
    In the case of (c), the focal agent joins the proto of a 
    randomly-chosen neighbor.
    
    Modifies focal and neighbor agents' proto_id field, if proto-
    formation or joining occurs.
    Reduces the focal agent and neighbor agents' sugarlevels by an amount 
    = sugarlevel - threshold.

    Modifies the array of protos by adding, as needed, new proto struct.
    
    Returns an array of Proto structures to the caller.
    """
    # println("At the beginning form_possible_protos! the array of protos is:")
    # ## println(arr_protos)
    # println("Length of arr_protos is: ", string(length(arr_protos)))
    
    start_proto_id = begin
        if length(arr_protos) == 1 && arr_protos[1].proto_id == -1
            1
        else
            maximum([protoobj.proto_id for protoobj in arr_protos]) + 1
        end
    end
    ## println("The start_proto_id is: ", string(start_proto_id))
    for agobj in arr_agents
        ## update their proto_ids. 
        ## println("Current object:", string(agobj.agent_id), 
                # " its excess sugarlevel: ", agobj.sugar_level - threshold)
        ## println("Enter enter")
        ## readline()
        if agobj.proto_id == -1 && agobj.sugar_level > threshold 
            ## fetch neighbors
            ## println("Checking to see if ", string(agobj.agent_id), " can join an", " proto")
            arr_neighbors = fetch_neighbors(agobj, arr_agents, sugscape_obj, threshold)
            ## println("Here here here!")
            if !isempty(arr_neighbors)

                partner_agent = arr_neighbors[1]

                if partner_agent.proto_id == -1 ## (a)
                    ## println("No neighbor found with an existing membership, ",
                            # "so creating a new proto with agent: ", 
                            # string(partner_agent.agent_id))
                    agobj.proto_id = start_proto_id
                    agobj.sugar_level = agobj.sugar_level - threshold
                    transaction1 = Transaction(agobj.sugar_level - threshold,
                                               timeperiod, "deposit", 
                                               agobj.agent_id)

                    partner_agent.proto_id = start_proto_id
                    partner_agent.sugar_level = partner_agent.sugar_level - threshold
                    transaction2 = Transaction(partner_agent.sugar_level - threshold,
                                               timeperiod, "deposit", 
                                               partner_agent.agent_id)

                    prototn_obj = Proto(start_proto_id, 
                                       transaction1.transaction_amount +
                                       transaction2.transaction_amount,
                                       true,
                                       [agobj.agent_id, partner_agent.agent_id],
                                       [transaction1, transaction2])

                    push!(arr_protos, prototn_obj)
                    ## println("Added ", string(agobj.agent_id), " to a new ",
                            # "proto with id:", start_proto_id)

                else ## (b) and (c)
                    ## println("Inside the (b) and (c) conditional branch")
                    protot_obj = fetch_specific_proto_obj(arr_protos, 
                                                        partner_agent.proto_id)
                    @assert protot_obj.proto_id > 0
                    agobj.proto_id = partner_agent.proto_id
                    agobj.sugar_level = agobj.sugar_level - threshold
                    transaction1 = Transaction(agobj.sugar_level - threshold,
                                               timeperiod, "deposit", 
                                               agobj.agent_id)
                    push!(protot_obj.ledger_transactions, transaction1)
                    
                end ## (a), (b), (c)
            end ## !isempty(arr_neighbors)
        end ## agobj.sugar_level > threshold & agobj.proto_id > 0
    end ## for agobj    
end ## end of form_possible_protos

function update_proto_statuses!(arr_protos, timeperiod)
    """
    Goes through the array of protos and updates the status
    of each proto: if all of the member agents are dead,
    the proto's alive status is set to false, and a closing transaction
    is added, with a transaction amount of zero, date = current period, and
    agent_id = -999.
    """
    for proto_obj in arr_protos
        if proto_obj.sugar_level <= 0
            proto_obj.alive = false
            proto_obj.arr_member_ids = []
            push!(proto_obj.ledger_transactions, 
                  Transaction(0, timeperiod, "closure", -999))
        end
    end
end
