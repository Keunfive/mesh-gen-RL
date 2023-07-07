from Meshpkg import params as p


def penalty_reward(replay_memory, info, step_ended, penalty_layer_scale = 5, penalty_lr = 3, penalty = [-50, -30, -10, 0]): 

    def layer_penalty(reward_layer, penalty_factor): 
        penalty_applied = set()
        
        for i in info:
            start_idx = max(0, i - penalty_lr)
            end_idx = min(p.surf_length, (i+ penalty_lr+1))
            penalty_applied = set.union(penalty_applied, set(range(start_idx, end_idx)))
        if penalty_factor < 0: #
            for j in penalty_applied:
                reward_layer[j] = penalty_factor
        else:
            for j in penalty_applied:
                reward_layer[j] *= penalty_factor

        return reward_layer

    for n in range(1, min(3*penalty_layer_scale+2, step_ended+1)): 
        replay_memory[-n] = list(replay_memory[-n]) # list로 가져오기
        
        # 최외각(-1층) layer(꼬임 발생 당사자)에 penalty 부여
        if (n == 1): 
            replay_memory[-n][2] = layer_penalty(replay_memory[-n][2], penalty[0])
        
         #(-1 ~ -1*penalty layer_scale층) layer에 penalty 부여
        elif (n >= 2 and n <= penalty_layer_scale+1):
            replay_memory[-n][2] = layer_penalty(replay_memory[-n][2], penalty[1])
        
        #(-1*penalty layer_scale층 ~ -2*penalty layer_scale층) layer에 penalty 부여
        elif (n >= penalty_layer_scale + 2 and n <= 2*penalty_layer_scale+1): 
            replay_memory[-n][2] = layer_penalty(replay_memory[-n][2], penalty[2])
        
        #(-2*penalty layer_scale층 ~ -3*penalty layer_scale층) layer에 penalty 부여
        elif (n >= 2*penalty_layer_scale + 2 and n <= 3*penalty_layer_scale+1): 
            replay_memory[-n][2] = layer_penalty(replay_memory[-n][2], penalty[3])

        replay_memory[-n] = tuple(replay_memory[-n]) # 다시 tuple로 묶기

    return replay_memory