from Meshpkg import params as p


def penalty_reward(replay_memory, info, step_ended, penalty_step_1_3 = 5, penalty_lr = 3): # reward penalty 부여

    def layer_penalty(reward_layer, penalty): # reward_layer, index, penalty 배율: penalty
        penalty_applied = set()
        
        for i in info:
            start_idx = max(0, i - penalty_lr)
            end_idx = min(p.surf_length, (i+ penalty_lr+1))
            penalty_applied = set.union(penalty_applied, set(range(start_idx, end_idx)))
        if penalty < 0:
            for j in penalty_applied:
                reward_layer[j] = penalty
        else:
            for j in penalty_applied:
                reward_layer[j] *= penalty

        return reward_layer

    for n in range(1, min(3*penalty_step_1_3+2, step_ended+1)): 
        replay_memory[-n] = list(replay_memory[-n])
    #원랜 0 / 0.5 / 0.6 / 0.7 순서 
        if (n == 1):
            replay_memory[-n][2] = layer_penalty(replay_memory[-n][2], -50)
        elif (n >= 2 and n <= penalty_step_1_3+1):
            replay_memory[-n][2] = layer_penalty(replay_memory[-n][2], -30)
        elif (n >= penalty_step_1_3+2 and n <= 2*penalty_step_1_3+1):
            replay_memory[-n][2] = layer_penalty(replay_memory[-n][2], -10)
        elif (n >= 2*penalty_step_1_3+2 and n <= 3*penalty_step_1_3+1):
            replay_memory[-n][2] = layer_penalty(replay_memory[-n][2], 0)

        replay_memory[-n] = tuple(replay_memory[-n])

    return replay_memory