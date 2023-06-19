
def get_action_type(dim):
    if dim == 1:
        actions = [i for i in range(1,6)]

    elif dim == 2:
        actions = [ ]
        for i in range(1,6):
            for j in range(1,6):
                actions.append([i,j])

    return actions