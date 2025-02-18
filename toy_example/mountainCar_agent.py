def game_agent(state, control):
    """
    Input: (location, velocity), control : list of length 4
    Output: action
    """
    (location, velocity) = state[0], state[1]
    if location <= 0 and velocity <= 0:
        return 0 if control[0] < 1 else 2
    if location <= 0 and velocity > 0:
        return 0 if control[1] < 1 else 2
    if location > 0 and velocity <= 0:
        return 0 if control[2] < 1 else 2
    if location > 0 and velocity > 0:
        return 0 if control[3] < 1 else 2
    return 1
