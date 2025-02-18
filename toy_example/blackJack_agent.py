opt_bj_agent = [
    [18,17,17,17,17,17,17,17,18,18],
    [16,12,12,11,11,11,16,16,16,16]
]

def game_agent(state, control):
    """
    Input: state: (dealer's first card, usable Ace, player's total points), control : list(2,10)
    usable Ace: control[0]
    un-usable Ace: control[1]
    Output: True: stick, False: Hit
    """
    (dealer_first, ace_usable, total_value) = state[0]
    if ace_usable:
        return control[0][dealer_first - 1] >= total_value
    else:
        return control[1][dealer_first - 1] >= total_value
