import os
import numpy as np
import random


## Helper Functions Defined Here ##


def initial_shuffle():
    card_list = [i for i in range(52)]
    random.shuffle(card_list)
    random.shuffle(card_list)
    return card_list


# define card to numb mapping: 1,2,3,4,5: Ace of Spade, Ace of Hearts, Ace of Diamonds, Ace of Clubs, 2 of Spade, etc.


def num_to_card(num):
    suit_list = ["Spade", "Heart", "Diamond", "Club"]
    rank = (num // 4) + 1
    suit = num % 4
    return "{} of {}".format(rank, suit_list[suit])


def num_to_value(num):
    rank = (num // 4) + 1
    return rank if rank < 10 else 10


class BlackJackEnv:
    def __init__(self):
        self.reset()

    def reset(self):
        self.card_list = initial_shuffle()
        self.dealer_hand = []
        self.player_hand = []
        # self.player_hand.extend([16, 28, 2, 8, 7])
        # self.dealer_hand.extend([19, 32, 1, 47])
        self.player_hand.extend(self.card_list[:2])
        self.dealer_hand.extend(self.card_list[2:4])
        self.card_list_counter = 4
        self.resolve = False
        return self.state()

    def state(self):
        """
        query state: return dealer's card (first), usable ace, and player value
        return [state], Ended, value
        """
        dealers_card = num_to_value(self.dealer_hand[0])
        player_card_value_list = [num_to_value(i) for i in self.player_hand]
        player_value = sum(player_card_value_list)
        usable_ace = 1 in player_card_value_list and player_value <= 11
        # resolve player's usable ace
        if usable_ace:
            player_value += 10
        # Test end
        end = False
        reward = 0
        if player_value > 21:
            end = True
            reward = -1

        if self.resolve:
            end = True
            dealer_card_value_list = [num_to_value(i) for i in self.dealer_hand]
            dealer_value = sum(dealer_card_value_list)
            dealer_usable_ace = 1 in dealer_card_value_list and dealer_value <= 11
            if dealer_usable_ace:
                dealer_value += 10

            if dealer_value > player_value:
                reward = -1
            elif dealer_value < player_value:
                reward = 1
            else:
                reward = 0
            if dealer_value > 21:
                reward = 1
        return ([dealers_card, usable_ace, player_value], end, reward)

    def step(self, action):
        """
        action: 1: hit
        action: 0: stick
        """
        if action == 1:
            self.player_hand.append(self.card_list[self.card_list_counter])
            self.card_list_counter += 1
        else:
            self.resolve = True
            keep_dealing = True
            while keep_dealing:
                # fixed strategy for dealer
                dealer_card_value_list = [num_to_value(i) for i in self.dealer_hand]
                dealer_value = sum(dealer_card_value_list)
                usable_ace = 1 in dealer_card_value_list and dealer_value <= 11
                if usable_ace:
                    dealer_value += 10
                if dealer_value >= 17:
                    keep_dealing = False
                else:
                    self.dealer_hand.append(self.card_list[self.card_list_counter])
                    self.card_list_counter += 1
        return self.state()

    def expose(self):
        return self.player_hand, self.dealer_hand


if __name__ == "__main__":
    env = BlackJackEnv()
    p, d = env.expose()
    print(p, d)
    print([num_to_card(i) for i in p], [num_to_card(i) for i in d])
    print([num_to_value(i) for i in p], [num_to_value(i) for i in d])
    print(env.state())
