from agent import MahjongGBAgent
from collections import defaultdict
import numpy as np
import platform

# if platform.python_version()[:3] == "3.6":
#     from agent.fanCalcLib import formMinComb_c
# elif platform.python_version()[:3] == "3.9":
#     from agent.fanCalcLibPy39 import formMinComb_c

try:
    from MahjongGB import MahjongFanCalculator
except:
    print(
        "MahjongGB library required! Please visit https://github.com/ailab-pku/PyMahjongGB for more information."
    )
    raise


def hash_custom_tiles(custom_tile_dict):
    """
    hash custom encoded tile list into string, encoding alg: 34 digits of tiles, each digit represent tile count, order: BWTFJ
    """
    result_list = [0] * 34
    dict_order = [
        *("B%d" % (i + 1) for i in range(9)),
        *("W%d" % (i + 1) for i in range(9)),
        *("T%d" % (i + 1) for i in range(9)),
        *("F%d" % ((i) % 4 + 1) for i in range(4)),
        *("J%d" % ((i) % 3 + 1) for i in range(3)),
    ]
    for key in custom_tile_dict:
        if key != "shown":
            dict_id = dict_order.index(key)
            val = custom_tile_dict[key]
            result_list[dict_id] += val
    return "".join(str(e) for e in result_list)


def hash_seperated_custom_tile(custom_tile_dict_in_list):
    """
    same function as "hash_custom_tile" function, but dicts are in list, i.e. [{...},{...}, ...]
    flatten dicts and then call "hash_custom_tile"
    """
    # print(custom_tile_dict_in_list)
    # print(custom_tile_dict_in_list)
    flatten_dict = defaultdict(int)
    isHepta = True
    for tile_dict in custom_tile_dict_in_list:
        for k in tile_dict:
            if k != "AnGang" and tile_dict[k] % 2 == 1:
                isHepta = False

    for tile_dict in custom_tile_dict_in_list:
        for k in tile_dict:
            # print(tile_dict)
            # print("key", k)
            # print(tile_dict[k])
            if k != "AnGang":
                flatten_dict[k] += (
                    tile_dict[k] if isHepta else min(tile_dict[k], 3)
                )  # disregard Gang
    return hash_custom_tiles(flatten_dict)


def sortation(
    list1,
    list2,
    list3,
    list4,
    already_chi_peng,
    for_QQR=False,
    list2_penalty=0.4,
    list3_penalty=1.5,
    list4_penalty=2.1,
):
    """
    sort results from form_min_comb
    according to 上听数 and list tier, find at least 5default_list
    list1: list 1 from form_min_comb
    list2: list 2 from form_min_comb
    list3: list 3 from form_min_comb
    list4: list 4 from form_min_comb
    list*_penalty: self defined penalty to 上听数, default: list 1 +0, list 2, list 3+1, list 4 + 2
    """
    seen_target_list = []
    dist_penalty = [0, list2_penalty, list3_penalty, list4_penalty]
    dist_dict = defaultdict(list)
    list_holder = []
    list_holder.append(list1)
    list_holder.append(list2)
    list_holder.append(list3)
    list_holder.append(list4)
    considered_list_range = 4
    if already_chi_peng:
        considered_list_range = 2
    if for_QQR:
        considered_list_range = 4
        dist_penalty = [0, 0, 0, 0]
    for i in range(considered_list_range):
        if len(list_holder[i]) > 1:
            for entry in list_holder[i]:
                if (isinstance(entry, list) or isinstance(entry, tuple)) and len(
                    entry
                ) == 6:
                    entry = list(entry)
                    dist = entry[0] + dist_penalty[i]
                    # append indicator for list tier at the end of entry list
                    entry.append(i)
                    dist_dict[dist].append(entry)
    result_list = []
    # dict_dist_list = []
    for e in range(100):
        i = e / 10
        len_dict = len(dist_dict[i])
        # dict_dist_list.extend([i for _ in range(len_dict)])
        for result_case in dist_dict[i]:
            target_hash = hash_seperated_custom_tile(result_case[4])
            if target_hash not in seen_target_list:
                seen_target_list.append(target_hash)
                result_list.append(result_case)

    return (
        result_list  # ,  min(dict_dist_list), sum(dict_dist_list)/len(dict_dist_list)
    )


def find_redundant_and_missing(hand_list, target_info):
    selected_tiles = target_info[1]
    target_tile_list = target_info[4]
    target_tile_dict = defaultdict(int)
    for d in target_tile_list:
        for t in d:
            target_tile_dict[t] += d[t]

    # find redundant tile from hand and selected_tiles
    redundant_tile = defaultdict(int)
    for t in hand_list:
        if t not in selected_tiles:
            selected_tiles[t] = 0
        if hand_list[t] - selected_tiles[t] > 0:
            redundant_tile[t] += hand_list[t] - selected_tiles[t]

    # find missing tile from selected_tiles and target_tile_dict
    missing_tile = defaultdict(int)
    for t in target_tile_dict:
        if target_tile_dict[t] - selected_tiles.get(t, 0) > 0:
            missing_tile[t] = target_tile_dict[t] - selected_tiles.get(t, 0)

    return redundant_tile, missing_tile


class SearchAgentSlim(MahjongGBAgent):
    """
    observation: 12*4*9
        (game1+hand1+meld(4*1)+history(4*1)+unshown1+unknown1+search(32*2))*4*9
        where seatWind is stored on game1[0], prevailingWind is stored on game1[1]
    action_mask: 235
        pass1+hu1+discard34+chi63(3*7*3)+peng34+gang34+angang34+bugang34
    """

    OBS_SIZE = 12
    ACT_SIZE = 235

    OFFSET_OBS = {
        "GAME": 0,
        "HAND": 1,
        "MELD": 2,
        "HISTORY": 6,
        "UNSHOWN": 10,
        "UNKNOWN": 11,
        "SEARCH": 12,
    }
    OFFSET_ACT = {
        "Pass": 0,
        "Hu": 1,
        "Play": 2,
        "Chi": 36,
        "Peng": 99,
        "Gang": 133,
        "AnGang": 167,
        "BuGang": 201,
    }
    # TILE_LIST = [
    #     *("W%d" % (i + 1) for i in range(9)),
    #     *("T%d" % (i + 1) for i in range(9)),
    #     *("B%d" % (i + 1) for i in range(9)),
    #     *("F%d" % (i + 1) for i in range(4)),
    #     *("J%d" % (i + 1) for i in range(3)),
    # ]
    # OFFSET_TILE = {c: i for i, c in enumerate(TILE_LIST)}
    # OFFSET_TILE["CONCEALED"] = 35

    def __init__(self, seatWind, tile_coding):
        self.TILE_LIST = tile_coding
        self.OFFSET_TILE = {c: i for i, c in enumerate(self.TILE_LIST)}
        self.OFFSET_TILE["CONCEALED"] = 35
        self.seatWind = seatWind
        self.packs = [[] for i in range(4)]
        self.history = [[] for i in range(4)]
        self.tileWall = [21] * 4
        self.shownTiles = defaultdict(int)
        self.knownTiles = defaultdict(int)  # tiles that player does not know
        self.wallLast = False
        self.isAboutKong = False
        self.obs = np.zeros((self.OBS_SIZE, 36))
        self.obs[self.OFFSET_OBS["GAME"], self.seatWind] = 1
        self.game_step = 0

    """
    Wind 0..3
    Deal XX XX ...
    Player N Draw
    Player N Gang
    Player N(me) AnGang XX
    Player N(me) Play XX
    Player N(me) BuGang XX
    Player N(not me) Peng
    Player N(not me) Chi XX
    Player N(not me) AnGang
    
    Player N Hu
    Huang
    Player N Invalid
    Draw XX
    Player N(not me) Play XX
    Player N(not me) BuGang XX
    Player N(me) Peng
    Player N(me) Chi XX
    """

    def request2obs(self, request):
        t = request.split()
        if t[0] == "Wind":
            self.prevalentWind = int(t[1])
            self.obs[self.OFFSET_OBS["GAME"], 4 + self.prevalentWind] = 1
            return
        if t[0] == "Deal":
            self.hand = t[1:]
            for tile in t[1:]:
                self.knownTiles[tile] += 1
            self._hand_embedding_update()
            self._unshown_embedding_update()
            self._unknown_embedding_update()
            return
        if t[0] == "Huang":
            self.valid = []
            return self._obs()
        if t[0] == "Draw":
            # Available: Hu, Play, AnGang, BuGang
            self.tileWall[0] -= 1
            self.wallLast = self.tileWall[1] == 0
            tile = t[1]
            self.valid = []
            if self._check_mahjong(
                tile, isSelfDrawn=True, isAboutKong=self.isAboutKong
            ):
                self.valid.append(self.OFFSET_ACT["Hu"])
            self.isAboutKong = False
            self.hand.append(tile)
            self._hand_embedding_update()
            self.knownTiles[tile] += 1
            self._unknown_embedding_update()
            for tile in set(self.hand):
                self.valid.append(self.OFFSET_ACT["Play"] + self.OFFSET_TILE[tile])
                if (
                    self.hand.count(tile) == 4
                    and not self.wallLast
                    and self.tileWall[0] > 0
                ):
                    self.valid.append(
                        self.OFFSET_ACT["AnGang"] + self.OFFSET_TILE[tile]
                    )
            if not self.wallLast and self.tileWall[0] > 0:
                for packType, tile, offer in self.packs[0]:
                    if packType == "PENG" and tile in self.hand:
                        self.valid.append(
                            self.OFFSET_ACT["BuGang"] + self.OFFSET_TILE[tile]
                        )
            self._search_embedding_update()
            return self._obs()
        # Player N Invalid/Hu/Draw/Play/Chi/Peng/Gang/AnGang/BuGang XX
        p = (int(t[1]) + 4 - self.seatWind) % 4
        if t[2] == "Draw":
            self.tileWall[p] -= 1
            self.wallLast = self.tileWall[(p + 1) % 4] == 0
            return
        if t[2] == "Invalid":
            self.valid = []
            return self._obs()
        if t[2] == "Hu":
            self.valid = []
            return self._obs()
        if t[2] == "Play":
            self.tileFrom = p
            self.curTile = t[3]
            self.shownTiles[self.curTile] += 1
            self._unshown_embedding_update()
            self.history[p].append(self.curTile)
            self._history_embedding_update(p)
            if p == 0:
                self.hand.remove(self.curTile)
                self._hand_embedding_update()
                return
            else:
                # update known tiles
                self.knownTiles[self.curTile] += 1
                self._unknown_embedding_update()
                # Available: Hu/Gang/Peng/Chi/Pass
                self.valid = []
                if self._check_mahjong(self.curTile):
                    self.valid.append(self.OFFSET_ACT["Hu"])
                if not self.wallLast:
                    if self.hand.count(self.curTile) >= 2:
                        self.valid.append(
                            self.OFFSET_ACT["Peng"] + self.OFFSET_TILE[self.curTile]
                        )
                        if self.hand.count(self.curTile) == 3 and self.tileWall[0]:
                            self.valid.append(
                                self.OFFSET_ACT["Gang"] + self.OFFSET_TILE[self.curTile]
                            )
                    color = self.curTile[0]
                    if p == 3 and color in "WTB":
                        num = int(self.curTile[1])
                        tmp = []
                        for i in range(-2, 3):
                            tmp.append(color + str(num + i))
                        if tmp[0] in self.hand and tmp[1] in self.hand:
                            self.valid.append(
                                self.OFFSET_ACT["Chi"]
                                + "WTB".index(color) * 21
                                + (num - 3) * 3
                                + 2
                            )
                        if tmp[1] in self.hand and tmp[3] in self.hand:
                            self.valid.append(
                                self.OFFSET_ACT["Chi"]
                                + "WTB".index(color) * 21
                                + (num - 2) * 3
                                + 1
                            )
                        if tmp[3] in self.hand and tmp[4] in self.hand:
                            self.valid.append(
                                self.OFFSET_ACT["Chi"]
                                + "WTB".index(color) * 21
                                + (num - 1) * 3
                            )
                self.valid.append(self.OFFSET_ACT["Pass"])
                if len(self.valid) > 1:
                    self._search_embedding_update()
                return self._obs()
        if t[2] == "Chi":
            tile = t[3]
            color = tile[0]
            num = int(tile[1])
            self.packs[p].append(("CHI", tile, int(self.curTile[1]) - num + 2))
            self._pack_embedding_update(p)
            self.shownTiles[self.curTile] -= 1
            for i in range(-1, 2):
                self.shownTiles[color + str(num + i)] += 1
            self._unshown_embedding_update()
            if p != 0:
                self.knownTiles[self.curTile] -= 1
                for i in range(-1, 2):
                    self.knownTiles[color + str(num + i)] += 1
                self._unknown_embedding_update()
            self.wallLast = self.tileWall[(p + 1) % 4] == 0
            if p == 0:
                # Available: Play
                self.valid = []
                self.hand.append(self.curTile)
                for i in range(-1, 2):
                    self.hand.remove(color + str(num + i))
                self._hand_embedding_update()
                for tile in set(self.hand):
                    self.valid.append(self.OFFSET_ACT["Play"] + self.OFFSET_TILE[tile])
                return self._obs()
            else:
                return
        if t[2] == "UnChi":
            tile = t[3]
            color = tile[0]
            num = int(tile[1])
            self.packs[p].pop()
            self._pack_embedding_update(p)
            self.shownTiles[self.curTile] += 1
            for i in range(-1, 2):
                self.shownTiles[color + str(num + i)] -= 1
            self._unshown_embedding_update()
            if p != 0:
                self.knownTiles[self.curTile] += 1
                for i in range(-1, 2):
                    self.knownTiles[color + str(num + i)] -= 1
                self._unknown_embedding_update()
            if p == 0:
                for i in range(-1, 2):
                    self.hand.append(color + str(num + i))
                self.hand.remove(self.curTile)
                self._hand_embedding_update()
            return
        if t[2] == "Peng":
            self.packs[p].append(("PENG", self.curTile, (4 + p - self.tileFrom) % 4))
            self._pack_embedding_update(p)
            self.shownTiles[self.curTile] += 2
            self._unshown_embedding_update()
            if p != 0:
                self.knownTiles[self.curTile] += 2
                self._unknown_embedding_update()
            self.wallLast = self.tileWall[(p + 1) % 4] == 0
            if p == 0:
                # Available: Play
                self.valid = []
                for i in range(2):
                    self.hand.remove(self.curTile)
                self._hand_embedding_update()
                for tile in set(self.hand):
                    self.valid.append(self.OFFSET_ACT["Play"] + self.OFFSET_TILE[tile])
                return self._obs()
            else:
                return
        if t[2] == "UnPeng":
            self.packs[p].pop()
            self._pack_embedding_update(p)
            self.shownTiles[self.curTile] -= 2
            self._unshown_embedding_update()
            if p != 0:
                self.knownTiles[self.curTile] -= 2
                self._unknown_embedding_update()
            if p == 0:
                for i in range(2):
                    self.hand.append(self.curTile)
                self._hand_embedding_update()
            return
        if t[2] == "Gang":
            self.packs[p].append(("GANG", self.curTile, (4 + p - self.tileFrom) % 4))
            self._pack_embedding_update(p)
            self.shownTiles[self.curTile] += 3
            self._unshown_embedding_update()
            if p != 0:
                self.knownTiles[self.curTile] += 3
                self._unknown_embedding_update()
            if p == 0:
                for i in range(3):
                    self.hand.remove(self.curTile)
                self._hand_embedding_update()
                self.isAboutKong = True
            return
        if t[2] == "AnGang":
            tile = "CONCEALED" if p else t[3]
            self.packs[p].append(("GANG", tile, 0))
            self._pack_embedding_update(p)
            if p == 0:
                self.isAboutKong = True
                for i in range(4):
                    self.hand.remove(tile)
            else:
                self.isAboutKong = False
            return
        if t[2] == "BuGang":
            tile = t[3]
            for i in range(len(self.packs[p])):
                if tile == self.packs[p][i][1]:
                    self.packs[p][i] = ("GANG", tile, self.packs[p][i][2])
                    self._pack_embedding_update(p)
                    break
            self.shownTiles[tile] += 1
            self._unshown_embedding_update()
            if p != 0:
                self.knownTiles[tile] += 1
                self._unknown_embedding_update()
            if p == 0:
                self.hand.remove(tile)
                self._hand_embedding_update()
                self.isAboutKong = True
                return
            else:
                # Available: Hu/Pass
                self.valid = []
                if self._check_mahjong(tile, isSelfDrawn=False, isAboutKong=True):
                    self.valid.append(self.OFFSET_ACT["Hu"])
                self.valid.append(self.OFFSET_ACT["Pass"])
                return self._obs()
        raise NotImplementedError("Unknown request %s!" % request)

    """
    Pass
    Hu
    Play XX
    Chi XX
    Peng
    Gang
    (An)Gang XX
    BuGang XX
    """

    def action2response(self, action):
        if action < self.OFFSET_ACT["Hu"]:
            return "Pass"
        if action < self.OFFSET_ACT["Play"]:
            return "Hu"
        if action < self.OFFSET_ACT["Chi"]:
            return "Play " + self.TILE_LIST[action - self.OFFSET_ACT["Play"]]
        if action < self.OFFSET_ACT["Peng"]:
            t = (action - self.OFFSET_ACT["Chi"]) // 3
            return "Chi " + "WTB"[t // 7] + str(t % 7 + 2)
        if action < self.OFFSET_ACT["Gang"]:
            return "Peng"
        if action < self.OFFSET_ACT["AnGang"]:
            return "Gang"
        if action < self.OFFSET_ACT["BuGang"]:
            return "Gang " + self.TILE_LIST[action - self.OFFSET_ACT["AnGang"]]
        return "BuGang " + self.TILE_LIST[action - self.OFFSET_ACT["BuGang"]]

    """
    Pass
    Hu
    Play XX
    Chi XX
    Peng
    Gang
    (An)Gang XX
    BuGang XX
    """

    def response2action(self, response):
        t = response.split()
        if t[0] == "Pass":
            return self.OFFSET_ACT["Pass"]
        if t[0] == "Hu":
            return self.OFFSET_ACT["Hu"]
        if t[0] == "Play":
            return self.OFFSET_ACT["Play"] + self.OFFSET_TILE[t[1]]
        if t[0] == "Chi":
            return (
                self.OFFSET_ACT["Chi"]
                + "WTB".index(t[1][0]) * 7 * 3
                + (int(t[2][1]) - 2) * 3
                + int(t[1][1])
                - int(t[2][1])
                + 1
            )
        if t[0] == "Peng":
            return self.OFFSET_ACT["Peng"] + self.OFFSET_TILE[t[1]]
        if t[0] == "Gang":
            return self.OFFSET_ACT["Gang"] + self.OFFSET_TILE[t[1]]
        if t[0] == "AnGang":
            return self.OFFSET_ACT["AnGang"] + self.OFFSET_TILE[t[1]]
        if t[0] == "BuGang":
            return self.OFFSET_ACT["BuGang"] + self.OFFSET_TILE[t[1]]
        return self.OFFSET_ACT["Pass"]

    def _obs(self):
        mask = np.zeros(self.ACT_SIZE)
        for a in self.valid:
            mask[a] = 1
        self.game_step += 1
        # convert game_step into binary
        game_step_binary_list = list(int(i) for i in format(self.game_step, f"010b"))
        self.obs[self.OFFSET_OBS["GAME"], 8:18] = game_step_binary_list
        return {
            "observation": self.obs.reshape((self.OBS_SIZE, 4, 9)).copy(),
            "action_mask": mask,
        }

    def _unshown_embedding_update(self):
        for tile in self.TILE_LIST:
            self.obs[self.OFFSET_OBS["UNSHOWN"], self.OFFSET_TILE[tile]] = 0.25 * (
                4 - self.shownTiles[tile]
            )

    def _unknown_embedding_update(self):
        for tile in self.TILE_LIST:
            self.obs[self.OFFSET_OBS["UNKNOWN"], self.OFFSET_TILE[tile]] = 0.25 * (
                4 - self.knownTiles[tile]
            )

    def _search_embedding_update(self):
        """
        perform target search and return 32 closest targets
        """
        return
        search_feature = np.zeros((64, 36))
        # construct hand
        hand_dict = defaultdict(int)
        for t in self.hand:
            hand_dict[t] += 1
        # construct pack
        pack_dict_list = []
        for pack in self.packs[0]:
            packdict = defaultdict(int)
            packType, tile, offer = pack
            if packType == "CHI":
                color = tile[0]
                num = int(tile[1])
                for j in range(-1, 2):
                    packdict[color + str(num + j)] += 1
                    # self.obs[offset, self.OFFSET_TILE[tile] + i] += 1
            elif packType == "PENG":
                packdict[tile] += 3
                # self.obs[offset : offset + 3, self.OFFSET_TILE[tile]] += 1
            else:
                packdict[tile] += 4
                # self.obs[offset : offset + 4, self.OFFSET_TILE[tile]] += 1
            pack_dict_list.append(packdict.copy())
        # construct tileWall, a.k.a. unknown tiles
        tile_wall_dict = defaultdict(int)
        for tile in self.knownTiles:
            tile_wall_dict[tile] = max(0, 4 - self.knownTiles[tile])
        (list1, list1id, list2, list2id, list3, list3id, list4, list4id) = (
            formMinComb_c(
                hand_dict,
                pack_dict_list,
                tile_wall_dict,
                self.seatWind,
                self.prevalentWind,
                32,
                6,
                8,
                False,
            )
        )
        # from four lists sort and construct 32 closest targets
        already_chi_peng = False
        for ent in pack_dict_list:
            if "AnGang" not in ent.keys():
                already_chi_peng = True
        list1.append(list1id)
        list2.append(list2id)
        list3.append(list3id)
        list4.append(list4id)
        sorted_list = sortation(list1, list2, list3, list4, already_chi_peng)
        if len(sorted_list) > 32:
            sorted_list = sorted_list[:32]

        self.search_redundant = []
        self.search_missing = []

        for i, entry in enumerate(sorted_list):
            # from sorted_list generate need tile and redundant tile
            redundant_tile, missing_tile = find_redundant_and_missing(
                hand_dict.copy(), entry
            )
            self.search_redundant.append(redundant_tile)
            self.search_missing.append(missing_tile)
            # construct feature map from redundant_tile and missing_tile
            for t in missing_tile:
                search_feature[2 * i, self.OFFSET_TILE[t]] = max(
                    1, 0.25 * missing_tile[t]
                )
            for t in redundant_tile:
                search_feature[2 * i + 1, self.OFFSET_TILE[t]] = max(
                    1, 0.25 * redundant_tile[t]
                )
        # copy search feature into obs
        self.obs[self.OFFSET_OBS["SEARCH"] : self.OFFSET_OBS["SEARCH"] + 64] = (
            search_feature
        )

    def _pack_embedding_update(self, p):
        """
        Convert self.packs into dict, then into 1*4*9
        """
        # calculate offset
        offset = self.OFFSET_OBS["MELD"] + p
        # clear embedding
        self.obs[offset] = 0
        # calculate dict
        l = len(self.packs[p])
        packdict = defaultdict(int)
        for i in range(l):
            packType, tile, offer = self.packs[p][i]
            if packType == "CHI":
                for j in range(-1, 2):
                    packdict[self.OFFSET_TILE[tile] + j] += 1
                    # self.obs[offset, self.OFFSET_TILE[tile] + i] += 1
            elif packType == "PENG":
                packdict[self.OFFSET_TILE[tile]] += 3
                # self.obs[offset : offset + 3, self.OFFSET_TILE[tile]] += 1
            else:
                packdict[self.OFFSET_TILE[tile]] += 4
                # self.obs[offset : offset + 4, self.OFFSET_TILE[tile]] += 1
        # make embedding
        for tile_offset in packdict:
            self.obs[offset, tile_offset] = 0.25 * packdict[tile_offset]

    def _hand_embedding_update(self):
        self.obs[self.OFFSET_OBS["HAND"]] = 0
        d = defaultdict(int)
        for tile in self.hand:
            d[tile] += 1
        for tile in d:
            self.obs[self.OFFSET_OBS["HAND"], self.OFFSET_TILE[tile]] = 0.25 * d[tile]

    def _history_embedding_update(self, p):
        play_history = self.history[p]
        d = defaultdict(int)
        for tile in play_history:
            d[tile] += 1
        # update embedding
        for tile in d:
            self.obs[self.OFFSET_OBS["HISTORY"] + p, self.OFFSET_TILE[tile]] = (
                0.25 * d[tile]
            )

    def _check_mahjong(self, winTile, isSelfDrawn=False, isAboutKong=False):
        try:
            fans = MahjongFanCalculator(
                pack=tuple(self.packs[0]),
                hand=tuple(self.hand),
                winTile=winTile,
                flowerCount=0,
                isSelfDrawn=isSelfDrawn,
                is4thTile=(self.shownTiles[winTile] + isSelfDrawn) == 4,
                isAboutKong=isAboutKong,
                isWallLast=self.wallLast,
                seatWind=self.seatWind,
                prevalentWind=self.prevalentWind,
                verbose=True,
            )
            fanCnt = 0
            for fanPoint, cnt, fanName, fanNameEn in fans:
                fanCnt += fanPoint * cnt
            if fanCnt < 8:
                raise Exception("Not Enough Fans")
        except:
            return False
        return True
