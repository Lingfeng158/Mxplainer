#ifndef __RULE_H__
#define __RULE_H__

#include <stdio.h>
#include <string>
#include <map>
#include <tuple>
#include <unordered_map>
#include <vector>
#include <unordered_set>
#include <set>

typedef std::map<std::string, int, std::less<std::string>> tileHolderMap;
typedef std::vector<tileHolderMap> dictHolderVec;
typedef std::tuple<int, tileHolderMap, int, std::unordered_set<std::string>, dictHolderVec, std::unordered_set<std::string>> resultInfo;
typedef std::vector<resultInfo> resultInfoVec;
typedef std::tuple<int, tileHolderMap, int, dictHolderVec> partialResult;
typedef std::tuple<int, tileHolderMap, int, dictHolderVec, tileHolderMap, tileHolderMap> partialResultDP;
typedef std::vector<partialResult> partialResultVec;
typedef std::vector<partialResultDP> partialResultDPVec;

/**
 * Count tiles of one type, specified by tileType, within handDict
 * @param tileType: desired tileType, "B","W","T","X", where "X" represents Feng and Jian （风，箭）
 * @param handDict: dict that hold tiles representing player's hand tiles
 */
int oneTypeTileCount(std::string tileType, tileHolderMap handDict);

/**
 * Provide necessary information, given handDict and tileWallDict, to form trio-s for specified tileType
 * return dist_to_trio (for each trio formation), avail_tiles_to_trios (remaining tiles from tileWallDict)
 */
std::tuple<std::vector<int>, std::vector<int>> oneTypeDistToTrio(std::string tileType, tileHolderMap handDict, tileHolderMap tileWallDict);

std::tuple<int, int> distToTrio(std::string tileType, int rank, tileHolderMap handDict, tileHolderMap tileWallDict);

/**
 * Provide necessary information, given handDict and tileWallDict, to form trio-s for specified tileType
 * Return structure similar to that of **_to_trio
 */
std::tuple<std::vector<int>, std::vector<int>> oneTypeDistToStraight(std::string tileType, tileHolderMap handDict, tileHolderMap tileWallDict);

std::tuple<int, int> distToStraight(std::string tileType, int rank, tileHolderMap handDict, tileHolderMap tileWallDict);

/**
 * Provide necessary information, given handDict and tileWallDict, to form trio-s for specified tileType
 * Return strucutre similar to that of **_to_strio
 */
std::tuple<std::vector<int>, std::vector<int>> oneTypeDistToDuo(std::string tileType, tileHolderMap handDict, tileHolderMap tileWallDict);

std::tuple<int, int> distToDuo(std::string tileType, int rank, tileHolderMap handDict, tileHolderMap tileWallDict);

std::tuple<int, std::tuple<tileHolderMap, tileHolderMap>, int> oneTypeMinDistToDuo(std::string tileType, tileHolderMap handDict, tileHolderMap tileWallDict);
std::tuple<int, std::tuple<tileHolderMap, tileHolderMap>, int> oneTypeMinDistToTriplet(std::string tileType, tileHolderMap handDict, tileHolderMap tileWallDict, bool isFengJian = false);

/**
 * Helper function for one_type_dist_to_**
 * For one tileType, return the minimum-distance combination's information to desired structure
 */
std::tuple<int, std::string, int> oneTypeMinDistAndTile(std::string tileType, std::vector<int> dists2Structure, std::vector<int> availTiles2Structure, bool isStraight);

/**
 * Mapping from tile_choice to tile_composition, based on handDict
 * Return tiles-held, tiles-desired
 */
std::tuple<tileHolderMap, tileHolderMap> tileComposition(tileHolderMap handDict, std::string tileSelection, bool isStraight, bool isDuo = false);

tileHolderMap updateTileInfo(tileHolderMap tileWallDict, const tileHolderMap &tileAppearanceDict, const dictHolderVec &packAppearanceDict = dictHolderVec());

std::tuple<int, std::unordered_set<std::string>> calcExactFanWithMahJongGB(tileHolderMap handWallDict, dictHolderVec pack, std::string winTile, bool isLastTile, bool isSelfDrawn, bool is4thTile, bool isKongRelated, int seatWind, int prevailingWind);

/**
 * Equivalent to calc_fan_with_PyMahJongGB_quick in rule.py
 * Test if proposed hand set satisfies eight-fan, but only try from unfinished_set
 * @param handWallDict: 手牌
 * @param pack: 附录
 * @param targetTileDict: 以dictHolderMap形式呈现的所期望的组合
 * return: 是否可行，吃碰是否有限制，胡牌是否有限制
 */
std::tuple<bool, bool, bool, std::unordered_set<std::string>, std::unordered_set<std::string>> calcFanWithMahJongGB(tileHolderMap selectedTileDict, dictHolderVec pack, dictHolderVec targetTileDict, int seatWind = 0, int prevailingWind = 1, int targetFanVal = 8, bool disableJueZhang = false);

bool checkHu(tileHolderMap handDict, dictHolderVec packList, tileHolderMap tileWallDict, std::string newTile, int seatWind, int prevailingWind, bool isSelfDrawn, bool isAboutKong = false);

std::tuple<std::unordered_set<std::string>, std::unordered_set<std::string>> seperateIncompleteSet(tileHolderMap selectedEnc, tileHolderMap targetEnc);

// Helper Functions

std::vector<std::string> fromCustomToCanonicalEncoding(tileHolderMap custom_enc);

tileHolderMap fromCanonicalToCustomEncoding(std::vector<std::string> can_enc);

// tileHolderMap fromCanonicalToCustomPack(std::vector<int> canonicalPack);

// std::vector<int> fromCustomToCanonicalPack(tileHolderMap customPack);

std::string hashCustomTiles(tileHolderMap customTileDict);

std::string hashCustomVec(dictHolderVec customTileVec);

// tileHolderMap restoreHashedTiles(std::string tileHash);

// C++ specific function for adapting to MahJongGB
/**
 * Converting tile representation format to that specfied by MahJongGB
 */
std::string tileConversion2MahJongGB(tileHolderMap handDict, dictHolderVec packList, std::string winTile = "");

/**
 * Convert MahJongGB encoding to string
 * tile_value_t @ line 160 "tile.h"
 */
std::string tilefromMahJongGBEnc(int mahjongGBEncoding);

// std::string toString(tileHolderMap inputContainer);

// std::string toString(dictHolderVec inputContainer);

// std::string toString(std::vector<float> inputContainer);

// std::string toString(std::vector<int> inputContainer);

// std::string toString(std::unordered_set<std::string> inputContainer);

// std::string toString(resultInfoVec inputContainer);

// std::string toString(partialResult inputContainer);

// std::string toString(std::vector<std::string> inputContainer);

#endif