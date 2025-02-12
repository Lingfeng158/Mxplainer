#ifndef __GENERIC_H__
#define __GENERIC_H__

#include "rule.h"
#include "special.h"

/**
 * for @param tileType, generate possible combinations given specified @param tripleCount and @param duoCount
 * @param tileType: BWTX, where X reresents F and J
 * @param totalIteration: record internal state of function for recursion. recursion strategy, find triple first, then duo, if any
 * @param passedOnDist: record dist encountered so far, from previous iteraitons
 */
std::tuple<partialResultVec, int, tileHolderMap, int, dictHolderVec> oneTypeAllCombination(std::string tileType, tileHolderMap handDict, tileHolderMap tileWallDict, int tripleCount, int duoCount, int maxDist = 8, int totalIteration = 0, int passedOnDist = 0);

/**
 * Similar to oneTypeAllCombination but in dynamic programming
 */
void oneTypeAllCombinationDP(std::string tileType, tileHolderMap &handDict, tileHolderMap &tileWallDict, int tripleCount, int duoCount, std::map<std::string, partialResultDPVec> &lookupTable, int maxDist = 6);

/**
 * for @param tileType, generate a lookup table of all distances
 */
std::map<std::string, partialResultVec> oneTypeLookup(std::string tileType, tileHolderMap handDict, dictHolderVec pack, tileHolderMap tileWallDict);

/**
 * Similar to oneTypeLookup but in dynamic programming
 */
std::map<std::string, partialResultDPVec> oneTypeLookupDP(std::string tileType, tileHolderMap handDict, dictHolderVec pack, tileHolderMap tileWallDict);

/**
 * return 4 levels of arrays of possible hu combinations
 * list 0: unrestricted on chi/peng and hu
 * list 1: restricted on hu's winTile
 * list 2: restricted on chi/peng
 * list 3: restricted on both winTile and chi/peng
 */
std::tuple<resultInfoVec, int, resultInfoVec, int, resultInfoVec, int, resultInfoVec, int> formMinCombination(tileHolderMap handDict, dictHolderVec pack, tileHolderMap tileWallDict, int seatWind = 0, int prevailingWind = 0, int resultThreshold = 45, int maxDist = 7, int targetFanVal = 8, bool disableJueZhang = false);

#endif