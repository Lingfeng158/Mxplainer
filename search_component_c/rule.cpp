#include "rule.h"
#include "tile.h"
#include "shanten.h"
#include "stringify.h"
#include "fan_calculator.h"
#include <stdio.h>
#include <algorithm>
#include <iterator>
#include <string>
#include <sstream>

// typedef std::unordered_map<std::string, int, std::less<std::string>> tileHolderMap;
// typedef std::vector<tileHolderMap> dictHolderVec;

int oneTypeTileCount(std::string tileType, tileHolderMap handDict)
{
    int total = 0;
    for (int i = 1; i < 10; i++)
    {
        std::string key = tileType + std::to_string(i);
        total += handDict[key];
    }
    return total;
}

std::tuple<std::vector<int>, std::vector<int>> oneTypeDistToStraight(std::string tileType, tileHolderMap handDict, tileHolderMap tileWallDict)
{
    std::vector<int> dist2Straight, availTiles2Straight;
    for (int rank = 2; rank < 9; rank++)
    {
        std::string midKey = tileType + std::to_string(rank);
        std::string lowKey = tileType + std::to_string(rank - 1);
        std::string highKey = tileType + std::to_string(rank + 1);
        int midDist = 9;
        if (handDict[midKey] > 0)
        {
            midDist = 0;
        }
        else if (tileWallDict[midKey] > 0)
        {
            midDist = 1;
        }
        int lowDist = 9;
        if (handDict[lowKey] > 0)
        {
            lowDist = 0;
        }
        else if (tileWallDict[lowKey] > 0)
        {
            lowDist = 1;
        }
        int highDist = 9;
        if (handDict[highKey] > 0)
        {
            highDist = 0;
        }
        else if (tileWallDict[highKey] > 0)
        {
            highDist = 1;
        }
        int sumDist = midDist + lowDist + highDist;
        dist2Straight.push_back(sumDist);
        if (sumDist == 0 || sumDist > 3)
        {
            availTiles2Straight.push_back(0);
        }
        else
        {
            int tmpTile = 0;
            if (lowDist != 0)
            {
                tmpTile += tileWallDict[lowKey];
            }
            if (midDist != 0)
            {
                tmpTile += tileWallDict[midKey];
            }
            if (highDist != 0)
            {
                tmpTile += tileWallDict[highKey];
            }
            availTiles2Straight.push_back(tmpTile);
        }
    }
    return std::make_tuple(dist2Straight, availTiles2Straight);
}

std::tuple<int, int> distToStraight(std::string tileType, int rank, tileHolderMap handDict, tileHolderMap tileWallDict)
{
    int dist2Straight, availTiles2Straight;

    std::string midKey = tileType + std::to_string(rank);
    std::string lowKey = tileType + std::to_string(rank - 1);
    std::string highKey = tileType + std::to_string(rank + 1);
    int midDist = 9;
    if (handDict[midKey] > 0)
    {
        midDist = 0;
    }
    else if (tileWallDict[midKey] > 0)
    {
        midDist = 1;
    }
    int lowDist = 9;
    if (handDict[lowKey] > 0)
    {
        lowDist = 0;
    }
    else if (tileWallDict[lowKey] > 0)
    {
        lowDist = 1;
    }
    int highDist = 9;
    if (handDict[highKey] > 0)
    {
        highDist = 0;
    }
    else if (tileWallDict[highKey] > 0)
    {
        highDist = 1;
    }
    int sumDist = midDist + lowDist + highDist;
    dist2Straight = sumDist;
    if (sumDist == 0 || sumDist > 3)
    {
        availTiles2Straight = 0;
    }
    else
    {
        int tmpTile = 0;
        if (lowDist != 0)
        {
            tmpTile += tileWallDict[lowKey];
        }
        if (midDist != 0)
        {
            tmpTile += tileWallDict[midKey];
        }
        if (highDist != 0)
        {
            tmpTile += tileWallDict[highKey];
        }
        availTiles2Straight = tmpTile;
    }

    return std::make_tuple(dist2Straight, availTiles2Straight);
}

std::tuple<std::vector<int>, std::vector<int>> oneTypeDistToTrio(std::string tileType, tileHolderMap handDict, tileHolderMap tileWallDict)
{
    std::vector<int> dist2Trio, availTiles2Trio;
    if (tileType != "X")
    {
        for (int rank = 1; rank < 10; rank++)
        {
            std::string key = tileType + std::to_string(rank);
            int ownedCount = handDict[key];
            int dist = ownedCount + tileWallDict[key] >= 3 ? std ::max(0, 3 - ownedCount) : 9;
            dist2Trio.push_back(dist);
            availTiles2Trio.push_back(tileWallDict[key]);
        }
    }
    else
    {
        std::string enumTileList[] = {"F1", "F2", "F3", "F4", "J1", "J2", "J3"};
        for (std::string key : enumTileList)
        {
            int ownedCount = handDict[key];
            int dist = ownedCount + tileWallDict[key] >= 3 ? std ::max(0, 3 - ownedCount) : 9;
            dist2Trio.push_back(dist);
            availTiles2Trio.push_back(tileWallDict[key]);
        }
    }
    return std::make_tuple(dist2Trio, availTiles2Trio);
}

std::tuple<int, int> distToTrio(std::string tileType, int rank, tileHolderMap handDict, tileHolderMap tileWallDict)
{
    int dist2Trio, availTiles2Trio;
    if (tileType != "X")
    {

        std::string key = tileType + std::to_string(rank);
        int ownedCount = handDict[key];
        int dist = ownedCount + tileWallDict[key] >= 3 ? std ::max(0, 3 - ownedCount) : 9;
        dist2Trio = dist;
        availTiles2Trio = 0;
        if (dist != 0)
        {
            availTiles2Trio += tileWallDict[key];
        }
    }
    else
    {
        std::string enumTileList[] = {"F1", "F2", "F3", "F4", "J1", "J2", "J3"};
        std::string key = enumTileList[rank - 1];
        int ownedCount = handDict[key];
        int dist = ownedCount + tileWallDict[key] >= 3 ? std ::max(0, 3 - ownedCount) : 9;
        dist2Trio = dist;
        availTiles2Trio = 0;
        if (dist != 0)
        {
            availTiles2Trio += tileWallDict[key];
        }
    }
    return std::make_tuple(dist2Trio, availTiles2Trio);
}

std::tuple<std::vector<int>, std::vector<int>> oneTypeDistToDuo(std::string tileType, tileHolderMap handDict, tileHolderMap tileWallDict)
{
    std::vector<int> dist2Duo, availTiles2Duo;
    if (tileType != "X")
    {
        for (int rank = 1; rank < 10; rank++)
        {
            std::string key = tileType + std::to_string(rank);
            int ownedCount = handDict[key];
            int dist = ownedCount + tileWallDict[key] >= 2 ? std ::max(0, 2 - ownedCount) : 9;
            dist2Duo.push_back(dist);
            availTiles2Duo.push_back(tileWallDict[key]);
        }
    }
    else
    {
        std::string enumTileList[] = {"F1", "F2", "F3", "F4", "J1", "J2", "J3"};
        for (std::string key : enumTileList)
        {
            int ownedCount = handDict[key];
            int dist = ownedCount + tileWallDict[key] >= 2 ? std ::max(0, 2 - ownedCount) : 9;
            dist2Duo.push_back(dist);
            availTiles2Duo.push_back(tileWallDict[key]);
        }
    }
    return std::make_tuple(dist2Duo, availTiles2Duo);
}

std::tuple<int, int> distToDuo(std::string tileType, int rank, tileHolderMap handDict, tileHolderMap tileWallDict)
{
    int dist2Duo, availTiles2Duo;
    if (tileType != "X")
    {

        std::string key = tileType + std::to_string(rank);
        int ownedCount = handDict[key];
        int dist = ownedCount + tileWallDict[key] >= 2 ? std ::max(0, 2 - ownedCount) : 9;
        dist2Duo = dist;
        availTiles2Duo = 0;
        if (dist != 0)
        {
            availTiles2Duo += tileWallDict[key];
        }
    }
    else
    {
        std::string enumTileList[] = {"F1", "F2", "F3", "F4", "J1", "J2", "J3"};
        std::string key = enumTileList[rank - 1];
        int ownedCount = handDict[key];
        int dist = ownedCount + tileWallDict[key] >= 2 ? std ::max(0, 2 - ownedCount) : 9;
        dist2Duo = dist;
        availTiles2Duo = 0;
        if (dist != 0)
        {
            availTiles2Duo += tileWallDict[key];
        }
    }
    return std::make_tuple(dist2Duo, availTiles2Duo);
}

std::tuple<int, std::string, int> oneTypeMinDistAndTile(std::string tileType, std::vector<int> dists2Structure, std::vector<int> availTiles2Structure, bool isStraight)
{
    int maxAvail = 0;
    int minDist = 9;
    int index = 0;
    for (int i = 0; i < dists2Structure.size(); i++)
    {
        if (dists2Structure[i] < minDist)
        {
            minDist = dists2Structure[i];
            maxAvail = availTiles2Structure[i];
            index = i;
        }
        else if (dists2Structure[i] == minDist && availTiles2Structure[i] > maxAvail)
        {
            maxAvail = availTiles2Structure[i];
            index = i;
        }
    }
    if (isStraight)
    {
        return std::make_tuple(dists2Structure[index], tileType + std::to_string(index + 2), availTiles2Structure[index]);
    }
    else
    {
        std::string tile;
        if (tileType == "X")
        {
            if (index < 4)
            {
                tile = "F" + std::to_string(index + 1);
            }
            else
            {
                tile = "J" + std::to_string(index - 3);
            }
        }
        else
        {
            tile = tileType + std::to_string(index + 1);
        }
        return std::make_tuple(dists2Structure[index], tile, availTiles2Structure[index]);
    }
}

std::tuple<tileHolderMap, tileHolderMap> tileComposition(tileHolderMap handDict, std::string tileSelection, bool isStraight, bool isDuo)
{
    char tileType = tileSelection[0];
    int tileRank = tileSelection[1] - '0';

    // std::stoi(std::string(tileSelection[1]));
    tileHolderMap tileComposition, tileTarget;
    if (isStraight)
    {
        std::string midKey = tileType + std::to_string(tileRank);
        std::string lowKey = tileType + std::to_string(tileRank - 1);
        std::string highKey = tileType + std::to_string(tileRank + 1);
        tileComposition[lowKey] = handDict[lowKey] > 0 ? 1 : 0;
        tileComposition[midKey] = handDict[midKey] > 0 ? 1 : 0;
        tileComposition[highKey] = handDict[highKey] > 0 ? 1 : 0;
        tileTarget[lowKey] = 1;
        tileTarget[midKey] = 1;
        tileTarget[highKey] = 1;
    }
    else
    {
        if (isDuo)
        {
            int tileCount = std::min(handDict[tileSelection], 2);
            tileComposition[tileSelection] = tileCount;
            tileTarget[tileSelection] = 2;
        }
        else
        {
            int tileCount = std::min(handDict[tileSelection], 3);
            tileComposition[tileSelection] = tileCount;
            tileTarget[tileSelection] = 3;
        }
    }
    return std::make_tuple(tileComposition, tileTarget);
}

tileHolderMap updateTileInfo(tileHolderMap tileWallDict, const tileHolderMap &tileAppearanceDict, const dictHolderVec &packAppearanceDict)
{
    for (const auto &kv : tileAppearanceDict)
    {
        tileWallDict[kv.first] -= kv.second;
    }
    for (const tileHolderMap &mapEntry : packAppearanceDict)
    {
        for (const auto &kv : mapEntry)
        {
            if (kv.first == "AnGang")
                continue;
            tileWallDict[kv.first] -= kv.second;
        }
    }
    return tileWallDict;
}

std::string tileConversion2MahJongGB(tileHolderMap handDict, dictHolderVec packList, std::string winTile)
{
    std::unordered_map<std::string, std::string> conversionMap({{"B", "p"}, {"W", "m"}, {"T", "s"}, {"F1", "E"}, {"F2", "S"}, {"F3", "W"}, {"F4", "N"}, {"J1", "C"}, {"J2", "F"}, {"J3", "P"}});
    std::string ret;
    // pack
    for (tileHolderMap &mapEntry : packList)
    {
        ret += '[';
        for (const auto &kv : mapEntry)
        {
            if (kv.first == "AnGang")
                continue;
            for (int i = 0; i < kv.second; i++)
            {
                if (conversionMap[kv.first] != "")
                {
                    // 东西南北
                    ret += conversionMap[kv.first];
                }
                else
                {
                    ret += kv.first[1];
                    ret += conversionMap[std::string(1, kv.first[0])];
                }
            }
        }
        if (mapEntry["AnGang"] == 0)
        {
            ret += ",1";
        }
        ret += ']';
    }
    // handDict
    for (const auto &kv : handDict)
    {
        if (kv.first == "AnGang")
            continue;
        for (int i = 0; i < kv.second; i++)
        {
            if (conversionMap[kv.first] != "")
            {
                // 东西南北
                ret += conversionMap[kv.first];
            }
            else
            {
                ret += kv.first[1];
                ret += conversionMap[std::string(1, kv.first[0])];
            }
        }
    }
    // winTile
    if (winTile != "")
    {
        if (conversionMap[winTile] != "")
        {
            // 东西南北
            ret += conversionMap[winTile];
        }
        else
        {
            ret += winTile[1];
            ret += conversionMap[std::string(1, winTile[0])];
        }
    }

    return ret;
}

// HERE
std::tuple<int, std::unordered_set<std::string>> calcExactFanWithMahJongGB(tileHolderMap handWallDict, dictHolderVec pack, std::string winTile, bool isLastTile, bool isSelfDrawn, bool is4thTile, bool isKongRelated, int seatWind, int prevailingWind)
{
    mahjong::calculate_param_t param;
    auto handWallDictCp = handWallDict;
    handWallDictCp[winTile] -= 1;
    std::string retStr = tileConversion2MahJongGB(handWallDictCp, pack, winTile);
    long retCode = string_to_tiles(retStr.c_str(), &param.hand_tiles, &param.win_tile);
    if (retCode != PARSE_NO_ERROR)
    {
        printf("error at line %d error = %ld\n", __LINE__, retCode);
    }
    param.flower_count = 0;
    uint8_t win_flag =
        isSelfDrawn * WIN_FLAG_SELF_DRAWN +
        isLastTile * WIN_FLAG_WALL_LAST +
        is4thTile * WIN_FLAG_4TH_TILE +
        isKongRelated * WIN_FLAG_ABOUT_KONG;

    param.win_flag = win_flag;
    param.prevalent_wind = (mahjong::wind_t)prevailingWind;
    param.seat_wind = (mahjong::wind_t)seatWind;
    mahjong::fan_table_t fan_table;
    int points = calculate_fan(&param, &fan_table);
    std::unordered_set<std::string> fanSet;
    for (int i = 0; i < mahjong::FAN_TABLE_SIZE; i++)
    {
        for (int j = 0; j < fan_table[i]; j++)
            fanSet.insert(mahjong::fan_name[i]);
    }
    return std::make_tuple(points, fanSet);
}

bool checkHu(tileHolderMap handDict, dictHolderVec packList, tileHolderMap tileWallDict, std::string winTile, int seatWind, int prevailingWind, bool isSelfDrawn, bool isAboutKong)
{
    bool is4thTile = true;
    if (tileWallDict[winTile] > 0)
        is4thTile = false;
    if (handDict[winTile] > 1)
        is4thTile = false;
    try
    {
        int fanSum;
        std::unordered_set<std::string> fanSet;
        std::tie(fanSum, fanSet) = calcExactFanWithMahJongGB(handDict, packList, winTile, false, isSelfDrawn, is4thTile, isAboutKong, seatWind, prevailingWind);
        if (fanSum >= 8)
            return fanSum;
        else
            return 0;
    }
    catch (int errCode)
    {
        return 0;
    }
}

std::tuple<std::unordered_set<std::string>, std::unordered_set<std::string>> seperateIncompleteSet(tileHolderMap selectedEnc, dictHolderVec targetEnc)
{
    tileHolderMap HandVecCopy = selectedEnc;
    std::unordered_set<std::string> canonicalIncompleteSet, canonicalCompleteSet;
    for (tileHolderMap targ : targetEnc)
    {
        bool setCompleted = true;
        std::vector<std::string> partialTargVec = fromCustomToCanonicalEncoding(targ);
        for (std::string tile : partialTargVec)
        {
            if (HandVecCopy[tile] > 0)
                HandVecCopy[tile] -= 1;
            else
                setCompleted = false;
        }
        if (setCompleted)
            canonicalCompleteSet.insert(partialTargVec.begin(), partialTargVec.end());
        else
            canonicalIncompleteSet.insert(partialTargVec.begin(), partialTargVec.end());
    }
    return std::make_tuple(canonicalCompleteSet, canonicalIncompleteSet);
}

std::vector<std::string> fromCustomToCanonicalEncoding(tileHolderMap custom_enc)
{
    std::vector<std::string> ret;
    for (const auto &kv : custom_enc)
    {
        for (int i = 0; i < kv.second; i++)
        {
            ret.push_back(kv.first);
        }
    }
    return ret;
}

tileHolderMap fromCanonicalToCustomEncoding(std::vector<std::string> can_enc)
{
    tileHolderMap ret;
    for (std::string tile : can_enc)
    {
        ret[tile] += 1;
    }
    return ret;
}

std::tuple<bool, bool, bool, std::unordered_set<std::string>, std::unordered_set<std::string>> calcFanWithMahJongGB(tileHolderMap selectedTileDict, dictHolderVec pack, dictHolderVec targetTileDict, int seatWind, int prevailingWind, int targetFanVal, bool disableJueZhang)
{
    int fanSum = 0;
    std::unordered_set<std::string> lastTileSelection, targetFanType, WFHLastTileSelection;
    std::unordered_set<std::string> canCompleteSet, canIncompleteSet;
    std::tie(canCompleteSet, canIncompleteSet) = seperateIncompleteSet(selectedTileDict, targetTileDict);
    int fanMax = 0;
    int fanMin = 100;
    bool allowChiPeng = true, allowHu = true, containsWFH = false;
    tileHolderMap handDictProposed;
    std::string point64[] = {"四暗刻"};
    std::string point16[] = {"三暗刻"};
    std::string point6[] = {"双暗杠"};
    std::string point4[] = {"不求人"};
    std::string point2[] = {"门前清", "暗杠", "双暗刻"};
    std::string point1[] = {"边张", "嵌张", "单钓将"};
    for (auto &dict : targetTileDict)
    {
        for (auto &kv : dict)
        {
            handDictProposed[kv.first] += kv.second;
        }
    }
    if (canIncompleteSet.size() == 0)
    {
        canIncompleteSet.swap(canCompleteSet);
    }
    for (std::string tile : canIncompleteSet)
    {
        int tmpFan;
        bool isWFH = false;
        std::unordered_set<std::string> tmpFanSet;
        std::tie(tmpFan, tmpFanSet) = calcExactFanWithMahJongGB(handDictProposed, pack, tile, false, false, false, false, seatWind, prevailingWind);
        if (tmpFanSet.find("无番和") != tmpFanSet.end())
        {
            isWFH = true;
            containsWFH = true;
            WFHLastTileSelection.insert(tile);
        }
        std::tie(tmpFan, tmpFanSet) = calcExactFanWithMahJongGB(handDictProposed, pack, tile, false, true, false, false, seatWind, prevailingWind);
        if (disableJueZhang && tmpFanSet.find("和绝张") != tmpFanSet.end())
        {
            tmpFan -= 4;
        }
        fanMax = tmpFan > fanMax ? tmpFan : fanMax;
        fanMin = tmpFan < fanMin ? tmpFan : fanMin;
        if (tmpFan >= targetFanVal)
        {
            lastTileSelection.insert(tile);
            targetFanType.insert(tmpFanSet.begin(), tmpFanSet.end());
        }

        // rectify for chi/peng
        int rectifiedFan = tmpFan;
        for (std::string p64 : point64)
            if (tmpFanSet.find(p64) != tmpFanSet.end())
                rectifiedFan -= 64;
        for (std::string p16 : point16)
            if (tmpFanSet.find(p16) != tmpFanSet.end())
                rectifiedFan -= 16;
        for (std::string p6 : point6)
            if (tmpFanSet.find(p6) != tmpFanSet.end())
                rectifiedFan -= 6;
        for (std::string p4 : point4)
            if (tmpFanSet.find(p4) != tmpFanSet.end())
                rectifiedFan -= 4;
        for (std::string p2 : point2)
            if (tmpFanSet.find(p2) != tmpFanSet.end())
                rectifiedFan -= 2;
        if (tmpFan >= targetFanVal && rectifiedFan < targetFanVal)
            allowChiPeng = false;

        rectifiedFan = tmpFan;
        for (std::string p1 : point1)
            if (tmpFanSet.find(p1) != tmpFanSet.end())
                rectifiedFan -= 1;
        if (rectifiedFan < targetFanVal)
            allowHu = false;
    }
    // for (std::string p6 : point6)
    //     targetFanType.erase(p6);
    // for (std::string p4 : point4)
    //     targetFanType.erase(p4);
    // for (std::string p2 : point2)
    //     targetFanType.erase(p2);
    // for (std::string p1 : point1)
    //     targetFanType.erase(p1);
    if (containsWFH)
        return std::make_tuple(true, true, false, WFHLastTileSelection, std::unordered_set<std::string>({"无番和"}));
    if (fanMin >= targetFanVal && allowChiPeng)
        return std::make_tuple(true, true, true, std::unordered_set<std::string>(), targetFanType);
    if (fanMax < targetFanVal)
        return std::make_tuple(false, false, false, std::unordered_set<std::string>(), std::unordered_set<std::string>());
    return std::make_tuple(true, allowChiPeng, allowHu, lastTileSelection, targetFanType);
}

/**
 * Convert MahJongGB encoding to string
 * tile_value_t @ line 160 "tile.h"
 */
std::string tilefromMahJongGBEnc(int mahjongGBEncoding)
{
    if (mahjongGBEncoding >= 17 && mahjongGBEncoding < 26)
    {
        std::string tileType = "W";
        return tileType + std::to_string(mahjongGBEncoding - 16);
    }
    if (mahjongGBEncoding >= 33 && mahjongGBEncoding < 42)
    {
        std::string tileType = "T";
        return tileType + std::to_string(mahjongGBEncoding - 32);
    }
    if (mahjongGBEncoding >= 49 && mahjongGBEncoding < 58)
    {
        std::string tileType = "B";
        return tileType + std::to_string(mahjongGBEncoding - 48);
    }
    if (mahjongGBEncoding >= 65 && mahjongGBEncoding < 69)
    {
        std::string tileType = "F";
        return tileType + std::to_string(mahjongGBEncoding - 64);
    }
    if (mahjongGBEncoding >= 69 && mahjongGBEncoding < 72)
    {
        std::string tileType = "J";
        return tileType + std::to_string(mahjongGBEncoding - 68);
    }
    return "";
}

std::tuple<int, std::tuple<tileHolderMap, tileHolderMap>, int> oneTypeMinDistToDuo(std::string tileType, tileHolderMap handDict, tileHolderMap tileWallDict)
{
    std::vector<int> tileDist, tileAvail;
    std::tie(tileDist, tileAvail) = oneTypeDistToDuo(tileType, handDict, tileWallDict);
    int duoMinDist, duoAvail;
    std::string duoChoice;
    std::tie(duoMinDist, duoChoice, duoAvail) = oneTypeMinDistAndTile(tileType, tileDist, tileAvail, false);
    return std::make_tuple(duoMinDist, tileComposition(handDict, duoChoice, false, true), duoAvail);
}

std::tuple<int, std::tuple<tileHolderMap, tileHolderMap>, int> oneTypeMinDistToTriplet(std::string tileType, tileHolderMap handDict, tileHolderMap tileWallDict, bool isFengJian)
{
    if (isFengJian)
    {
        std::vector<int> tileDist2Trio, tileAvail2Trio;
        std::tie(tileDist2Trio, tileAvail2Trio) = oneTypeDistToTrio(tileType, handDict, tileWallDict);
        int trioMinDist, trioAvail;
        std::string trioChoice;
        std::tie(trioMinDist, trioChoice, trioAvail) = oneTypeMinDistAndTile(tileType, tileDist2Trio, tileAvail2Trio, false);
        return std::make_tuple(trioMinDist, tileComposition(handDict, trioChoice, false), trioAvail);
    }
    else
    {
        std::vector<int> tileDist2Trio, tileAvail2Trio;
        std::tie(tileDist2Trio, tileAvail2Trio) = oneTypeDistToTrio(tileType, handDict, tileWallDict);
        int trioMinDist, trioAvail;
        std::string trioChoice;
        std::tie(trioMinDist, trioChoice, trioAvail) = oneTypeMinDistAndTile(tileType, tileDist2Trio, tileAvail2Trio, false);

        std::vector<int> tileDist2Straight, tileAvail2Straight;
        std::tie(tileDist2Straight, tileAvail2Straight) = oneTypeDistToStraight(tileType, handDict, tileWallDict);
        int straightMinDist, straightAvail;
        std::string straightChoice;
        std::tie(straightMinDist, straightChoice, straightAvail) = oneTypeMinDistAndTile(tileType, tileDist2Straight, tileAvail2Straight, true);
        if (trioMinDist > straightMinDist)
        {
            return std::make_tuple(straightMinDist, tileComposition(handDict, straightChoice, true), straightAvail);
        }
        else
        {
            return std::make_tuple(trioMinDist, tileComposition(handDict, trioChoice, false), trioAvail);
        }
    }
}

// std::string toString(tileHolderMap inputContainer)
// {
//     std::ostringstream stream;
//     // stream << "\n";
//     for (const auto &kv : inputContainer)
//     {
//         stream << kv.first << ": " << std::to_string(kv.second) << " ";
//     }
//     stream << "\n";
//     return stream.str();
// }

// std::string toString(std::vector<int> inputContainer)
// {
//     std::ostringstream stream;
//     // stream << "\n";
//     for (auto val : inputContainer)
//     {
//         stream << std::to_string(val) << " ";
//     }
//     stream << "\n";
//     return stream.str();
// }

// std::string toString(std::vector<float> inputContainer)
// {
//     std::ostringstream stream;
//     // stream << "\n";
//     for (auto val : inputContainer)
//     {
//         stream << std::to_string(val) << " ";
//     }
//     stream << "\n";
//     return stream.str();
// }

// std::string toString(std::vector<std::string> inputContainer)
// {
//     std::ostringstream stream;
//     // stream << "\n";
//     for (auto val : inputContainer)
//     {
//         stream << val << " ";
//     }
//     stream << "\n";
//     return stream.str();
// }

// std::string toString(dictHolderVec inputContainer)
// {
//     std::ostringstream stream;
//     // stream << "\n";
//     for (auto val : inputContainer)
//     {
//         stream << toString(val);
//     }
//     stream << "\n";
//     return stream.str();
// }

// std::string toString(partialResult inputContainer)
// {
//     std::ostringstream stream;
//     int dist, avail;
//     tileHolderMap selTiles;
//     dictHolderVec targetTiles;
//     std::unordered_set<std::string> lastTileSelection;
//     // stream << "\n";
//     std::tie(dist, selTiles, avail, lastTileSelection, targetTiles) = inputContainer;

//     stream << "Dist " << dist << " \n"
//            << "Selected Tiles: \n"
//            << toString(selTiles)
//            << "Avail: \n"
//            << avail << "\n"
//            << "Target Tiles: \n"
//            << toString(targetTiles);
//     return stream.str();
// }

// std::string toString(resultInfoVec inputContainer)
// {
//     std::ostringstream stream;
//     // stream << "\n";
//     for (auto val : inputContainer)
//     {
//         stream << toString(val);
//     }
//     stream << "\n";
//     return stream.str();
// }

// std::string toString(std::unordered_set<std::string> inputContainer)
// {
//     std::ostringstream stream;
//     // stream << "\n";
//     for (auto val : inputContainer)
//     {
//         stream << val << " ";
//     }
//     stream << "\n";
//     return stream.str();
// }

std::string hashCustomTiles(tileHolderMap customTileDict)
{
    std::string hashedString = "";
    std::vector<std::string> tileOrder{"B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B9", "W1", "W2", "W3", "W4", "W5", "W6", "W7", "W8", "W9", "T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8", "T9", "F1", "F2", "F3", "F4", "J1", "J2", "J3"};
    for (const std::string k : tileOrder)
    {
        hashedString += std::to_string(customTileDict[k]);
    }
    return hashedString;
}

std::string hashCustomVec(dictHolderVec customTileVec)
{
    tileHolderMap flattenedVec;
    for (const tileHolderMap &mapEntry : customTileVec)
    {
        for (const auto &kv : mapEntry)
        {
            if (kv.first == "AnGang")
                continue;
            flattenedVec[kv.first] += kv.second;
        }
    }
    return hashCustomTiles(flattenedVec);
}