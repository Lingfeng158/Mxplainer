#include "special.h"
#include "rule.h"
#include "shanten.h"
#include "tile.h"
#include "stringify.h"
#include <algorithm>
#include <iterator>

partialResult thirteenYao(tileHolderMap handDict, dictHolderVec pack, tileHolderMap tileWallDict)
{
    auto canonicalHandDict = fromCustomToCanonicalEncoding(handDict);
    auto handDictCp = handDict;
    tileHolderMap selectedTiles;
    int availability = 0;
    if (canonicalHandDict.size() < 13)
        return std::make_tuple(13, tileHolderMap(), 0, dictHolderVec());

    std::unordered_set<std::string> thirteenYaoTileRefList{"W1", "W9", "T1", "T9", "B1", "B9", "F1", "F2", "F3", "F4", "J1", "J2", "J3"};
    std::unordered_set<std::string> thirteenYaoTileList{"W1", "W9", "T1", "T9", "B1", "B9", "F1", "F2", "F3", "F4", "J1", "J2", "J3"};
    for (auto &kv : handDictCp)
    {
        auto it = thirteenYaoTileList.find(kv.first);
        if (it != thirteenYaoTileList.end())
        {
            kv.second -= 1;
            thirteenYaoTileList.erase(it);
            if (selectedTiles[kv.first] == 0)
                selectedTiles[kv.first] = 1;
            else
                selectedTiles[kv.first] += 1;
        }
    }
    bool secondAvailable = false;
    std::string additionalTile = "";
    for (auto &kv : handDictCp)
    {
        auto it = thirteenYaoTileRefList.find(kv.first);
        if (it != thirteenYaoTileRefList.end())
        {
            if (kv.second >= 1)
            {
                secondAvailable = true;
                kv.second -= 1;
                selectedTiles[kv.first] += 1;
                additionalTile = kv.first;
                break;
            }
        }
    }

    float maxAvail = 0;
    bool maxAvailIsInHand = false;
    std::string maxAvailTile = "";
    // 单张补全13张， 13面听牌
    for (auto tile : thirteenYaoTileRefList)
    {
        int singleTileAvail = tileWallDict[tile];
        bool isInHand = thirteenYaoTileList.find(tile) == thirteenYaoTileList.end();
        availability += isInHand ? 0 : singleTileAvail;
        float effectiveAvail = isInHand || secondAvailable ? singleTileAvail : singleTileAvail / 2.;
        // 某张牌没了
        if (singleTileAvail == 0 && !isInHand)
        {
            // 某张牌没了，而且手牌也没有
            return std::make_tuple(13, tileHolderMap(), 0, dictHolderVec());
        }

        if (effectiveAvail > maxAvail && !secondAvailable)
        {
            maxAvail = effectiveAvail;
            maxAvailTile = tile;
            maxAvailIsInHand = isInHand;
        }
    }
    if (!secondAvailable)
    {
        if (maxAvailIsInHand)
        {
            availability += maxAvail;
        }
    }

    additionalTile = secondAvailable ? additionalTile : maxAvailTile;
    // int dist = secondAvailable ? 13 - thirteenYaoTileList.size() : 14 - thirteenYaoTileList.size();
    if (additionalTile == "")
        return std::make_tuple(13, tileHolderMap(), 0, dictHolderVec());

    std::vector<std::string> refListVec(thirteenYaoTileRefList.begin(), thirteenYaoTileRefList.end());
    tileHolderMap tmpMap = fromCanonicalToCustomEncoding(refListVec);
    tmpMap[additionalTile] += 1;
    dictHolderVec tmpVec({tmpMap});
    if (secondAvailable)
    {
        int dist = thirteenYaoTileList.size();
        return std::make_tuple(dist, selectedTiles, availability, tmpVec);
    }
    else
    {
        int dist = thirteenYaoTileList.size() + 1;
        return std::make_tuple(dist, selectedTiles, availability, tmpVec);
    }
}

partialResult buKao(tileHolderMap handDict, dictHolderVec pack, tileHolderMap tileWallDict)
{
    std::vector<std::string> canonicalHandDict = fromCustomToCanonicalEncoding(handDict);
    std::unordered_set<std::string> canonicalHandset(canonicalHandDict.begin(), canonicalHandDict.end());
    if (canonicalHandDict.size() < 13)
    {
        return std::make_tuple(13, tileHolderMap(), 0, dictHolderVec());
    }
    // std::map<std::string, int> dragonType({{"B", 1}, {"W", 1}, {"T", 1}});
    std::string dragonTypeRef[] = {"B", "W", "T"};
    std::string dragonType[] = {"B", "W", "T"};
    std::unordered_set<std::string> fjList = {"F1", "F2", "F3", "F4", "J1", "J2", "J3"};
    int dragonList1[] = {1, 4, 7}, dragonList2[] = {2, 5, 8}, dragonList3[] = {3, 6, 9};
    int minDragonDist = 13, maxDragonAvail = 0;
    std::vector<std::string> minDragonReq, minDragonSel, minDragonHandRemain;
    dictHolderVec minDragonTarg;

    for (auto type1 : dragonTypeRef)
    {
        // auto dragonTypeCp = dragonType;
        // dragonTypeCp[type1] = 0;
        std::vector<std::string> dragonTypeCp(std::begin(dragonType), std::end(dragonType));
        auto it = std::remove(dragonTypeCp.begin(), dragonTypeCp.end(), type1);
        dragonTypeCp.erase(it, dragonTypeCp.end());
        for (auto type2 : dragonTypeCp)
        {

            auto type3 = type2 == dragonTypeCp[0] ? dragonTypeCp[1] : dragonTypeCp[0];

            std::unordered_set<std::string> multicoloredDragonTile;
            for (auto rank : dragonList1)
                multicoloredDragonTile.insert(type1 + std::to_string(rank));
            for (auto rank : dragonList2)
                multicoloredDragonTile.insert(type2 + std::to_string(rank));
            for (auto rank : dragonList3)
                multicoloredDragonTile.insert(type3 + std::to_string(rank));
            multicoloredDragonTile.insert(fjList.begin(), fjList.end());
            std::vector<std::string> multiColoredDragonTileRef(multicoloredDragonTile.begin(), multicoloredDragonTile.end());

            // try formed dragon and record distance
            auto handDictCp = handDict;
            std::vector<std::string> selectedTileArr;
            for (auto tile : multiColoredDragonTileRef)
            {
                if (handDictCp[tile] > 0)
                {
                    multicoloredDragonTile.erase(tile);
                    handDictCp[tile] -= 1;
                    selectedTileArr.push_back(tile);
                }
            }
            int dragonDist = multicoloredDragonTile.size();
            int dragonAvail = 0;
            int zeroedTiles = 0;
            for (auto tile : multicoloredDragonTile)
            {
                int avail = tileWallDict[tile];
                dragonAvail += avail;
                if (avail == 0)
                {
                    zeroedTiles += 1;
                }
            }
            if (zeroedTiles > 2)
                continue;
            if (dragonDist < minDragonDist)
            {
                minDragonDist = dragonDist;
                minDragonReq = std::vector<std::string>(multicoloredDragonTile.begin(), multicoloredDragonTile.end());
                minDragonHandRemain = fromCustomToCanonicalEncoding(handDictCp);
                minDragonSel = selectedTileArr;
                maxDragonAvail = dragonAvail;
                minDragonTarg.clear();
                minDragonTarg.push_back(fromCanonicalToCustomEncoding(multiColoredDragonTileRef));
            }
        }
    }
    auto minDragonSelEnc = fromCanonicalToCustomEncoding(minDragonSel);
    return std::make_tuple(minDragonDist - 2, minDragonSelEnc, maxDragonAvail, minDragonTarg);
}

partialResult heptaPairs(tileHolderMap handDict, dictHolderVec pack, tileHolderMap tileWallDict)
{
    tileHolderMap selectedTile;
    dictHolderVec targetTile;
    int formedPairs = 0;
    for (auto &kv : handDict)
    {
        if (kv.second == 4)
        {
            selectedTile[kv.first] = 4;
            targetTile.push_back({{kv.first, 4}});
            kv.second -= 4;
            formedPairs += 2;
        }
        else if (kv.second >= 2)
        {
            selectedTile[kv.first] = 2;
            targetTile.push_back({{kv.first, 2}});
            kv.second -= 2;
            formedPairs += 1;
        }
    }
    auto canonicalHandDict = fromCustomToCanonicalEncoding(handDict);
    if (((int)canonicalHandDict.size()) < 13 - formedPairs * 2)
    {
        return std::make_tuple(13, tileHolderMap(), 0, dictHolderVec());
    }
    std::string tileTypeList[] = {"B", "W", "T", "X"};
    int finalDist = 0, finalAvail = 0;

    std::vector<int> distArr, availArr;
    std::vector<int> indexArr;
    for (int i = 0; i < 34; i++)
    {
        indexArr.push_back(i);
    }
    std::vector<int> rankedIndexList;
    for (auto type : tileTypeList)
    {
        std::vector<int> dist2Duo, availTile2Duo;
        std::tie(dist2Duo, availTile2Duo) = oneTypeDistToDuo(type, handDict, tileWallDict);
        distArr.insert(distArr.end(), dist2Duo.begin(), dist2Duo.end());
        availArr.insert(availArr.end(), availTile2Duo.begin(), availTile2Duo.end());
    }
    std::vector<int> dist0Idx, dist1Avail3Idx, dist1Avail2Idx, dist1Avail1Idx, dist2Avail4Idx, dist2Avail3Idx, dist2Avail2Idx;
    for (int i = 0; i < 34; i++)
    {
        if (distArr[i] == 0)
            dist0Idx.push_back(i);
        if (distArr[i] == 1 && availArr[i] == 3)
            dist1Avail3Idx.push_back(i);
        if (distArr[i] == 1 && availArr[i] == 2)
            dist1Avail2Idx.push_back(i);
        if (distArr[i] == 1 && availArr[i] == 1)
            dist1Avail1Idx.push_back(i);
        if (distArr[i] == 2 && availArr[i] == 4)
            dist2Avail4Idx.push_back(i);
        if (distArr[i] == 2 && availArr[i] == 3)
            dist2Avail3Idx.push_back(i);
        if (distArr[i] == 2 && availArr[i] == 2)
            dist2Avail2Idx.push_back(i);
    }
    rankedIndexList.insert(rankedIndexList.end(), dist0Idx.begin(), dist0Idx.end());
    rankedIndexList.insert(rankedIndexList.end(), dist1Avail3Idx.begin(), dist1Avail3Idx.end());
    rankedIndexList.insert(rankedIndexList.end(), dist1Avail2Idx.begin(), dist1Avail2Idx.end());
    rankedIndexList.insert(rankedIndexList.end(), dist1Avail1Idx.begin(), dist1Avail1Idx.end());
    rankedIndexList.insert(rankedIndexList.end(), dist2Avail4Idx.begin(), dist2Avail4Idx.end());
    rankedIndexList.insert(rankedIndexList.end(), dist2Avail3Idx.begin(), dist2Avail3Idx.end());
    rankedIndexList.insert(rankedIndexList.end(), dist2Avail2Idx.begin(), dist2Avail2Idx.end());
    if ((int)rankedIndexList.size() > 6 - formedPairs)
    {
        for (int i = 0; i < 7 - formedPairs; i++)
        {
            int index = rankedIndexList[i];
            int dist = distArr[index];
            int avail = availArr[index];
            std::string tileType = tileTypeList[index / 9];
            int tileRank = index - 9 * (index / 9);
            if (tileType == "X")
            {
                if (tileRank < 4)
                    tileType = "F";
                else
                {
                    tileType = "J";
                    tileRank -= 4;
                }
            }
            tileRank += 1;
            std::string tile = tileType + std::to_string(tileRank);
            finalDist += dist;
            targetTile.push_back({{tile, 2}});
            if (dist == 0)
            {
                selectedTile[tile] = 2;
                // targetTile[tile]=2;
            }
            else if (dist >= 1 && avail >= dist)
            {
                selectedTile[tile] = 2 - dist;
                // targetTile[tile]=2;
                finalAvail += availArr[index];
            }
            else
            {
                return std::make_tuple(13, tileHolderMap(), 0, dictHolderVec());
            }
        }
    }
    else
    {
        return std::make_tuple(13, tileHolderMap(), 0, dictHolderVec());
    }

    return std::make_tuple(finalDist, selectedTile, finalAvail, targetTile);
}

partialResult multicoloredDragon(tileHolderMap handDict, dictHolderVec pack, tileHolderMap tileWallDict)
{
    std::vector<std::string> canonicalHandArr = fromCustomToCanonicalEncoding(handDict);
    if (canonicalHandArr.size() + pack.size() * 3 >= 13)
    {
        // std::map<std::string, int> dragonType({{"B", 1}, {"W", 1}, {"T", 1}});
        std::string dragonTypeRef[] = {"B", "W", "T"};
        std::string dragonType[] = {"B", "W", "T"};
        int dragonList1[] = {1, 4, 7}, dragonList2[] = {2, 5, 8}, dragonList3[] = {3, 6, 9};
        int minDragonDist = 9, maxDragonAvail = 0;
        std::vector<std::string> minDragonReq, minDragonSel, minDragonHandRemain;
        dictHolderVec minDragonTarg;

        for (auto type1 : dragonTypeRef)
        {
            // auto dragonTypeCp = dragonType;
            // dragonTypeCp[type1] = 0;
            std::vector<std::string> dragonTypeCp(std::begin(dragonType), std::end(dragonType));
            auto it = std::remove(dragonTypeCp.begin(), dragonTypeCp.end(), type1);
            dragonTypeCp.erase(it, dragonTypeCp.end());
            for (auto type2 : dragonTypeCp)
            {

                auto type3 = type2 == dragonTypeCp[0] ? dragonTypeCp[1] : dragonTypeCp[0];

                std::unordered_set<std::string> multicoloredDragonTile;
                for (auto rank : dragonList1)
                    multicoloredDragonTile.insert(type1 + std::to_string(rank));
                for (auto rank : dragonList2)
                    multicoloredDragonTile.insert(type2 + std::to_string(rank));
                for (auto rank : dragonList3)
                    multicoloredDragonTile.insert(type3 + std::to_string(rank));
                std::vector<std::string> multiColoredDragonTileRef(multicoloredDragonTile.begin(), multicoloredDragonTile.end());

                // try formed dragon and record distance
                auto handDictCp = handDict;
                std::vector<std::string> selectedTileArr;
                for (auto tile : multiColoredDragonTileRef)
                {
                    if (handDictCp[tile] > 0)
                    {
                        multicoloredDragonTile.erase(tile);
                        handDictCp[tile] -= 1;
                        selectedTileArr.push_back(tile);
                    }
                }
                int dragonDist = multicoloredDragonTile.size();
                int dragonAvail = 0;

                bool zeroAvailDetection = false;
                for (auto tile : multicoloredDragonTile)
                {
                    int avail = tileWallDict[tile];
                    dragonAvail += avail;
                    if (avail == 0)
                    {
                        zeroAvailDetection = true;
                    }
                }
                if (zeroAvailDetection)
                    continue;
                if (dragonDist < minDragonDist)
                {
                    minDragonDist = dragonDist;
                    minDragonReq = std::vector<std::string>(multicoloredDragonTile.begin(), multicoloredDragonTile.end());
                    minDragonHandRemain = fromCustomToCanonicalEncoding(handDictCp);
                    minDragonSel = selectedTileArr;
                    maxDragonAvail = dragonAvail;
                    minDragonTarg.clear();
                    minDragonTarg.push_back(fromCanonicalToCustomEncoding(multiColoredDragonTileRef));
                }
            }
        }
        // with min dragon information
        // form 3 + 2
        auto tileWallDictCp = tileWallDict;
        tileWallDictCp = updateTileInfo(tileWallDictCp, fromCanonicalToCustomEncoding(minDragonReq));
        if (pack.size() == 1)
        {
            // a triplet is already formed
            int minDuoDist = 2, maxDuoAvail = 0;
            tileHolderMap minDuoSel, minDuoTarg;
            std::string tileTypeList[] = {"B", "W", "T", "X"};
            for (auto tileType : tileTypeList)
            {
                int duoDist, duoAvail;
                std::tuple<tileHolderMap, tileHolderMap> duo_comp;
                std::tie(duoDist, duo_comp, duoAvail) = oneTypeMinDistToDuo(tileType, fromCanonicalToCustomEncoding(minDragonHandRemain), tileWallDictCp);

                if (duoDist < minDuoDist || duoDist == minDuoDist && duoAvail > maxDuoAvail)
                {
                    minDuoDist = duoDist;
                    // tileHolderMap minDuoSel, minDuoTarg;
                    std::tie(minDuoSel, minDuoTarg) = duo_comp;
                    maxDuoAvail = minDuoDist == 0 ? 0 : duoAvail;
                    auto minDuoSelCanonical = fromCustomToCanonicalEncoding(minDuoSel);
                }
            }
            dictHolderVec retTileTarg = minDragonTarg;
            retTileTarg.push_back(minDuoTarg);
            std::vector<std::string> minDragonSelCp = minDragonSel;
            auto minDuoSelCanonical = fromCustomToCanonicalEncoding(minDuoSel);
            minDragonSelCp.insert(minDragonSelCp.end(), minDuoSelCanonical.begin(), minDuoSelCanonical.end());
            return std::make_tuple(minDragonDist + minDuoDist, fromCanonicalToCustomEncoding(minDragonSelCp), maxDragonAvail + maxDuoAvail, retTileTarg);
        }
        else
        {
            // need a triplet and a duo
            int minDuoDist = 2, maxDuoAvail = 0, minTripletDist = 3, maxTripletAvail = 0;
            tileHolderMap minDuoSel, minDuoTarg, minTripletSel, minTripletTarg;
            std::string tileTypeList[] = {"B", "W", "T", "X"};
            for (auto tileType : tileTypeList)
            {
                int tripletDist, tripletAvail;
                std::tuple<tileHolderMap, tileHolderMap> triplet_comp;
                std::tie(tripletDist, triplet_comp, tripletAvail) = oneTypeMinDistToTriplet(tileType, fromCanonicalToCustomEncoding(minDragonHandRemain), tileWallDictCp, tileType == "X");

                if (tripletDist < minTripletDist || tripletDist == minTripletDist && tripletAvail > maxTripletAvail)
                {
                    minTripletDist = tripletDist;
                    // tileHolderMap minTripletSel, minTripletTarg;
                    std::tie(minTripletSel, minTripletTarg) = triplet_comp;
                    maxTripletAvail = minTripletDist == 0 ? 0 : tripletAvail;
                    std::vector<std::string> minTripletSelCanonical = fromCustomToCanonicalEncoding(minTripletSel);
                }
            }
            // update tile list
            // auto tileWallDictCp = tileWallDict;
            tileWallDictCp = updateTileInfo(tileWallDictCp, minTripletTarg);
            // std::vector<std::string> handVecRemain = minDragonHandRemain;
            tileHolderMap handRemain = fromCanonicalToCustomEncoding(minDragonHandRemain);
            for (auto sel : minTripletSel)
            {
                std::string selTile;
                int selCt;
                std::tie(selTile, selCt) = sel;
                handRemain[selTile] -= selCt;
            }
            std::vector<std::string> handDictRemain = fromCustomToCanonicalEncoding(handRemain);
            for (auto tileType : tileTypeList)
            {
                int duoDist, duoAvail;
                std::tuple<tileHolderMap, tileHolderMap> duo_comp;
                std::tie(duoDist, duo_comp, duoAvail) = oneTypeMinDistToDuo(tileType, fromCanonicalToCustomEncoding(handDictRemain), tileWallDictCp);

                if (duoDist < minDuoDist || duoDist == minDuoDist && duoAvail > maxDuoAvail)
                {
                    minDuoDist = duoDist;
                    // tileHolderMap minDuoSel, minDuoTarg;
                    std::tie(minDuoSel, minDuoTarg) = duo_comp;
                    maxDuoAvail = minDuoDist == 0 ? 0 : duoAvail;
                    std::vector<std::string> minDuoSelCanonical = fromCustomToCanonicalEncoding(minDuoSel);
                }
            }
            dictHolderVec retTileTarg = minDragonTarg;
            retTileTarg.push_back(minDuoTarg);
            retTileTarg.push_back(minTripletTarg);
            std::vector<std::string> minDragonSelCp = minDragonSel;
            std::vector<std::string> minDuoSelCanonical = fromCustomToCanonicalEncoding(minDuoSel);
            std::vector<std::string> minTripletSelCanonical = fromCustomToCanonicalEncoding(minTripletSel);
            minDragonSelCp.insert(minDragonSelCp.end(), minDuoSelCanonical.begin(), minDuoSelCanonical.end());
            minDragonSelCp.insert(minDragonSelCp.end(), minTripletSelCanonical.begin(), minTripletSelCanonical.end());
            return std::make_tuple(minDragonDist + minDuoDist + minTripletDist, fromCanonicalToCustomEncoding(minDragonSelCp), maxDragonAvail + maxDuoAvail + maxTripletAvail, retTileTarg);
        }
    }
    else
    {
        return std::make_tuple(13, tileHolderMap(), 0, dictHolderVec());
    }
}