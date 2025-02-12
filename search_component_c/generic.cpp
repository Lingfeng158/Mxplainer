#include "generic.h"

void oneTypeAllCombinationDP(std::string tileType, tileHolderMap &handDict, tileHolderMap &tileWallDict, int tripleCount, int duoCount, std::map<std::string, partialResultDPVec> &lookupTable, int maxDist)
{
    int pruneCount = 12;
    std::unordered_set<std::string> hashKeySet;
    if (tripleCount == 1 && duoCount == 0)
    {
        // lookupTable[tileType + "T1D0"] = partialResultDPVec();
        partialResultDPVec lookupPage;
        for (int tempRank = 1; tempRank < 17; ++tempRank)
        {
            // pre-declare variables, copy construct handDictCP and tileWallDictCP for future modifications
            tileHolderMap handDictCP = handDict, tileWallDictCP = tileWallDict, tmpTileComposition, tmpTileTarget;
            int rank = tempRank, tmpDist, tmpAvail;

            if (rank < 10)
            {
                // not F or J
                if (tileType != "X")
                {
                    // trios
                    std::tie(tmpDist, tmpAvail) = distToTrio(tileType, rank, handDictCP, tileWallDictCP);
                    std::tie(tmpTileComposition, tmpTileTarget) = tileComposition(handDictCP, tileType + std::to_string(rank), false);
                }
                else
                {
                    // does not work for F or J
                    continue;
                }
            }
            else
            {
                if (tileType != "X")
                {
                    rank -= 8;
                    // straight
                    std::tie(tmpDist, tmpAvail) = distToStraight(tileType, rank, handDictCP, tileWallDictCP);
                    std::tie(tmpTileComposition, tmpTileTarget) = tileComposition(handDictCP, tileType + std::to_string(rank), true);
                }
                else
                {
                    rank -= 9;
                    std::string enumTileList[] = {"F1", "F2", "F3", "F4", "J1", "J2", "J3"};
                    std::string key = enumTileList[rank - 1];
                    std::tie(tmpDist, tmpAvail) = distToTrio(tileType, rank, handDictCP, tileWallDictCP);
                    std::tie(tmpTileComposition, tmpTileTarget) = tileComposition(handDictCP, key, false);
                }
            }
            int passedDist = 0;
            // 减枝, 如果未循环到底部，但距离已经大于已知的min_dist
            if (tmpDist + passedDist > maxDist)
            {
                continue;
            }

            // 减枝未发生，更新手牌信息
            // tmpTileComposition 中只包含手牌中有的牌
            for (auto &kv : tmpTileComposition)
            {
                handDictCP[kv.first] -= kv.second;
            }

            tileHolderMap missingTile;
            for (auto &kv : tmpTileTarget)
            {
                missingTile[kv.first] = kv.second - tmpTileComposition[kv.first];
            }

            // tileWallDictCopyUpdate: 如果选择了（假设中）已选的牌，牌墙信息的变化更新
            tileHolderMap tileWallDictCopyUpdate = updateTileInfo(tileWallDictCP, missingTile);
            dictHolderVec returnTarget = {tmpTileTarget};
            lookupPage.push_back(std::make_tuple(tmpDist, tmpTileComposition, tmpAvail, returnTarget, handDictCP, tileWallDictCopyUpdate));
        }

        // prunning: remove high dist entries if there are more than enough entires
        // skip prunning
        if (lookupPage.size() < pruneCount)
        {
            lookupTable[tileType + "T1D0"] = lookupPage;
            // return lookupTable;
            return;
        }
        // prunning
        std::map<int, int> distDict;
        for (partialResultDP entry : lookupPage)
        {
            distDict[std::get<0>(entry)] += 1;
        }
        int totalEntryCount = 0, cutDist = 9;
        for (int i = 0; i < maxDist; ++i)
        {
            totalEntryCount += distDict[i];
            if (totalEntryCount >= pruneCount)
            {
                cutDist = i;
                break;
            }
        }
        for (partialResultDP entry : lookupPage)
        {
            if (std::get<0>(entry) <= cutDist)
            {
                lookupTable[tileType + "T1D0"].push_back(entry);
            }
        }
        // return lookupTable;
        return;
    }
    else if (tripleCount == 0 && duoCount == 1)
    {
        // create new entry in lookupTable
        // lookupTable[tileType + "T0D1"] = partialResultDPVec();
        partialResultDPVec lookupPage;
        for (int tempRank = 1; tempRank < 10; ++tempRank)
        {
            // pre-declare variables, copy construct handDictCP and tileWallDictCP for future modifications
            tileHolderMap handDictCP = handDict, tileWallDictCP = tileWallDict, tmpTileComposition, tmpTileTarget;
            int rank = tempRank, tmpDist, tmpAvail;
            if (tileType != "X")
            {
                // duo
                std::tie(tmpDist, tmpAvail) = distToDuo(tileType, rank, handDictCP, tileWallDictCP);
                std::tie(tmpTileComposition, tmpTileTarget) = tileComposition(handDictCP, tileType + std::to_string(rank), false, true);
            }
            else
            {
                std::string enumTileList[] = {"F1", "F2", "F3", "F4", "J1", "J2", "J3"};
                if (rank < 8)
                {
                    std::string key = enumTileList[rank - 1];
                    std::tie(tmpDist, tmpAvail) = distToDuo(tileType, rank, handDictCP, tileWallDictCP);
                    std::tie(tmpTileComposition, tmpTileTarget) = tileComposition(handDictCP, key, false, true);
                }
                else
                {
                    continue;
                }
            }
            int passedDist = 0;
            // 减枝, 如果未循环到底部，但距离已经大于已知的min_dist
            if (tmpDist + passedDist > maxDist)
            {
                continue;
            }

            // 减枝未发生，更新手牌信息
            // tmpTileComposition 中只包含手牌中有的牌
            for (auto &kv : tmpTileComposition)
            {
                handDictCP[kv.first] -= kv.second;
            }

            tileHolderMap missingTile;
            for (auto &kv : tmpTileTarget)
            {
                missingTile[kv.first] = kv.second - tmpTileComposition[kv.first];
            }

            // tileWallDictCopyUpdate: 如果选择了（假设中）已选的牌，牌墙信息的变化更新
            tileHolderMap tileWallDictCopyUpdate = updateTileInfo(tileWallDictCP, missingTile);
            dictHolderVec returnTarget = {tmpTileTarget};
            lookupPage.push_back(std::make_tuple(tmpDist, tmpTileComposition, tmpAvail, returnTarget, handDictCP, tileWallDictCopyUpdate));
        }
        // prunning: remove high dist entries if there are more than enough entires
        // skip prunning
        if (lookupPage.size() < pruneCount)
        {
            lookupTable[tileType + "T0D1"] = lookupPage;
            return;
            // return lookupTable;
        }
        // prunning
        std::map<int, int> distDict;
        for (partialResultDP entry : lookupPage)
        {
            distDict[std::get<0>(entry)] += 1;
        }
        int totalEntryCount = 0, cutDist = 9;
        for (int i = 0; i < maxDist; ++i)
        {
            totalEntryCount += distDict[i];
            if (totalEntryCount >= pruneCount)
            {
                cutDist = i;
                break;
            }
        }
        for (partialResultDP entry : lookupPage)
        {
            if (std::get<0>(entry) <= cutDist)
            {
                lookupTable[tileType + "T0D1"].push_back(entry);
            }
        }
        // return lookupTable;
        return;
    }
    else
    {
        // use lookupTable and search triplets
        // load lookupTable entries
        std::string prevKey = tileType + "T" + std::to_string(tripleCount - 1) + "D" + std::to_string(duoCount);
        partialResultDPVec lookupList = lookupTable[prevKey];
        // // create new entry in lookupTable
        // lookupTable[tileType + "T" + std::to_string(tripleCount - 1) + "D" + std::to_string(duoCount)] = partialResultDPVec();
        partialResultDPVec lookupPage;
        for (partialResultDP lookupEntry : lookupList)
        {
            int passedDist, passedAvail;
            tileHolderMap passedCompo, handM, tileM;
            dictHolderVec passedTarg;
            std::tie(passedDist, passedCompo, passedAvail, passedTarg, handM, tileM) = lookupEntry;
            // early stop
            int totalReqTiles = 3;
            int totalHoldTiles = oneTypeTileCount(tileType, handM);
            if (totalReqTiles - totalHoldTiles + passedDist > maxDist)
            {
                continue;
            }

            for (int tempRank = 1; tempRank < 17; ++tempRank)
            {
                tileHolderMap handDictCP = handM, tileWallDictCP = tileM, tmpTileComposition, tmpTileTarget;
                int rank = tempRank, tmpDist, tmpAvail;
                if (rank < 10)
                {
                    // not F or J
                    if (tileType != "X")
                    {
                        // trios
                        std::tie(tmpDist, tmpAvail) = distToTrio(tileType, rank, handDictCP, tileWallDictCP);
                        std::tie(tmpTileComposition, tmpTileTarget) = tileComposition(handDictCP, tileType + std::to_string(rank), false);
                    }
                    else
                    {
                        // does not work for F or J
                        continue;
                    }
                }
                else
                {
                    if (tileType != "X")
                    {
                        rank -= 8;
                        // straight
                        std::tie(tmpDist, tmpAvail) = distToStraight(tileType, rank, handDictCP, tileWallDictCP);
                        std::tie(tmpTileComposition, tmpTileTarget) = tileComposition(handDictCP, tileType + std::to_string(rank), true);
                    }
                    else
                    {
                        rank -= 9;
                        std::string enumTileList[] = {"F1", "F2", "F3", "F4", "J1", "J2", "J3"};
                        std::string key = enumTileList[rank - 1];
                        std::tie(tmpDist, tmpAvail) = distToTrio(tileType, rank, handDictCP, tileWallDictCP);
                        std::tie(tmpTileComposition, tmpTileTarget) = tileComposition(handDictCP, key, false);
                    }
                }
                if (tmpDist + passedDist > maxDist)
                {
                    continue;
                }
                // 减枝未发生，更新手牌信息
                // tmpTileComposition 中只包含手牌中有的牌
                for (auto &kv : tmpTileComposition)
                {
                    handDictCP[kv.first] -= kv.second;
                }

                // calculate selected tiles
                tileHolderMap retTiles = passedCompo;
                for (auto &kv : tmpTileComposition)
                {
                    retTiles[kv.first] += kv.second;
                }
                dictHolderVec retTarg = passedTarg;
                retTarg.push_back(tmpTileTarget);

                // calculate and check for duplicates
                std::string hashedRep = hashCustomVec(retTarg);
                if (hashKeySet.find(hashedRep) == hashKeySet.end())
                {
                    hashKeySet.insert(hashedRep);
                }
                else
                {
                    continue;
                }

                // 计算需更新牌墙信息
                tileHolderMap missingTile;
                for (auto &kv : tmpTileTarget)
                {
                    missingTile[kv.first] = kv.second - tmpTileComposition[kv.first];
                }
                // tileWallDictCopyUpdate: 如果选择了（假设中）已选的牌，牌墙信息的变化更新
                tileHolderMap tileWallDictCopyUpdate = updateTileInfo(tileWallDictCP, missingTile);

                lookupPage.push_back(std::make_tuple(tmpDist + passedDist, retTiles, tmpAvail + passedAvail, retTarg, handDictCP, tileWallDictCopyUpdate));
            }
        }
        if (lookupPage.size() <= pruneCount)
        {
            lookupTable[tileType + "T" + std::to_string(tripleCount) + "D" + std::to_string(duoCount)] = lookupPage;
            // return lookupTable;
            return;
        }
        std::map<int, int> distDict;
        for (partialResultDP entry : lookupPage)
        {
            distDict[std::get<0>(entry)] += 1;
        }
        int totalEntryCount = 0, cutDist = 9;
        for (int i = 0; i < cutDist; ++i)
        {
            totalEntryCount += distDict[i];
            if (totalEntryCount >= pruneCount)
            {
                cutDist = i;
                break;
            }
        }
        for (partialResultDP entry : lookupPage)
        {
            if (std::get<0>(entry) <= cutDist)
            {
                lookupTable[tileType + "T" + std::to_string(tripleCount) + "D" + std::to_string(duoCount)].push_back(entry);
            }
        }
        // return lookupTable;
        return;
    }
}

std::tuple<partialResultVec, int, tileHolderMap, int, dictHolderVec> oneTypeAllCombination(std::string tileType, tileHolderMap handDict, tileHolderMap tileWallDict, int tripleCount, int duoCount, int maxDist, int totalIteration, int passedOnDist)
{
    // calculate total required tiles as record
    int totalReqTiles = tripleCount * 3 + duoCount * 2 - totalIteration * 3;
    // calculate total available tiles as record
    int totalHoldTiles = oneTypeTileCount(tileType, handDict);
    // early break: required-available > maxDist
    if (totalReqTiles - totalHoldTiles > maxDist)
        return std::make_tuple(partialResultVec(), 0, tileHolderMap(), 0, dictHolderVec());

    int minDist = maxDist;
    tileHolderMap selectedTiles;
    dictHolderVec targetTiles;
    int tileAvailability = 0;
    tileHolderMap tileWallDictCopy = tileWallDict;
    partialResultVec returnArray;
    partialResultVec returnArrayDiscard;
    for (int i = 1; i < 10 + 7; i++)
    {
        tileHolderMap handDictCopy = handDict;

        int tmpDist, tmpAvail;
        tileHolderMap tmpTileComposition, tmpTileTarget;
        if (i < 10)
        {
            // specific for BWT's trio and duo
            int rank = i;
            if (tileType != "X")
            {
                // find trios
                if (totalIteration < tripleCount)
                {
                    std::tie(tmpDist, tmpAvail) = distToTrio(tileType, rank, handDictCopy, tileWallDictCopy);
                    std::tie(tmpTileComposition, tmpTileTarget) = tileComposition(handDictCopy, tileType + std::to_string(rank), false);
                }
                else
                {
                    std::tie(tmpDist, tmpAvail) = distToDuo(tileType, rank, handDictCopy, tileWallDictCopy);
                    std::tie(tmpTileComposition, tmpTileTarget) = tileComposition(handDictCopy, tileType + std::to_string(rank), false, true);
                }
            }
            else
                // does not work for F or J
                continue;
        }
        else
        {
            // for BWT's straight (midtile 2-9)
            // and X's trio and duo
            if (tileType != "X")
            {
                // i= 10-16
                // rank = 2-8
                int rank = i - 8;
                if (totalIteration < tripleCount)
                {
                    std::tie(tmpDist, tmpAvail) = distToStraight(tileType, rank, handDictCopy, tileWallDictCopy);
                    std::tie(tmpTileComposition, tmpTileTarget) = tileComposition(handDictCopy, tileType + std::to_string(rank), true);
                }
                else
                    continue;
            }
            else
            {
                // i=10-16
                // rank = 1-7
                int rank = i - 9;
                std::string enumTileList[] = {"F1", "F2", "F3", "F4", "J1", "J2", "J3"};
                std::string key = enumTileList[rank - 1];
                if (totalIteration < tripleCount)
                {
                    std::tie(tmpDist, tmpAvail) = distToTrio(tileType, rank, handDictCopy, tileWallDictCopy);
                    std::tie(tmpTileComposition, tmpTileTarget) = tileComposition(handDictCopy, key, false);
                }
                else
                {
                    std::tie(tmpDist, tmpAvail) = distToDuo(tileType, rank, handDictCopy, tileWallDictCopy);
                    std::tie(tmpTileComposition, tmpTileTarget) = tileComposition(handDictCopy, key, false, true);
                }
            }
        }
        // 减枝, 如果未循环到底部，但距离已经大于已知的min_dist
        if (tmpDist + passedOnDist > maxDist)
            continue;
        // 减枝未发生，更新手牌信息
        // tmpTileComposition 中只包含手牌中有的牌
        for (auto &kv : tmpTileComposition)
        {
            handDictCopy[kv.first] -= kv.second;
        }

        tileHolderMap missingTile;
        for (auto &kv : tmpTileTarget)
        {
            missingTile[kv.first] = kv.second - tmpTileComposition[kv.first];
        }

        // tileWallDictCopyUpdate: 如果选择了（假设中）已选的牌，牌墙信息的变化更新
        tileHolderMap tileWallDictCopyUpdate = updateTileInfo(tileWallDict, missingTile);
        int returnDist, returnAvail;
        tileHolderMap returnTile;
        dictHolderVec returnTarget;

        if (totalIteration + 1 != tripleCount + duoCount)
        {
            // 非终止recusion, 获取下一层信息
            // 因为非最终结果，舍弃returnArray返回项
            // 其余项为下一层中最佳选择
            std::tie(returnArrayDiscard, returnDist, returnTile, returnAvail, returnTarget) = oneTypeAllCombination(tileType, handDictCopy, tileWallDictCopyUpdate, tripleCount, duoCount, maxDist, totalIteration + 1, tmpDist);
        }
        else
        {
            // 终止recursion条件
            std::tie(returnDist, returnTile, returnAvail, returnTarget) = std::make_tuple(0, tileHolderMap(), 0, dictHolderVec());
        }

        int calculatedDist = tmpDist + returnDist;
        if (calculatedDist > maxDist)
            continue;
        else
        {
            // 整合 selected tiles
            for (auto &kv : tmpTileComposition)
                returnTile[kv.first] += kv.second;
            returnTarget.push_back(tmpTileTarget);
            returnAvail += tmpAvail;
            // only construct returnArray at outer most recusion
            if (returnTarget.size() == tripleCount + duoCount)
            {
                returnArray.push_back(std::make_tuple(calculatedDist, returnTile, returnAvail, returnTarget));
            }
        }
        // update minDist etc.
        // select best response for minDist, etc, and return
        if (calculatedDist < minDist || (calculatedDist == minDist && tileAvailability < returnAvail))
        {
            minDist = calculatedDist;
            selectedTiles = returnTile;
            tileAvailability = returnAvail;
            targetTiles = returnTarget;
        }
    }
    return std::make_tuple(returnArray, minDist, selectedTiles, tileAvailability, targetTiles);
}

std::map<std::string, partialResultVec> oneTypeLookup(std::string tileType, tileHolderMap handDict, dictHolderVec pack, tileHolderMap tileWallDict)
{
    std::map<std::string, partialResultVec> lookupDict;
    int maxAllowedDist = 6;
    int entryInPack = pack.size();
    partialResultVec returnArray;
    // int d_d, d_ta;
    // tileHolderMap d_m;
    // dictHolderVec d_v;
    // std::vector<float> d_vf;
    std::tie(returnArray, std::ignore, std::ignore, std::ignore, std::ignore) = oneTypeAllCombination(tileType, handDict, tileWallDict, 0, 1, maxAllowedDist);
    lookupDict[tileType + "T0D0"] = {partialResult()};
    lookupDict[tileType + "T0D1"] = returnArray;
    bool fillDefault0 = false, fillDefault1 = false;
    for (int tripletCt = 1; tripletCt < 5 - entryInPack; tripletCt++)
    {
        // for 清一色
        int maxDist = maxAllowedDist;
        if (tripletCt == 4 - entryInPack)
        {
            int maxDist = maxAllowedDist + 2;
        }
        if (!fillDefault0)
        {
            // std::tie(returnArray, d_d, d_m, d_ta, d_vf, d_v) = oneTypeAllCombination(tileType, handDict, tileWallDict, tripletCt, 0, maxDist);
            std::tie(returnArray, std::ignore, std::ignore, std::ignore, std::ignore) = oneTypeAllCombination(tileType, handDict, tileWallDict, tripletCt, 0, maxDist);
            lookupDict[tileType + "T" + std::to_string(tripletCt) + "D0"] = returnArray;
            if (returnArray.size() == 0)
                fillDefault0 = true;
        }
        else
            lookupDict[tileType + "T" + std::to_string(tripletCt) + "D0"] = partialResultVec();

        if (!fillDefault1)
        {
            std::tie(returnArray, std::ignore, std::ignore, std::ignore, std::ignore) = oneTypeAllCombination(tileType, handDict, tileWallDict, tripletCt, 1, maxDist);
            lookupDict[tileType + "T" + std::to_string(tripletCt) + "D1"] = returnArray;
            if (returnArray.size() == 0)
                fillDefault1 = true;
        }
        else
            lookupDict[tileType + "T" + std::to_string(tripletCt) + "D1"] = partialResultVec();
    }

    return lookupDict;
}

std::map<std::string, partialResultDPVec> oneTypeLookupDP(std::string tileType, tileHolderMap handDict, dictHolderVec pack, tileHolderMap tileWallDict)
{
    int entryInPackCount = pack.size();
    std::map<int, int> maxAllowedDistDict = {{2, 1}, {3, 2}, {5, 3}, {6, 4}, {8, 6}, {9, 6}, {11, 5}, {12, 6}, {14, 6}};
    std::map<std::string, partialResultDPVec> lookupTable;
    lookupTable[tileType + "T0D0"].push_back(partialResultDP());
    bool fillDefault0 = false;
    bool fillDefault1 = false;
    for (int tripleCount = 0; tripleCount < 5 - entryInPackCount; ++tripleCount)
    {
        if (tripleCount != 0)
        {
            if (!fillDefault0)
            {
                oneTypeAllCombinationDP(tileType, handDict, tileWallDict, tripleCount, 0, lookupTable, maxAllowedDistDict[tripleCount * 3]);
                if (lookupTable[tileType + "T" + std::to_string(tripleCount) + "D0"].size() == 0)
                {
                    fillDefault0 = true;
                }
            }
            else
            {
                lookupTable[tileType + "T" + std::to_string(tripleCount) + "D0"] = partialResultDPVec();
            }
        }
        if (fillDefault1 == false)
        {
            oneTypeAllCombinationDP(tileType, handDict, tileWallDict, tripleCount, 1, lookupTable, maxAllowedDistDict[tripleCount * 3 + 2]);
            if (lookupTable[tileType + "T" + std::to_string(tripleCount) + "D1"].size() == 0)
            {
                fillDefault1 = true;
            }
        }
        else
        {
            lookupTable[tileType + "T" + std::to_string(tripleCount) + "D1"] = partialResultDPVec();
        }
    }
    return lookupTable;
}

std::tuple<resultInfoVec, int, resultInfoVec, int, resultInfoVec, int, resultInfoVec, int> formMinCombination(tileHolderMap handDict, dictHolderVec pack, tileHolderMap tileWallDict, int seatWind, int prevailingWind, int resultThreshold, int maxDist, int targetFanVal, bool disableJueZhang)
{
    int countAvailable = 0;
    std::vector<std::map<std::string, partialResultDPVec>> lookupList;
    int entryInPack = pack.size();
    resultInfoVec lv0List, lv1List, lv2List, lv3List;
    int lv0MinDist = 10, lv1MinDist = 10, lv2MinDist = 10, lv3MinDist = 10, lv0MaxAvail = 0, lv1MaxAvail = 0, lv2MaxAvail = 0, lv3MaxAvail = 0;
    int lv0id = -1, lv1id = -1, lv2id = -1, lv3id = -1;
    std::string tileTypeList[] = {"B", "W", "T", "X"};
    for (std::string t : tileTypeList)
    {
        auto lookupDict = oneTypeLookupDP(t, handDict, pack, tileWallDict);
        lookupList.push_back(lookupDict);
    }
    for (int tryDist = 0; tryDist < maxDist + 1; tryDist++)
    {
        for (const auto &b_kv : lookupList[0])
        {
            for (const auto &w_kv : lookupList[1])
            {
                for (const auto &t_kv : lookupList[2])
                {
                    for (const auto &x_kv : lookupList[3])
                    {
                        int b_tripleCt = b_kv.first[2] - '0';
                        int w_tripleCt = w_kv.first[2] - '0';
                        int t_tripleCt = t_kv.first[2] - '0';
                        int x_tripleCt = x_kv.first[2] - '0';
                        int b_duoCt = b_kv.first[4] - '0';
                        int w_duoCt = w_kv.first[4] - '0';
                        int t_duoCt = t_kv.first[4] - '0';
                        int x_duoCt = x_kv.first[4] - '0';
                        if (b_tripleCt + w_tripleCt + t_tripleCt + x_tripleCt == 4 - entryInPack &&
                            b_duoCt + w_duoCt + t_duoCt + x_duoCt == 1)
                        {

                            auto b = b_kv.second;
                            auto w = w_kv.second;
                            auto t = t_kv.second;
                            auto x = x_kv.second;

                            // DEBUG
                            // printf("BT%dD%d, WT%dD%d,TT%dD%d,XT%dD%d\n", b_tripleCt, b_duoCt, w_tripleCt, w_duoCt, t_tripleCt, t_duoCt, x_tripleCt, x_duoCt);
                            // if (b_kv.first == "BT1D1" && w_kv.first == "WT1D0" && t_kv.first == "TT2D0" && x_kv.first == "XT0D0" && tryDist == 2)
                            // {
                            //     printf("B:\n%s\n\n W:\n%s\n\n T:\n%s \n\n", toString(b).c_str(), toString(w).c_str(), toString(t).c_str());
                            // }

                            if (b.size() == 0 || w.size() == 0 || t.size() == 0 || x.size() == 0)
                            {
                                continue;
                            }
                            if (countAvailable >= resultThreshold)
                            {
                                break;
                            }

                            // form union
                            for (const auto &b_content : b)
                            {
                                for (const auto &w_content : w)
                                {
                                    for (const auto &t_content : t)
                                    {
                                        for (const auto &x_content : x)
                                        {
                                            int dist = std::get<0>(b_content) + std::get<0>(w_content) + std::get<0>(t_content) + std::get<0>(x_content);
                                            if (dist == tryDist)
                                            {
                                                int availability = std::get<2>(b_content) + std::get<2>(w_content) + std::get<2>(t_content) + std::get<2>(x_content);

                                                tileHolderMap tiles;
                                                for (const auto &kv : std::get<1>(b_content))
                                                {
                                                    tiles[kv.first] += kv.second;
                                                }
                                                for (const auto &kv : std::get<1>(w_content))
                                                {
                                                    tiles[kv.first] += kv.second;
                                                }
                                                for (const auto &kv : std::get<1>(t_content))
                                                {
                                                    tiles[kv.first] += kv.second;
                                                }
                                                for (const auto &kv : std::get<1>(x_content))
                                                {
                                                    tiles[kv.first] += kv.second;
                                                }

                                                dictHolderVec plannedTiles;
                                                plannedTiles.insert(plannedTiles.end(), std::get<3>(b_content).begin(), std::get<3>(b_content).end());
                                                plannedTiles.insert(plannedTiles.end(), std::get<3>(w_content).begin(), std::get<3>(w_content).end());
                                                plannedTiles.insert(plannedTiles.end(), std::get<3>(t_content).begin(), std::get<3>(t_content).end());
                                                plannedTiles.insert(plannedTiles.end(), std::get<3>(x_content).begin(), std::get<3>(x_content).end());

                                                /// HERE
                                                bool validity, allowChiPeng, allowFreeWinTile;
                                                std::unordered_set<std::string> lastTileSelection, targetFanType;
                                                std::tie(validity, allowChiPeng, allowFreeWinTile, lastTileSelection, targetFanType) = calcFanWithMahJongGB(tiles, pack, plannedTiles, seatWind, prevailingWind, targetFanVal, disableJueZhang);
                                                if (!validity)
                                                    continue;
                                                if (validity && allowChiPeng && allowFreeWinTile)
                                                {
                                                    countAvailable += 1;
                                                    lv0List.push_back(std::make_tuple(dist, tiles, availability, lastTileSelection, plannedTiles, targetFanType));

                                                    if (dist < lv0MinDist || (dist == lv0MinDist && availability > lv0MaxAvail))
                                                    {
                                                        lv0MaxAvail = availability;
                                                        lv0MinDist = dist;
                                                        lv0id = lv0List.size() - 1;
                                                    }
                                                    // return std::make_tuple(lv0List, lv0id, lv1List, lv1id, lv2List, lv2id, lv3List, lv3id);
                                                }
                                                if (validity && allowChiPeng && !allowFreeWinTile)
                                                {
                                                    countAvailable += 1;
                                                    lv1List.push_back(std::make_tuple(dist, tiles, availability, lastTileSelection, plannedTiles, targetFanType));
                                                    if (dist < lv1MinDist || (dist == lv1MinDist && availability > lv1MaxAvail))
                                                    {
                                                        lv1MaxAvail = availability;
                                                        lv1MinDist = dist;
                                                        lv1id = lv1List.size() - 1;
                                                    }
                                                }
                                                if (validity && !allowChiPeng && allowFreeWinTile)
                                                {
                                                    countAvailable += 1;
                                                    lv2List.push_back(std::make_tuple(dist, tiles, availability, lastTileSelection, plannedTiles, targetFanType));
                                                    if (dist < lv2MinDist || (dist == lv2MinDist && availability > lv2MaxAvail))
                                                    {
                                                        lv2MaxAvail = availability;
                                                        lv2MinDist = dist;
                                                        lv2id = lv2List.size() - 1;
                                                    }
                                                }
                                                if (validity && !allowChiPeng && !allowFreeWinTile)
                                                {
                                                    countAvailable += 1;
                                                    lv3List.push_back(std::make_tuple(dist, tiles, availability, lastTileSelection, plannedTiles, targetFanType));
                                                    if (dist < lv3MinDist || (dist == lv3MinDist && availability > lv3MaxAvail))
                                                    {
                                                        lv3MaxAvail = availability;
                                                        lv3MinDist = dist;
                                                        lv3id = lv3List.size() - 1;
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        if (countAvailable > resultThreshold)
            break;
    }

    int dist, avail;
    tileHolderMap selTiles;
    std::unordered_set<std::string> lastTileSelection, targetFanType;
    dictHolderVec targetTiles;
    std::tie(dist, selTiles, avail, targetTiles) = buKao(handDict, pack, tileWallDict);
    targetFanType.clear();
    targetFanType.insert("全不靠");
    if (dist <= maxDist)
    {
        lv2List.push_back(std::make_tuple(dist, selTiles, avail, lastTileSelection, targetTiles, targetFanType));
        if (dist < lv2MinDist || dist == lv2MinDist && avail > lv2MaxAvail)
        {
            lv2MaxAvail = avail;
            lv2MinDist = dist;
            lv2id = lv2List.size() - 1;
        }
    }
    std::tie(dist, selTiles, avail, targetTiles) = heptaPairs(handDict, pack, tileWallDict);
    targetFanType.clear();
    targetFanType.insert("七对");
    if (dist <= maxDist)
    {
        lv2List.push_back(std::make_tuple(dist, selTiles, avail, lastTileSelection, targetTiles, targetFanType));
        if (dist < lv2MinDist || dist == lv2MinDist && avail > lv2MaxAvail)
        {
            lv2MaxAvail = avail;
            lv2MinDist = dist;
            lv2id = lv2List.size() - 1;
        }
    }
    std::tie(dist, selTiles, avail, targetTiles) = thirteenYao(handDict, pack, tileWallDict);
    targetFanType.clear();
    targetFanType.insert("十三幺");
    if (dist <= maxDist)
    {
        lv2List.push_back(std::make_tuple(dist, selTiles, avail, lastTileSelection, targetTiles, targetFanType));
        if (dist < lv2MinDist || dist == lv2MinDist && avail > lv2MaxAvail)
        {
            lv2MaxAvail = avail;
            lv2MinDist = dist;
            lv2id = lv2List.size() - 1;
        }
    }
    std::tie(dist, selTiles, avail, targetTiles) = multicoloredDragon(handDict, pack, tileWallDict);
    targetFanType.clear();
    targetFanType.insert("组合龙");
    if (dist <= maxDist)
    {
        lv0List.push_back(std::make_tuple(dist, selTiles, avail, lastTileSelection, targetTiles, targetFanType));
        if (dist < lv0MinDist || dist == lv0MinDist && avail > lv0MaxAvail)
        {
            lv0MaxAvail = avail;
            lv0MinDist = dist;
            lv0id = lv0List.size() - 1;
        }
    }
    return std::make_tuple(lv0List, lv0id, lv1List, lv1id, lv2List, lv2id, lv3List, lv3id);
}