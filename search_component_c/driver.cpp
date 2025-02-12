#include "rule.h"
#include "generic.h"
#include <iostream>
#include "fan_calculator.h"
#include "shanten.h"
#include "stringify.h"
#include <sstream>
#include <tuple>
#include <time.h>

using namespace mahjong;

int main()
{

    tileHolderMap handDict = {{"B5", 3}, {"B2", 1}, {"W8", 1}, {"B1", 1}, {"B4", 1}, {"W9", 1}, {"W7", 1}, {"B6", 1}};
    dictHolderVec pack = {{{"T1", 1}, {"T2", 1}, {"T3", 1}}};

    tileHolderMap tileListRaw = {
        {"B1", 1},
        {"B2", 2},
        {"B3", 3},
        {"B4", 2},
        {"B5", 0},
        {"B6", 2},
        {"B7", 1},
        {"B8", 4},
        {"B9", 4},
        {"W1", 2},
        {"W2", 3},
        {"W3", 4},
        {"W4", 3},
        {"W5", 1},
        {"W6", 3},
        {"W7", 2},
        {"W8", 3},
        {"W9", 2},
        {"T1", 2},
        {"T2", 1},
        {"T3", 2},
        {"T4", 2},
        {"T5", 2},
        {"T6", 0},
        {"T7", 1},
        {"T8", 0},
        {"T9", 3},
        {"F1", 3},
        {"F2", 1},
        {"F3", 1},
        {"F4", 2},
        {"J1", 3},
        {"J2", 0},
        {"J3", 2},
    };
    int seatWind = 0;
    int p_wind = 3;
    // int len1, len2, len3, len4;
    // resultInfoVec v1, v2, v3, v4;
    // std::tie(v1, len1, v2, len2, v3, len3, v4, len4) = formMinCombination(handDict, pack, tileListRaw, seatWind, p_wind, 32, 5, 8);
    // std::cout << v1.size() << v2.size() << v3.size() << v4.size();
    // std::map<std::string, partialResultDPVec> lookupDP;

    // // lookupDP = oneTypeAllCombinationDP("B", handDict, tileListRaw, 1, 0, lookupDP);
    // lookupDP = oneTypeLookupDP("W", handDict, pack, tileListRaw);
    // // lookupDP = oneTypeAllCombinationDP("B", handDict, tileListRaw, 2, 0, lookupDP);
    // partialResultDPVec v = lookupDP["WT0D1"];
    // for (auto &kv : lookupDP)
    // {
    //     std::cout << kv.first << ": " << kv.second.size() << std::endl;
    // }
    // for (partialResultDP entry : v)
    // {
    //     auto sel = std::get<1>(entry);
    //     auto targ = std::get<3>(entry);
    //     auto hand = std::get<4>(entry);
    //     auto tile = std::get<5>(entry);
    //     std::cout << std::get<0>(entry) << " " << hashCustomTiles(sel) << " " << hashCustomVec(targ) << " " << hashCustomTiles(hand) << " " << hashCustomTiles(tile) << std::endl;
    //     // std::cout << std::endl;
    // }

    //     // testing oneTypeLookup
    //     // auto ret = oneTypeLookup("W", handDictFinal, pack, tileWallDictTmp);
    //     // for (auto &kv : ret)
    //     // {
    //     //     std::cout << kv.first << std::endl;
    //     //     std::cout << toString(kv.second);
    //     // }

    //     // bool a, b, c;
    //     // std::tie(a, b, c) = calcFanWithMahJongGB(handDictFinal, pack, handDictTargetFinal);
    //     // printf("%d, %d, %d", a, b, c);

    int id0, id1, id2, id3;
    resultInfoVec v0, v1, v2, v3;
    clock_t start, end;
    start = clock();
    long i = 10000000L;
    //     // auto lookup = oneTypeLookup("B", handDictFinal, pack, tileWallDictTmp);
    //     // for(const auto &kv: lookup){
    //     //     printf("%s", kv.first.c_str());
    //     // }
    for (int i = 0; i < 100; ++i)
        std::tie(v0, id0, v1, id1, v2, id2, v3, id3) = formMinCombination(handDict, pack, tileListRaw, seatWind, p_wind, 32, 5, 8);
    end = clock();
    printf("Time elapsed: %f\n", 1.0 * (end - start) / i);
    //     std::ostringstream stream;
    //     // if (v0.size() != 0)
    //     // {
    //     //     stream << toString(v0[id0]);
    //     // }
    //     // if (v1.size() != 0)
    //     // {
    //     //     stream << toString(v1[id1]);
    //     // }
    //     // if (v2.size() != 0)
    //     // {
    //     //     stream << toString(v2[id2]);
    //     // }
    //     // if (v3.size() != 0)
    //     // {
    //     //     stream << toString(v3[id3]);
    //     // }

    //     printf("%ld %ld %ld %ld\n", v0.size(), v1.size(), v2.size(), v3.size());
    //     std::cout << stream.str();

    //     // printf("handDict:\n");
    //     // for (auto &kv : handDict)
    //     // {
    //     //     printf("%s:%d ", kv.first.c_str(), kv.second);
    //     // }

    //     // printf("\ndist: %d, avail: %d\n", dist, avail);
    //     // printf("selTile:\n");
    //     // for (auto &kv : selTiles)
    //     // {
    //     //     printf("%s:%d ", kv.first.c_str(), kv.second);
    //     // }
    //     // printf("\navailDetail:\n");
    //     // for (auto distDetail : availDetail)
    //     // {
    //     //     printf("%f ", distDetail);
    //     // }
    //     // printf("\ntargetTiles:\n");
    //     // for (auto entry : targetTiles)
    //     // {
    //     //     for (auto &kv : entry)
    //     //     {
    //     //         printf("%s:%d ", kv.first.c_str(), kv.second);
    //     //     }
    //     // }
    //     // printf("\n");
    // }
    // // // oneTypeAllCombination() usage

    // // tileHolderMap selectedTiles;
    // // dictHolderVec targetTiles;
    // // int dist, tileAvailability;
    // // std::vector<float> tileAvailabilityDetail;
    // // tileHolderMap tileWallDictCopy = tileWallDict;
    // // std::vector<std::tuple<int, tileHolderMap, int, std::vector<float>, dictHolderVec>> returnArray;

    // // std::tie(returnArray, dist, selectedTiles, tileAvailability, tileAvailabilityDetail, targetTiles) = oneTypeAllCombination("T", handDict, tileWallDict, 2, 1);
    // // printf("dist: %d \n", dist);
    // // for (auto &kv : selectedTiles)
    // // {
    // //     printf("key: %s, value: %d \n", kv.first.c_str(), kv.second);
    // // }
    // // printf("targetTile:\n");
    // // for (tileHolderMap m : targetTiles)
    // // {
    // //     for (auto &kv : m)
    // //     {
    // //         printf("key: %s, value: %d \n", kv.first.c_str(), kv.second);
    // //     }
    // // }

    // // ***************************** SEPERATION *****************************

    // // // find all elements whose value equals to min of list
    // // std::vector<int>::iterator it2Min = std::min_element(dists2Structure.begin(), dists2Structure.end());
    // // std::vector<std::vector<int>::iterator> itArr2Min;

    // // std::for_each(dists2Structure.begin(), dists2Structure.end(), [&](int &i)
    // //               {
    // //         if(i==*it2Min){
    // //             std::vector<int>::iterator tmpIt = dists2Structure.begin() + std::distance(dists2Structure.data(), &i);
    // //             itArr2Min.push_back(tmpIt);
    // //         } });

    // // // check hu
    // // mahjong::calculate_param_t param;

    // // std::string input = "[222p][123m]5s6s78p4sFF9p";
    // // long ret = string_to_tiles(input.c_str(), &param.hand_tiles, &param.win_tile);
    // // if (ret != PARSE_NO_ERROR)
    // // {
    // //     printf("error at line %d error = %ld\n", __LINE__, ret);
    // // }
    // // param.flower_count = 0;
    // // uint8_t win_flag =
    // //     1 * WIN_FLAG_SELF_DRAWN +
    // //     0 * WIN_FLAG_WALL_LAST +
    // //     0 * WIN_FLAG_4TH_TILE +
    // //     0 * WIN_FLAG_ABOUT_KONG;

    // // param.win_flag = win_flag;
    // // param.prevalent_wind = (mahjong::wind_t)1;
    // // param.seat_wind = (mahjong::wind_t)2;
    // // mahjong::fan_table_t fan_table;
    // // int points = calculate_fan(&param, &fan_table);
    // // std::cout << points << std::endl;
    // // std::vector<std::string> fan_holder;
    // // for (int i = 0; i < mahjong::FAN_TABLE_SIZE; i++)
    // // {
    // //     for (int j = 0; j < fan_table[i]; j++)
    // //     {
    // //         fan_holder.push_back(mahjong::fan_name[i]);
    // //     }
    // // }
    // // for (std::string a : fan_holder)
    // //     std::cout << a << std::endl;
}