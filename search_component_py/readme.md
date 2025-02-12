 # File Explanation

 ## Custom Encoding

 Encode tiles as X:Y, where X represent 34 tiles, including BTW1-9, F1-4, and J1-3, special case is AnGang:True, which indicate a formed trio (4 tiles in this case) is a AnGang.

 ## rule.py

 A file containing basic functions for Mahjong

 ## generic.py & special.py

 Files containing functions for calculating 上听数 for generic and special 胡牌 combinations

 ## preprocess.py

 A file that contains a framework to translate Botzone log to desired data format. (Framework)

 ## feature.py

 A file that converts instructions from preprocess.py to desired data format. (Implementation)
 
 ## verify_processing_results.py

 A quality assurance executable for preprocess.py
 
 # Goal of preprocessing

 The goal of preprocessing is to generate readily usable data for Mahjong analysis from botzone game records. Specifically, a view of each round's players' hand and tile list are directly presented instead of sequential play records.

 ## Example of Botzone Play Record

Match 61602cb45ddc087351c04367
Wind 3
Player 0 Deal T8 B7 B1 W7 B2 F2 F4 T9 T5 B1 W8 W8 B7
Player 1 Deal B6 T9 T4 B4 F3 J2 T6 B2 F3 T7 T2 J1 B7
Player 2 Deal J1 B5 F1 B2 T9 W3 W9 T1 T5 T3 B1 B8 B6
Player 3 Deal B9 B4 W6 W9 B7 T7 B6 B8 T2 F3 W1 F4 B9
Player 0 Draw T5
Player 0 Play F4
Player 1 Draw B5

## Example of Target Data Format

### botzone_log: [exact copy of botzone_log], pre-preprocessed by 鲁云龙
### tileWall_log: [[tile_wall view of each player *4]*length of game], in dense encoding
### pack_log: [[pack view of each player *4]*length of game], in dense encoding
### handWall_log: [[handWall view of each player *4]*length of game], in dense encoding, length of 
### obsWall_log: [[public viewable tiles of the game]*length of game], in dense encoding
### remaining_tile_log: [[remaining tiles, in int * 4]*length of game]
### botzone_id: [botzone game record id]
### winner_id: [winner id, from 0 ~ 4]
### prevalingWind: [prevalingWind, from 0~4]
### fan_sum: [sum of fan points] 番种总和
### fan_list: [list  of fan types] 番种详解


### Uploading AI to botzone

1. upload .so file to botzone, under /data folder.
2. upload other necessary files excluding .so file as python file to botzone.

### AI Version

1. V1: vanilla bot, with no modification on history tracking and special processin for 全求人
2. V2: intended to work with supervised_learning_v2, in which 全求人 is not accounted, with fixed guard for search-result list-length = 0 and history / pack processing
3. V3: intened to work with supervised_learning_v3, in which 全求人 is specially treated, with fixed guard for search-result list-length = 0 and history / pack processing