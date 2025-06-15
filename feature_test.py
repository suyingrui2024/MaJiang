from agent import MahjongGBAgent
from collections import defaultdict
import numpy as np

try:
    from MahjongGB import MahjongFanCalculator
except:
    print('MahjongGB library required! Please visit https://gitHUb.com/ailab-pku/PyMahjongGB for more information.')
    raise

class FeatureAgent(MahjongGBAgent):
    
    '''
    observation: 6*4*9
        (men+quan+hand4)*4*9
    action_mask: 235
        PASS1+HU1+discard34+CHI63(3*7*3)+PENG34+GANG34+ANGANG34+BUGANG34
    '''
    
    OBS_SIZE = 6
    ACT_SIZE = 235
    
    OFFSET_OBS = {
        'SEAT_WIND' : 0,
        'PREVALENT_WIND' : 1,
        'HAND' : 2
    }
    OFFSET_ACT = {
        'PASS' : 0,
        'HU' : 1,
        'PLAY' : 2,
        'CHI' : 36,
        'PENG' : 99,
        'GANG' : 133,
        'ANGANG' : 167,
        'BUGANG' : 201
    }
    TILE_LIST = [
        *('W%d'%(i+1) for i in range(9)),
        *('T%d'%(i+1) for i in range(9)),
        *('B%d'%(i+1) for i in range(9)),
        *('F%d'%(i+1) for i in range(4)),
        *('J%d'%(i+1) for i in range(3))
    ]
    OFFSET_TILE = {c : i for i, c in enumerate(TILE_LIST)}
    
    def __init__(self, seatWind):
        self.seatWind = seatWind
        self.packs = [[] for i in range(4)]
        self.history = [[] for i in range(4)]
        self.tileWall = [21] * 4
        self.shownTiles = defaultdict(int)
        self.wallLast = False
        self.isAboutKong = False
        self.obs = np.zeros((self.OBS_SIZE, 36))
        self.obs[self.OFFSET_OBS['SEAT_WIND']][self.OFFSET_TILE['F%d' % (self.seatWind + 1)]] = 1

        self.last_fanshu = 0
        self.fanshu =0
    
    '''
    Wind 0..3
    Deal XX XX ...
    Player N Draw
    Player N GANG
    Player N(me) ANGANG XX
    Player N(me) Play XX
    Player N(me) BUGANG XX
    Player N(not me) PENG
    Player N(not me) CHI XX
    Player N(not me) ANGANG
    
    Player N HU
    HUang
    Player N Invalid
    Draw XX
    Player N(not me) Play XX
    Player N(not me) BUGANG XX
    Player N(me) PENG
    Player N(me) CHI XX
    '''
    def request2obs(self, request):
        t = request.split()
        if t[0] == '0':
            self.prevalentWind = int(t[1])
            self.obs[self.OFFSET_OBS['PREVALENT_WIND']][self.OFFSET_TILE['F%d' % (self.prevalentWind + 1)]] = 1
            return
        if t[0] == '1':
            self.hand = t[1:]
            self._hand_embedding_update()

            #更新番数
            t = self.last_fanshu
            self.last_fanshu = self.calculate_fanshu()
            if not self.last_fanshu:
                self.last_fanshu =t

            return
        if t[0] == 'HUANG':
            self.valid = []
            return self._obs()
        if t[0] == '2':
            # Available: HU, Play, ANGANG, BUGANG
            self.tileWall[0] -= 1
            self.wallLast = self.tileWall[1] == 0
            tile = t[1]
            self.valid = []
            if self._check_mahjong(tile, isSelfDrawn = True, isAboutKong = self.isAboutKong):
                self.valid.append(self.OFFSET_ACT['HU'])
            self.isAboutKong = False
            self.hand.append(tile)

            # 更新番数，计算新的番数
            self.last_fanshu = self.fanshu
            self.fanshu = self.calculate_fanshu(tile, isSelfDrawn = True, isAboutKong = self.isAboutKong)
            if not self.fanshu:
                self.fanshu = self.last_fanshu


            self._hand_embedding_update()
            for tile in set(self.hand):
                self.valid.append(self.OFFSET_ACT['PLAY'] + self.OFFSET_TILE[tile])
                if self.hand.count(tile) == 4 and not self.wallLast and self.tileWall[0] > 0:
                    self.valid.append(self.OFFSET_ACT['ANGANG'] + self.OFFSET_TILE[tile])
            if not self.wallLast and self.tileWall[0] > 0:
                for packType, tile, offer in self.packs[0]:
                    if packType == 'PENG' and tile in self.hand:
                        self.valid.append(self.OFFSET_ACT['BUGANG'] + self.OFFSET_TILE[tile])

        
            return self._obs()
        # Player N Invalid/HU/Draw/Play/CHI/PENG/GANG/ANGANG/BUGANG XX
        p = (int(t[1]) + 4 - self.seatWind) % 4

        # 如果这个动作不是自己操作，自己的番数不会发生改变，于是只需要将番数设为相同就行
        if p != 0:
            self.last_fanshu = self.fanshu



        if t[2] == 'BUHUA':
            self.tileWall[p] -= 1
            self.wallLast = self.tileWall[(p + 1) % 4] == 0
            return 
        if t[2] == 'DRAW':
            self.tileWall[p] -= 1
            self.wallLast = self.tileWall[(p + 1) % 4] == 0
            return
        if t[2] == 'Invalid':
            self.valid = []
            return self._obs()
        if t[2] == 'HU':
            self.valid = []
            return self._obs()
        if t[2] == 'PLAY':
            self.tileFrom = p
            self.curTile = t[3]
            self.shownTiles[self.curTile] += 1
            self.history[p].append(self.curTile)
            if p == 0:
                self.hand.remove(self.curTile)
                self._hand_embedding_update()

                # 计算番数，更新番数
                self.last_fanshu = self.fanshu
                self.fanshu = self.calculate_fanshu(None, isSelfDrawn = None, isAboutKong = self.isAboutKong)
                if not self.fanshu:
                    self.fanshu = self.last_fanshu
                return
            else:
                # Available: HU/GANG/PENG/CHI/PASS
                self.valid = []
                if self._check_mahjong(self.curTile):
                    self.valid.append(self.OFFSET_ACT['HU'])
                if not self.wallLast:
                    if self.hand.count(self.curTile) >= 2:
                        self.valid.append(self.OFFSET_ACT['PENG'] + self.OFFSET_TILE[self.curTile])
                        if self.hand.count(self.curTile) == 3 and self.tileWall[0]:
                            self.valid.append(self.OFFSET_ACT['GANG'] + self.OFFSET_TILE[self.curTile])
                    color = self.curTile[0]
                    if p == 3 and color in 'WTB':
                        num = int(self.curTile[1])
                        tmp = []
                        for i in range(-2, 3): tmp.append(color + str(num + i))
                        if tmp[0] in self.hand and tmp[1] in self.hand:
                            self.valid.append(self.OFFSET_ACT['CHI'] + 'WTB'.index(color) * 21 + (num - 3) * 3 + 2)
                        if tmp[1] in self.hand and tmp[3] in self.hand:
                            self.valid.append(self.OFFSET_ACT['CHI'] + 'WTB'.index(color) * 21 + (num - 2) * 3 + 1)
                        if tmp[3] in self.hand and tmp[4] in self.hand:
                            self.valid.append(self.OFFSET_ACT['CHI'] + 'WTB'.index(color) * 21 + (num - 1) * 3)
                self.valid.append(self.OFFSET_ACT['PASS'])
                return self._obs()
       
        if t[2] == 'CHI':
            tile = t[3]
            color = tile[0]
            num = int(tile[1])
            self.packs[p].append(('CHI', tile, int(self.curTile[1]) - num + 2))
            self.shownTiles[self.curTile] -= 1
            for i in range(-1, 2):
                self.shownTiles[color + str(num + i)] += 1
            self.wallLast = self.tileWall[(p + 1) % 4] == 0
            if p == 0:
                # Available: PLAY
                self.valid = []
                self.hand.append(self.curTile)
                for i in range(-1, 2):
                    self.hand.remove(color + str(num + i))
                self._hand_embedding_update()
                for tile in set(self.hand):
                    self.valid.append(self.OFFSET_ACT['PLAY'] + self.OFFSET_TILE[tile])
                
                # 更新番数
                self.last_fanshu = self.fanshu
                self.fanshu = self.calculate_fanshu(None, isSelfDrawn = None, isAboutKong = self.isAboutKong)
                if not self.fanshu:
                    self.fanshu = self.last_fanshu

                return self._obs()
            else:
                return
        if t[2] == 'UnCHI':
            tile = t[3]
            color = tile[0]
            num = int(tile[1])
            self.packs[p].pop()
            self.shownTiles[self.curTile] += 1
            for i in range(-1, 2):
                self.shownTiles[color + str(num + i)] -= 1
            if p == 0:
                for i in range(-1, 2):
                    self.hand.append(color + str(num + i))
                self.hand.remove(self.curTile)
                self._hand_embedding_update()
            return
        if t[2] == 'PENG':
            self.packs[p].append(('PENG', self.curTile, (4 + p - self.tileFrom) % 4))
            self.shownTiles[self.curTile] += 2
            self.wallLast = self.tileWall[(p + 1) % 4] == 0
            if p == 0:
                # Available: Play
                self.valid = []
                for i in range(2):

                    print(self.hand)
                    print(self.curTile)
                    self.hand.remove(self.curTile)
                self._hand_embedding_update()

                # 更新番数
                self.last_fanshu = self.fanshu
                self.fanshu = self.calculate_fanshu(None, isSelfDrawn = None, isAboutKong = self.isAboutKong)
                if not self.fanshu:
                    self.fanshu = self.last_fanshu

                for tile in set(self.hand):
                    self.valid.append(self.OFFSET_ACT['PLAY'] + self.OFFSET_TILE[tile])
                return self._obs()
            else:
                return
        if t[2] == 'UnPENG':
            self.packs[p].pop()
            self.shownTiles[self.curTile] -= 2
            if p == 0:
                for i in range(2):
                    self.hand.append(self.curTile)
                self._hand_embedding_update()
            return
        if t[2] == 'GANG':
            self.packs[p].append(('GANG', self.curTile, (4 + p - self.tileFrom) % 4))
            self.shownTiles[self.curTile] += 3
            if p == 0:
                for i in range(3):
                    self.hand.remove(self.curTile)
                self._hand_embedding_update()
                self.isAboutKong = True

                # 更新番数
                self.last_fanshu = self.fanshu
                self.fanshu = self.calculate_fanshu(None, isSelfDrawn = None, isAboutKong = self.isAboutKong)
                if not self.fanshu:
                    self.fanshu = self.last_fanshu
            return
        if t[2] == 'ANGANG':
            tile = 'CONCEALED' if p else t[3]
            self.packs[p].append(('GANG', tile, 0))
            if p == 0:
                self.isAboutKong = True
                for i in range(4):
                    self.hand.remove(tile)

                # 更新番数
                self.last_fanshu = self.fanshu
                self.fanshu = self.calculate_fanshu(None, isSelfDrawn = None, isAboutKong = self.isAboutKong)
                if not self.fanshu:
                    self.fanshu = self.last_fanshu
                
            else:
                self.isAboutKong = False
            return
        if t[2] == 'BUGANG':
            tile = t[3]
            for i in range(len(self.packs[p])):
                if tile == self.packs[p][i][1]:
                    self.packs[p][i] = ('GANG', tile, self.packs[p][i][2])
                    break
            self.shownTiles[tile] += 1
            if p == 0:
                self.hand.remove(tile)
                self._hand_embedding_update()
                self.isAboutKong = True
                # 更新番数
                self.last_fanshu = self.fanshu
                self.fanshu = self.calculate_fanshu(None, isSelfDrawn = None, isAboutKong = self.isAboutKong)
                if not self.fanshu:
                    self.fanshu = self.last_fanshu
                return
            else:
                # Available: HU/PASS
                self.valid = []
                if self._check_mahjong(tile, isSelfDrawn = False, isAboutKong = True):
                    self.valid.append(self.OFFSET_ACT['HU'])
                self.valid.append(self.OFFSET_ACT['PASS'])
                return self._obs()
        raise NotImplementedError('Unknown request %s!' % request)
    
    '''
    PASS
    HU
    Play XX
    CHI XX
    PENG
    GANG
    (An)GANG XX
    BUGANG XX
    '''
    def action2response(self, action):
        if action < self.OFFSET_ACT['HU']:
            return 'PASS'
        if action < self.OFFSET_ACT['PLAY']:
            return 'HU'
        if action < self.OFFSET_ACT['CHI']:
            return 'Play ' + self.TILE_LIST[action - self.OFFSET_ACT['PLAY']]
        if action < self.OFFSET_ACT['PENG']:
            t = (action - self.OFFSET_ACT['CHI']) // 3
            return 'CHI ' + 'WTB'[t // 7] + str(t % 7 + 2)
        if action < self.OFFSET_ACT['GANG']:
            return 'PENG'
        if action < self.OFFSET_ACT['ANGANG']:
            return 'GANG'
        if action < self.OFFSET_ACT['BUGANG']:
            return 'GANG ' + self.TILE_LIST[action - self.OFFSET_ACT['ANGANG']]
        return 'BUGANG ' + self.TILE_LIST[action - self.OFFSET_ACT['BUGANG']]
    
    '''
    PASS
    HU
    Play XX
    CHI XX
    PENG
    GANG
    (An)GANG XX
    BUGANG XX
    '''
    def response2action(self, response):
        t = response.split()
        if t[0] == 'PASS': return self.OFFSET_ACT['PASS']
        if t[0] == 'HU': return self.OFFSET_ACT['HU']
        if t[0] == 'Play': return self.OFFSET_ACT['PLAY'] + self.OFFSET_TILE[t[1]]
        if t[0] == 'CHI': return self.OFFSET_ACT['CHI'] + 'WTB'.index(t[1][0]) * 7 * 3 + (int(t[2][1]) - 2) * 3 + int(t[1][1]) - int(t[2][1]) + 1
        if t[0] == 'PENG': return self.OFFSET_ACT['PENG'] + self.OFFSET_TILE[t[1]]
        if t[0] == 'GANG': return self.OFFSET_ACT['GANG'] + self.OFFSET_TILE[t[1]]
        if t[0] == 'ANGANG': return self.OFFSET_ACT['ANGANG'] + self.OFFSET_TILE[t[1]]
        if t[0] == 'BUGANG': return self.OFFSET_ACT['BUGANG'] + self.OFFSET_TILE[t[1]]
        return self.OFFSET_ACT['PASS']
    
    def _obs(self):
        mask = np.zeros(self.ACT_SIZE)
        for a in self.valid:
            mask[a] = 1
        return {
            'observation': self.obs.reshape((self.OBS_SIZE, 4, 9)).copy(),
            'action_mask': mask
        }
    
    def _hand_embedding_update(self):
        self.obs[self.OFFSET_OBS['HAND'] : ] = 0
        d = defaultdict(int)
        for tile in self.hand:
            d[tile] += 1
        for tile in d:
            self.obs[self.OFFSET_OBS['HAND'] : self.OFFSET_OBS['HAND'] + d[tile], self.OFFSET_TILE[tile]] = 1
    
    def _check_mahjong(self, winTile, isSelfDrawn = False, isAboutKong = False):
        try:
            fans = MahjongFanCalculator(
                pack = tuple(self.packs[0]),
                hand = tuple(self.hand),
                winTile = winTile,
                flowerCount = 0,
                isSelfDrawn = isSelfDrawn,
                is4thTile = (self.shownTiles[winTile] + isSelfDrawn) == 4,
                isAboutKong = isAboutKong,
                isWallLast = self.wallLast,
                seatWind = self.seatWind,
                prevalentWind = self.prevalentWind,
                verbose = True
            )
            fanCnt = 0
            for fanPoint, cnt, fanName, fanNameEn in fans:
                fanCnt += fanPoint * cnt
            if fanCnt < 8: raise Exception('Not Enough Fans')
        except:
            return False
        return True
    
    def calculate_fanshu(self, winTile= None, isSelfDrawn = False, isAboutKong = False):
        try:
            fans = MahjongFanCalculator(
                pack = tuple(self.packs[0]),
                hand = tuple(self.hand),
                winTile = winTile,
                flowerCount = 0,
                isSelfDrawn = isSelfDrawn,
                is4thTile = (self.shownTiles[winTile] + isSelfDrawn) == 4,
                isAboutKong = isAboutKong,
                isWallLast = self.wallLast,
                seatWind = self.seatWind,
                prevalentWind = self.prevalentWind,
                verbose = True
            )
            fanCnt = 0
            for fanPoint, cnt, fanName, fanNameEn in fans:
                fanCnt += fanPoint * cnt
            return fanCnt
        except:
            return None
        

