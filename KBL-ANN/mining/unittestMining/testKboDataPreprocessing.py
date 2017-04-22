from unittest import TestCase
from ..kboDataPreprocessing import kboDataPreprocessing
import numpy as np

class TestKboDataPreprocessing(TestCase):

    @classmethod
    def setUpClass(cls):
        return

    def setUp(self):
        self.kboDataPreprocessing = kboDataPreprocessing()
        self.kboDataPreprocessing.dataFolder = "../../data"
        self.allStatus = {"2016":
                              {'player':
                                  {'batter':
                                       {'테임즈':
                                            [' 0.390', ' 0.381', ' 0.497', ' 0.790', ' 1.287', ' 0.520', ' 11.73']
                                        }}}}
        self.allMatchResult = np.array([
                                ['2016','2016','0','3.8','13:00','SK','6:6','롯데',' 레일리',' 테임즈',' 안중열'
                             ' 이우민',' 박종윤',' 강동수',' 김준태',' 오승택',' 정훈',' 오현근',' 박헌도',' 손아섭',' 황재균'
                             ' 이여상',' 김주현',' 손용석',' 김문호',' None',' None',' None',' 문광은',' 유서준',' 최정'
                             ' 박정권',' 이대수',' 김민식',' 조성모',' 고메즈',' 정의윤']]
                                        )

    def test_loadAllMatchResult(self):

        allStatus, allMatchResult = self.kboDataPreprocessing.loadAllMatchResult([2016])
        self.assertGreater(len(allMatchResult), 1)
        self.assertIsNotNone(allStatus.get("2016"))


    def test_loadStatusYear(self):

        statusYear = self.kboDataPreprocessing.loadStatusYear(2016,1)
        self.assertEqual(statusYear,'2015')


    def test_preprocessingData(self):

        preprocessedData = self.kboDataPreprocessing.preprocessingData(self.allStatus, self.allMatchResult)
        self.assertGreater(len(preprocessedData), 0)


    def test_calculateWinningRate(self):

        winningRate = self.kboDataPreprocessing.calculateWinningRate(1)
        self.assertEqual(winningRate, 0)


    def test_loadStatusByPlayerName(self):

        statusByPlayerName = self.kboDataPreprocessing.loadStatusByPlayerName(self.allStatus, self.allMatchResult)
        self.assertEqual(statusByPlayerName[0][0], 0)


    def test_oneHot(self):

        oneHot = self.kboDataPreprocessing.oneHot([1,0])
        self.assertEqual(oneHot[1][0], 1)


    def test_setFloat(self):

        floatValue = self.kboDataPreprocessing.setFloat(" -2")
        self.assertEqual(floatValue, -2)


    def test_saveInCsv(self):

        self.kboDataPreprocessing.dataFolder = "."
        self.kboDataPreprocessing.saveInCsv(np.array([[1,0],[0,1]]))
        xy = np.loadtxt('./kbo_data.csv', delimiter=',', dtype=np.float32)
        self.assertEqual(xy[0][0],1)
