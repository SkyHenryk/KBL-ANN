# -*- coding: utf-8 -*-

import csv
import numpy as np
import pandas as pd

class kboDataPreprocessing:

    matchYears = [2016]
    dataFolder = "../data"

    def start(self):

        allStatus, allMatchResult = self.loadAllMatchResult(self.matchYears)
        preprocessedData = self.preprocessingData(allStatus, allMatchResult)
        self.saveInCsv(preprocessedData)


    def loadAllMatchResult(self, matchYears):

        allStatus = {}
        allMatchResult = None

        for matchYearIndex, matchYear in enumerate(matchYears):

            for statusYearIndex in [0, 1]:

                statusYear = self.loadStatusYear(matchYear,statusYearIndex)
                allStatus[statusYear] = {}

                for statusType in ["player", "team"]:
                    allStatus[statusYear][statusType] = {}

                    for statusPosition in ["batter", "pitcher"]:
                        allStatus[statusYear][statusType][statusPosition] = {}

                        with open(f"{self.dataFolder}/kbo_{statusType}_{statusPosition}_{statusYear}.csv",
                                  newline='') as kboStatus:
                            kboStatusReader = csv.reader(kboStatus, delimiter=',', quotechar='|')

                            for status in kboStatusReader:
                                allStatus[statusYear][statusType][statusPosition][status[0]] = status[-7:]

                with open(f"{self.dataFolder}/kbo_result_{matchYear}.csv", newline='') as kboResult:
                    kboResultReader = csv.reader(kboResult, delimiter=',', quotechar='|')
                    kboResult = np.array(list(kboResultReader))

                kboResult = np.insert(kboResult, 0, matchYear, axis=1)
                kboResult = np.insert(kboResult, 1, statusYear, axis=1)
                kboResult = np.insert(kboResult, 2, matchYear - int(statusYear), axis=1)

                if allMatchResult is None:
                    allMatchResult = kboResult
                else:
                    allMatchResult = np.concatenate((allMatchResult, kboResult), axis=0)

        return allStatus, allMatchResult


    def loadStatusYear(self, matchYear, statusYearIndex):
        return str(matchYear - statusYearIndex)


    def preprocessingData(self, allStatus, allMatchResult):

        matchYear = allMatchResult[:, 0]
        statusYear = allMatchResult[:, 2]
        matchMonth = [x.split(".")[0] for x in allMatchResult[:, 3]]
        homeTeamName = allMatchResult[:, 5]
        scoreData = allMatchResult[:, 6]
        awayTeamName = allMatchResult[:, 7]

        matchYearOneHot = self.oneHot(matchYear)
        statusYearOneHot = self.oneHot(statusYear)
        matchMonthOneHot = self.oneHot(matchMonth)
        homeTeamOneHot = self.oneHot(homeTeamName)
        awayTeamOneHot = self.oneHot(awayTeamName)
        scoreDif = np.array([int(x.split(":")[0]) - int(x.split(":")[1]) for x in scoreData])
        scoreWinningRate = np.vectorize(self.calculateWinningRate)(scoreDif)
        scoreWinningRateOneShot = self.oneHot(scoreWinningRate)
        allPlayerStatus = self.loadStatusByPlayerName(allStatus, allMatchResult)

        preprocessedData = np.concatenate(
            (statusYearOneHot, matchYearOneHot, matchMonthOneHot, homeTeamOneHot, awayTeamOneHot, allPlayerStatus,
             scoreWinningRateOneShot), axis=1)
        np.random.shuffle(preprocessedData)

        return preprocessedData

    def calculateWinningRate(self, x):
        if x > 1:
            return 1
        else:
            return 0

    def loadStatusByPlayerName(self, allStatus, allMatchResult):

        allMatchStatus = []

        for MatchResult in allMatchResult:
            matchStatus = []
            yearStatus = allStatus.get(MatchResult[1])
            yearPlayerStatus = yearStatus.get("player")
            yearTeamStatus = yearStatus.get("team")
            if yearPlayerStatus is None:
                continue
            yearPlayerPitcherStatus = yearPlayerStatus.get("pitcher")
            yearPlayerBatterStatus = yearPlayerStatus.get("batter")
            if yearTeamStatus is None:
                continue
            yearTeamPitcherStatus = yearTeamStatus.get("pitcher")
            yearTeamBatterStatus = yearTeamStatus.get("batter")

            homePicherStatus = yearPlayerPitcherStatus.get(MatchResult[8].strip())
            if homePicherStatus is None:
                homePicherStatus = yearTeamPitcherStatus.get(MatchResult[5].strip())
            matchStatus += homePicherStatus
            awayPicherStatus = yearPlayerPitcherStatus.get(MatchResult[28].strip())
            if awayPicherStatus is None:
                awayPicherStatus = yearTeamPitcherStatus.get(MatchResult[7].strip())
            matchStatus += awayPicherStatus

            for homeBatterIndex in range(9, 28):
                homeBatterStatus = yearPlayerBatterStatus.get(MatchResult[homeBatterIndex].strip())
                if homeBatterStatus is None:
                    homeBatterStatus = yearTeamBatterStatus.get(MatchResult[5].strip())
                matchStatus += homeBatterStatus
            for awayBatterIndex in range(29, 38):
                awayBatterStatus = yearPlayerBatterStatus.get(MatchResult[awayBatterIndex].strip())
                if awayBatterStatus is None:
                    awayBatterStatus = yearTeamBatterStatus.get(MatchResult[7].strip())
                matchStatus += awayBatterStatus

            matchStatus = [self.setFloat(x) for x in matchStatus]
            allMatchStatus.append(matchStatus)

        if len(allMatchStatus) == 0:
            allMatchStatus = [[0]]
        return np.array(allMatchStatus)

    def oneHot(self,x):
        result = pd.get_dummies(pd.Series(x)).values
        return result

    def setFloat(self, x):

        if " -" in x:
            if len(x) > 2:
                result = float(x[2:]) * -1
            else:
                result = float(0)
        else:
            result = float(x)

        return result

    def saveInCsv(self, data):
        with open(f'{self.dataFolder}/kbo_data.csv',"w") as f:
            writer = csv.writer(f)
            writer.writerows(list(data))

if __name__ == "__main__":
    kboDataPreprocessing().start()