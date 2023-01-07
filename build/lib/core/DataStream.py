'''
DataStream.py
    Data class to store and deliver data values to the appropriate univariate or \
    multivariate process
    Author: Akshay Aravamudan
'''


####################################################################################################
#
# DataStream: class definition and implementation
#
# A class designed to contain information associated with an entry in
# the CVE (Common Vulnerabilities and Exploits) database.
# PUBLIC ATTRIBUTES:
#
#   streams:               List of list of timestamps that maps in the same order as the sourceNames variable
#   sourceNames:           List of source names: CVE-ID
#   cascadeNames:          List of cascade names, can represent CVE, thread episode title, hashtags, etc..
# PUBLIC INTERFACE:
#
#   DataStream():   DataStream object; constructor
#   __str__(): string; data statistics of DataStream object
#   print():   print data statistics of DataStream object
#
# USAGE EXAMPLES:
#
#   dataStream = DataStream(listOfTimestampArray, sourceNames)   # object construction, when parent node ID is unknown
#
# DEPENDENCIES:
#
# AUTHOR: Akshay Aravamudan, January 2020
#
####################################################################################################


class DataStream(object):
    def __init__(self):

        # self.streams = []
        # self.sourceNames = []
        # self.cascadeNames = []

        # convert all the timestamps to relative time (relative to its own mitre publish time)

        # append right censoring time to ecah stream for each cve

        self.processDataSequence = {}

        pass

    def initializeFromSimulatedSamples(self, MTPPRealizations, sourceNames):
        dataIntegrityFlag, error_messages = self.dataIntegrityCheck(
            MTPPRealizations)
        assert dataIntegrityFlag, error_messages

        self._streams = MTPPRealizations
        self.numRealizations = len(self._streams[0])
        self._sourceNames = sourceNames

    def dataIntegrityCheck(self, MTPPRealizations):
        error_messages = ''
        # check that MTPPRealizations is of type list
        typeFlag = isinstance(MTPPRealizations, list)
        if not typeFlag:
            error_messages += 'The data is not of list format, make sure it is of the form list of lists.'
        # check there is atleast one stream defined
        if typeFlag:
            sizeFlag = True if len(MTPPRealizations) > 0 else False
        else:
            sizeFlag = False

        if not sizeFlag:
            error_messages += 'The passed data must contain atleast one stream. '
        consistencyFlag = True
        # check that the number of realizations is consistent across all streams
        # if sizeFlag:
        #     expectedRealizations = len(MTPPRealizations[0])
        #     consistencyFlag = True
        #     for stream in MTPPRealizations:
        #         if not len(stream) == expectedRealizations:
        #             consistencyFlag = False
        #             break
        # else:
        #     consistencyFlag = False

        if not consistencyFlag:
            error_messages += 'The number of realizations are not consistent across data streams'

        return typeFlag and sizeFlag and consistencyFlag, error_messages

    def getDataStreamLearning(self, sourceName):
        assert isinstance(
            sourceName, str), "sourceName parameters must of string type"
        modifiedSourceNames = self._sourceNames.copy()
        modifiedTPPData = self._streams.copy()

        # ensure that the sourceName is on top
        index = modifiedSourceNames.index(sourceName)

        if not index == 0:
            modifiedTPPData[0], modifiedTPPData[index] = modifiedTPPData[index], modifiedTPPData[0]
            modifiedSourceNames[0], modifiedSourceNames[index] = modifiedSourceNames[index], modifiedSourceNames[0]
        # also retain the list of sourceNames for saving parameters back into the multivariate model
        self.processDataSequence[sourceName] = modifiedSourceNames

        return modifiedTPPData, modifiedSourceNames

    def convertToRelativeTime(self, mitre_timestamp):

        pass

    def getFeatureVectorLength(self):
        pass

    def getNumCVEs(self):
        pass

    def save(self, filename):
        statusFlag = False
        pass
        return statusFlag

    def load(self, filename):
        pass
        return self


def main():
    # first read master dataset

    # convert the timestamps into MTPP realizations

    pass


if __name__ == "__main__":
    main()
