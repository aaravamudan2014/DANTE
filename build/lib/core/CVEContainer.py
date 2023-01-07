##################################################
# CVEContainer.py
##################################################

from core.CVE import *
import datetime as datetime
import pandas as pd
import pickle
import traceback
##################################################
# CVEContainer: class definition & implementation
#
# A class designed to contain all the CVE objects and maintain
# meta information on the dataset that would be useful in modelling
# PUBLIC ATTRIBUTES:
#
#   TimeStamp:    scalar float; the time of the event in arbitrary time units.
#   EventType:    EventType (assumed unsigned); the type of event associated with the software
#                   vulnerability
# PUBLIC INTERFACE:
#
# USAGE EXAMPLES:
#
# DEPENDENCIES: Enum
#
# AUTHOR: Akshay Aravamudan, December 2019
#
##################################################


class CVEContainer(object):
    def __init__(self, CVE_list):
        self.CVE_List = CVE_list
        self.numCVEs = len(CVE_list)
        final_infection_time = None
        cnt = 0
        for cve in CVE_list:
            assert isinstance(
                cve, CVE), "Object in the list has to of CVE type"
            if cnt == 0:
                cnt = 1
                final_infection_time = cve.getFinalEventTime()
            elif cve.getFinalEventTime() > final_infection_time:
                final_infection_time = cve.getFinalEventTime()
        self.FinalInfectionTime = final_infection_time

    def getRightCensoringTime(self):
        return self.FinalInfectionTime

    def getFeatureVectorLength(self):
        feature_vector_length = None
        cnt = 0
        for cve in CVE_List:
            if cnt == 0:
                cnt = 1
                feature_vector_length = len(cve.getFeatureVector())
            else:
                assert len(cve.getFeatureVector(
                )) == feature_vector_length, "Feature vector lengths do not match, they must be the same for all CVEs"

        return feature_vector_length

    def getCVEList(self):
        return self.CVE_List

    def save(self, saveFileName):
        df = pd.DataFrame(columns=['CVE', 'eventTimeStamps', 'eventTypes',
                                   'SocialMediaCascades', 'SocialMediaPlatforms'])
        for cve in self.CVE_List:
            cve_id = cve.getCVEid()
            cve_event_timestamps = []
            cve_event_types = []
            soc_media_timestamps = []
            soc_media_platforms = []
            for event in cve.getCVEEvents():
                cve_event_timestamps.append(event.getTimestamp())
                cve_event_types.append(event.getEventType())

            for social_media_cascade in cve.getSocialMediaCascades():
                soc_media_timestamps.append(
                    social_media_cascade.getTimestamps())
                soc_media_platforms.append(
                    social_media_cascade.getSocialMediaPlatform())
            df = df.append({'CVE': cve_id, 'eventTimeStamps': cve_event_timestamps, 'eventTypes': cve_event_types,
                            'SocialMediaCascades': soc_media_timestamps, 'SocialMediaPlatforms': soc_media_platforms}, ignore_index=True)
        df.to_csv(saveFileName)

    def load(self):
        pass
        # TODO: load entries as a csv list

    def getNumCVEs(self):
        return self.numCVEs

    def convertToRelativeTime(self):
        for cve in self.CVE_List:
            cve.convertToRelativeTime()


def main():
    pass


if __name__ == "__main__":
    main()
