##################################################
# CVE.py
#
# CVE: class definition and implementation
#
# A class designed to contain information associated with an entry in
# the CVE (Common Vulnerabilities and Exploits) database.
# PUBLIC ATTRIBUTES:
#
#   FeatureVector:    vector of floats; Feature vector associated with the CVE
#   CVE_ID:             string: CVE-ID
#   EventList:          List of Events;
#   SocialMediaCascades: List of SocialMediaCascade objects, one for each social media platform
# PUBLIC INTERFACE:
#
#   CVE():   Event object; constructor
#   __str__(): string; string representation of event object
#   print():   print string representation of event object
#
# USAGE EXAMPLES:
#
#   cve = CVE(featureVector, eventList, socialMediaCascades)   # object construction, when parent node ID is unknown
#
#   print(cve)   # prints out event information
#   cve.print()   # equivalent method to print out event information
#
# DEPENDENCIES: Event, datetime
#
# AUTHOR: Akshay Aravamudan, December 2019
#
##################################################

from core.Event import EventType
from core.Event import *
from datetime import datetime


class CVE(object):
    def __init__(self, eventList, socialMediaCascades, cve_id):
        # assert len(featureVector) >= 1, "Feature vector length should be greater than 1"
        # self.FeatureVector = featureVector
        assert isinstance(cve_id, str), "CVE id input has to be a string"
        self.CVE_ID = cve_id
        assert len(
            eventList) >= 1, "Event list should contain more than one event"
        eventList.sort()

        # replace eventList with a list of streams

        self.EventList = eventList
        self.SocialMediaCascades = socialMediaCascades

    def __str__(self):
        return '<empty cascade>' if self.isEmpty() else ",".join(map(str, self.EventList))

    def print(self):
        print(self)

    def getCVEid(self):
        return self.CVE_ID

    def convertToRelativeTime(self):
        # find mitre time and make every other time relative.
        mitre_time = None
        for event in self.EventList:
            if event.getEventType() == EventType.MITRE_PUBLISH:
                mitre_time = event.getTimestamp()
                if not isinstance(mitre_time, datetime):
                    print("Times are already relative times")
                    break

        assert mitre_time is not None, "mitre time not found, cannot calculate relative times without it..."

        for event in self.EventList:
            event.setTimestamp(
                (event.getTimestamp() - mitre_time).total_seconds())

        for socialMediaCascade in self.SocialMediaCascades:
            socialMediaCascade.convertToRelativeTime(mitre_time)

    def isAbsoluteTime(self):
        # get mitre time
        # check if mitre time is an instance of datetime or int
        for event in self.EventList:
            if event.getEventType() == EventType.MITRE_PUBLISH:
                if isinstance(event.getTimestamp(), datetime):
                    return True
        return False

    def getFinalEventTime(self):
        # iterate through all events to look for
        cnt = 0
        finalInfectionTime = None
        for event in self.EventList:
            if cnt == 0:
                cnt = 1
                finalInfectionTime = event.getTimestamp()
            elif event.getTimestamp() >= finalInfectionTime:
                finalInfectionTime = event.getTimestamp()

        for soc_media_cascade in self.SocialMediaCascades:
            for event_timestamp in soc_media_cascade.getTimestamps():
                if event_timestamp > finalInfectionTime:
                    finalInfectionTime = event.getTimestamp()

        return finalInfectionTime

    def getFeatureVector(self):
        return self.FeatureVector

    def setFeatureVector(self, featureVector):
        self.FeatureVector = featureVector

    def setGroundTruthLabel(self, y_label):
        self.GroundTruthLabel = y_label

    def getGroundTruthLabel(self):
        assert GroundTruthLabel != None, "Ground truth label is empty, please check data pipeline"
        return self.GroundTruthLabel

    def isExploited(self):
        for event in self.EventList:
            if event.getEventType() == EventType.EXPLOIT:
                return True

        return False

    def getCVEEvents(self):
        return self.EventList

    def getMitreTime(self):
        mitre_time = None
        for event in self.EventList:
            if event.getEventType() == EventType.MITRE_PUBLISH:
                mitre_time = event.getTimestamp()

        return mitre_time

    def getSocialMediaCascades(self):
        return self.SocialMediaCascades

    def addPoint(self, timestamp, id):
        #  id : 0,1,2,3 - sm1, sm2, sm3, exploit
        if id < 3:
            self.getSocialMediaCascades()[id].addToTimestamps(timestamp)
        else:
            # add exploit
            event = Event(timestamp, EventType.EXPLOIT)


def main():
    pass


if __name__ == "__main__":
    main()
