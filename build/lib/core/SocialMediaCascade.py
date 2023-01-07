############################################################################
# SocialMediaCascade.py
############################################################################


############################################################################
# The following enumeration is intended to set the mark associated
# with each list of timestamps to indicate which social media platform
# it arises from.
############################################################################

from enum import Enum


class SocialMediaPlatform(Enum):
    TWITTER = 1
    REDDIT = 2
    GITHUB = 3
    NEWS = 4

############################################################################
# SocialMediaCascade: class definition & implementation
#
# A class designed to encapsulate social media events grouped
# by platforms. This version currently considers Twitter, Reddit
# and Github for social-media cascade.
#
# PUBLIC ATTRIBUTES:
#
#   TimeStamps:    list float; The list of timestamps in ascending order.
#   Platform:      SocialMediaPlatform; The platform to which the timestamps belong.
#
# PUBLIC INTERFACE:
#
#   SocialMediaCascade():   Event object; constructor
#   __str__(): string; string representation of event object
#   print():   print string representation of event object
#
# USAGE EXAMPLES:
#
#   soc_media_cascade = SocialMediaCascade(timeStamps, platform)   # object construction, when parent node ID is unknown
#
#   print(soc_media_cascade)   # prints out event information
#   soc_media_cascade.print()   # equivalent method to print out event information
#
# DEPENDENCIES: enum
#
# AUTHOR: Akshay Aravamudan, Decembers 2019
#
####################################################################################


class SocialMediaCascade(object):

    def __init__(self, timeStamps, platform):
        assert isinstance(timeStamps, list), "Timestamps must be in a list"
        assert len(
            timeStamps) > 1, "Timestamps should contain atleast one timestamp"
        timeStamps.sort()
        self.TimeStamps = timeStamps
        assert isinstance(
            platform, SocialMediaPlatform), "The platform should be an enumeration of SocialMediaPlatform"
        self.Platform = platform

    def __str__(self):
        return "\n Social media platform: %f  Number of timestamps: %d".format(self.Platform, len(self.TimeStamps))

    def convertToRelativeTime(self, startTime):
        newTimeStamps = []
        for timeStamp in self.TimeStamps:
            timeStamp -= startTime
            timeStamp = timeStamp.total_seconds()
            newTimeStamps.append(timeStamp)
        self.TimeStamps = newTimeStamps

    def print(self):
        print(self)

    def getTimestamps(self):
        return self.TimeStamps

    def addToTimestamps(self, timeStamp):
        self.TimeStamps.append(timeStamp)

    def getSocialMediaPlatform(self):
        return self.Platform


def main():
    # include all available tests for this class.
    pass


if __name__ == "__main__":
    main()
