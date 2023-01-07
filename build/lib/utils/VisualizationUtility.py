'''
VisualizationUtility.py
    Used to contain all helper functions for visualization in this library.
    prettyFormatTimeElapsed() : for printing time elapsed while iterations in training.
    intensity_plot():  for plotting intensity function of temporal point process objects.
    
    Author: Akshay Aravamudan, January 2020
'''

# prettyFormatTimeElapsed()
#
# Expresses time elapsed (as measured in seconds) as a string quoting hours, minutes and seconds.
#
# SYNTAX
#	prettyFormatTimeElapsed(elapsedTimeSeconds)
#
# INPUTS
#	elapsedTimeSeconds: a float representing elapsed time and measured in seconds.
#
# OUTPUTS
#	prettyTimeString: a string of the form "xx hours xx minutes xx.xx seconds"
#
# AUTHOR
#	Georgios C. Anagnostopoulos, August 2019
#
import time

# Some global settings for figure sizes
normalFigSize = (8, 6)  # (width,height) in inches
largeFigSize = (12, 9)
xlargeFigSize = (18, 12)


def prettyFormatTimeElapsed(elapsedTimeSeconds):
    """Expresses time elapsed (as measured in seconds) as a string quoting hours, minutes and seconds."""
    timeMinutes, timeSeconds = divmod(elapsedTimeSeconds, 60)
    timeHours, timeMinutes = divmod(timeMinutes, 60)
    prettyTimeString = ''
    if timeHours > 0:
        prettyTimeString = prettyTimeString + \
            '{:d} hours'.format(int(timeHours))
    if timeMinutes > 0:
        prettyTimeString = prettyTimeString + \
            ' {:d} minutes'.format(int(timeMinutes))
    if timeSeconds > 0:
        prettyTimeString = prettyTimeString + \
            ' {:04.02f} seconds'.format(timeSeconds)
    return prettyTimeString


################################################################################
#
# U N I T   T E S T I N G
#
###############################################################################
def unitTest_prettyFormatTimeElapsed():
    secs = 5  # seconds to sleep
    startTime = time.time()
    # do something, e.g.:
    time.sleep(secs)
    elapsedTimeSeconds = time.time() - startTime
    print('Finished after' + prettyFormatTimeElapsed(elapsedTimeSeconds))

#########################################################################################################################
# intensity_plot(): This functions will display a plot with two subplots. The first subplot shows the intesity function
#                   as the points arrive and the second subplot is a stem plot showing realizations on the timeline. Both
#                   have the same scale.
#  return:          None
# Arguments:
# tpp: The point process which generates the points.
# events: List of events associated with the point process.
# num_points: Number of points to be plotted, it is defaulted to 5 to reduce congestion.
#########################################################################################################################


def intensity_plot(tpp, events, num_points=5):
    num_points = min(len(events), num_points)
    events = events[0:num_points]
    plt.xticks(range(1, 150))
    fig, ax = plt.subplots(nrows=2, ncols=1)

    starting_point = 0
    earliest_event = min(events)
    i = 0

    def intensity_function(start, end):
        new_end = end
        new_end -= start
        new_start = 0
        t = np.linspace(new_start, new_end, 1000)
        points = tpp.intensity(t, events)
        return t + start, points

    max_pts = 0
    for i in range(len(events)):
        if i == 0:
            t, pts = intensity_function(0.0, events[i])
        else:
            t, pts = intensity_function(events[i-1], events[i])
        ax[0].plot(t, pts, 'b')
        if max(pts) > max_pts:
            max_pts = max(pts)

    ax[0].vlines(events, 0, max_pts * np.ones(len(events)), linestyle="dashed")
    ax[0].set_xlabel('Time (s)')
    ax[1].set_xlabel('Time (s)')
    ax[0].set_ylabel('Intensity')
    ax[1].set_ylabel('Marks')

    events = list(events)
    events.append(0.0)
    stem_points = np.ones(len(events))

    ax[1].stem(events, stem_points, use_line_collection=True)


################################################################################################
# generatePP_plot(): function to generate pp plot(), call plt.show() after function to see plot
# utility function to generate pp plot from using the point process'
def generatePP_plot(Realization, modeltpp):
    import matplotlib.pyplot as plt
    _, ax = plt.subplots(1, 1, figsize=normalFigSize)

    # generate the P-P plot
    pvalue = modeltpp.GOFks([Realization], ax, showConfidenceBands=True)
    print('KS test p-value={}'.format(pvalue))


def main():
    unitTest_prettyFormatTimeElapsed()


if __name__ == '__main__':
    main()
