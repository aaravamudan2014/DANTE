import logging


# Logging levels available: CRITICAL, ERROR, WARNING,  INFO, DEBUG, NOTSET

def createLogger(loggerFilename):
    # Create a custom logger
    logger = logging.getLogger(__name__)

    # Create handlers
    c_handler = logging.StreamHandler()
    f_handler = logging.FileHandler(loggerFilename + '.log')
    c_handler.setLevel(logging.WARNING)
    f_handler.setLevel(logging.ERROR)

    # Create formatters and add it to handlers
    c_format = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
    f_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    c_handler.setFormatter(c_format)
    f_handler.setFormatter(f_format)

    # Add handlers to the logger
    logger.addHandler(c_handler)
    logger.addHandler(f_handler)

    logger.info('logger has been initiated')

    return logger


# Function to return the loggers associated with a multivariate process.
# It is assumed that objects of these classes have the following functions:
# (i) getSourceNames() - to return the names of all individual processes within the multivariate process
# (ii) getSourceName() - to return the name of the multivariate process
def getLoggersMultivariateProcess(MultiVariateProcessObject):

    processNames = MultiVariateProcessObject.getSourceNames()
    multiVariateProcessName = MultiVariateProcessObject.getSourceName()

    logger_directory_multivariate = '../logs/MultivariateProcess/'
    logger_directory_univariate = '../logs/UnivariateProcess/'

    logger_list = []
    # create a logger for the multivariate process to print out relevant information for the multivariate process
    logger_list.append(createLogger(
        logger_directory_multivariate + multiVariateProcessName))
    #  create a logger for all the univariate processes along with the appropriate configurations
    for processName in processNames:
        logger_list.append(createLogger(
            logger_directory_univariate + processName))

    return logger_list
