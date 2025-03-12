import logging
import os
import sys

from .aruco import ArucoMarker
from .sub_aruco import SubAruco
from .marvelmind import MarvelmindHedge
from .sub_marvelmind import SubMM
from .controller import Controller

############### logging 
DEBUG = 0 # setup the debugging mode which store all loggings to a log file

logger = logging.getLogger('sc')

if DEBUG == 1:
    # if in debugging mode, save all message in log file
    logger.setLevel(logging.DEBUG)
    handler = logging.FileHandler("log.log", mode='w')

else:
    # if not, only print error message in console
    logger.setLevel(logging.ERROR)
    handler = logging.StreamHandler()

formatter = logging.Formatter("%(asctime)-15s %(levelname)s %(filename)s:%(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)



__all__ = ['logger', 'ArucoMarker', 'SubAruco', 'SubPos', 'SubIMU', 'MarvelmindHedge']

