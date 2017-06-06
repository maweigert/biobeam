from .version import __version__

from biobeam.core import *
from biobeam.simlsm import SimLSM_Cylindrical, SimLSM_DSLM

#
import logging
logging.basicConfig(format='%(levelname)s:%(name)s | %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)



#logger.setLevel(logging.DEBUG)


# try:
#     import spimagine
#     from biobeam.beam_gui.volbeam import volbeam
# except ImportError:
#     pass


