import cmsisdsp.version

from cmsisdsp_filtering import *
from cmsisdsp_matrix import *
from cmsisdsp_support import *
from cmsisdsp_statistics import *
from cmsisdsp_complexf import *
from cmsisdsp_basic import *
from cmsisdsp_controller import *
from cmsisdsp_transform import *
from cmsisdsp_interpolation import *
from cmsisdsp_quaternion import *
from cmsisdsp_fastmath import *
from cmsisdsp_distance import *
from cmsisdsp_bayes import *
from cmsisdsp_svm import *


__version__ = cmsisdsp.version.__version__

# CMSIS-DSP Version used to build the wrapper
cmsis_dsp_version="1.10.0"


# Commit hash used to build the wrapper
commit_hash="7106010d41b3bf810403e954c36c2643d98c58e7"

# True if development version used
developmentVersion=True

__all__ = ["datatype", "fixedpoint", "mfcc"]