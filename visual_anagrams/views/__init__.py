from pathlib import Path
from PIL import Image
import numpy as np

from .view_jigsaw_8 import JigsawView_8
from .view_jigsaw_32 import JigsawView_32

from .view_identity import IdentityView
from .view_flip import FlipView
from .view_rotate import Rotate180View, Rotate90CCWView, Rotate90CWView
from .view_negate import NegateView
from .view_skew import SkewView
from .view_patch_permute import PatchPermuteView
from .view_inner_circle import InnerCircleView
from .view_square_hinge import SquareHingeView

VIEW_MAP = {
    'identity': IdentityView,
    'flip': FlipView,
    'rotate_cw': Rotate90CWView,
    'rotate_ccw': Rotate90CCWView,
    'rotate_180': Rotate180View,
    'negate': NegateView,
    'skew': SkewView,
    'patch_permute': PatchPermuteView,
    'pixel_permute': PatchPermuteView,
    'jigsaw': JigsawView_8,
    'jigsaw_32': JigsawView_32,
    'inner_circle': InnerCircleView,
    'square_hinge': SquareHingeView,
}

def get_views(view_names):
    '''
    Bespoke function to get views (just to make command line usage easier)
    '''
    views = []
    for view_name in view_names:
        if view_name == 'patch_permute':
            args = [8]
        elif view_name == 'pixel_permute':
            args = [64]
        elif view_name == 'skew':
            args = [1.5]
        else:
            args = []

        view = VIEW_MAP[view_name](*args)
        views.append(view)

    return views
