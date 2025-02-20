from pathlib import Path
from PIL import Image
import numpy as np

def get_jigsaw_pieces(size):
    '''
    Load all pieces of the 4x4 jigsaw puzzle.

    size (int) :
        Should be 64 or 256, indicating side length of jigsaw puzzle
    '''

    # Location of pieces
    piece_dir = Path(__file__).parent / 'assets'

    # Helper function to load pieces as np arrays
    def load_pieces(path):
        '''
        Load a piece, from the given path, as a binary numpy array.
        Return a list of the "base" piece, and all four of its rotations.
        '''
        piece = Image.open(path)
        piece = np.array(piece)[:,:,0] // 255
        pieces = np.stack([np.rot90(piece, k=-i) for i in range(4)])
        return pieces

    # Load pieces and rotate to get 16 pieces, and cat
    pieces_corner = load_pieces(piece_dir / f'4x4/4x4_corner_{size}.png')
    pieces_inner = load_pieces(piece_dir / f'4x4/4x4_inner_{size}.png')
    pieces_edge1 = load_pieces(piece_dir / f'4x4/4x4_edge1_{size}.png')
    pieces_edge2 = load_pieces(piece_dir / f'4x4/4x4_edge2_{size}.png')
    pieces = np.concatenate([pieces_corner, pieces_inner, pieces_edge1, pieces_edge2])

    return pieces

# Helper function to load pieces as np arrays
def load_pieces(original_path, transform_path, rotation=0):
    '''
    Load a piece, from the given path, as a binary numpy array.
    Return a list of the "base" piece, and the specified rotation
    Note this only returns the piece
    '''
    # The base state
    piece = Image.open(original_path)
    piece = np.array(piece)[:,:,0] // 255
    # The final state
    transform_piece = Image.open(transform_path)
    transform_piece = np.array(transform_piece)[:,:,0] // 255
    # The directory of the piece
    #transform_index = rotation // 90
    #pieces = np.zeros((4,8,8))
    #pieces[0] = piece
    #pieces[transform_index] = transform_piece
    pieces = np.stack([piece, transform_piece])
    return pieces

# Helper function to load the transform matrix
def load_transform_matrix(path, only_rotations=True):
    '''
    The transform matrix contains the destination and rotation for every single piece, for example:
    [x=1,y=7,n=1,r=180], [x=2,y=1,n=9,r=0], ... 
    Where this is a nxn puzzle

    Each cell represents the starting position of a piece, the x and y are the destination,
    the n represents the unique piece ID (which is currently not used), and r is the rotation in degrees 
    '''
    matrix = np.loadtxt(path, delimiter=',', dtype=str)
    if not only_rotations:
        return matrix
    
    # Only return the rotations in a 2D array
    return matrix[:,3]

def get_jigsaw_pieces_exhaustive(size, dims=8):
    '''
    Load all pieces of the a puzzle defined by dims x dims and in folder

    size (int) :
        Should be 64 or 256, indicating side length of jigsaw puzzle
    dims (int) :
        The number of pieces in the puzzle on each side
    '''

    dims_folder = f'{dims}x{dims}'

    # Location of pieces
    piece_dir = Path(__file__).parent / 'assets'
    transform_dir = piece_dir / dims_folder / f"{dims_folder}-transform.txt"
    piece_dir = piece_dir / dims_folder / str(size)
    original_dir = piece_dir / 'original'
    permutation_dir = piece_dir / 'transform'

    # Get the total number of files in the folder
    num_files = len(list(original_dir.glob('*.png')))

    # Get the rotations
    rotations = load_transform_matrix(transform_dir, only_rotations=True)
    
    pieces = np.array([])
    for i in range(num_files):
        file_name = f"{dims_folder}-{i}{'-256' if size == 256 else ''}.png"
        original_file = original_dir / file_name
        transform_file = permutation_dir / file_name
        pieces = np.append(pieces, load_pieces(original_file, transform_file, rotations[i]))

    return pieces