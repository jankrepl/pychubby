"""Collection of functions focused on obtaining data."""

import bz2
import pathlib
import urllib

from pychubby.base import CACHE_FOLDER


def get_pretrained_68(folder=None, verbose=True):
    """Get pretrained landmarks model for dlib.

    Parameters
    ----------
    folder : str or pathlib.Path or None
        Folder where to save the .dat file.

    verbose : bool
        Print some output.

    References
    ----------
    [1] C. Sagonas, E. Antonakos, G, Tzimiropoulos, S. Zafeiriou, M. Pantic.
        300 faces In-the-wild challenge: Database and results. Image and Vision Computing (IMAVIS),
        Special Issue on Facial Landmark Localisation "In-The-Wild". 2016.

    [2] C. Sagonas, G. Tzimiropoulos, S. Zafeiriou, M. Pantic.
        A semi-automatic methodology for facial landmark annotation. Proceedings of IEEE Int’l Conf.
        Computer Vision and Pattern Recognition (CVPR-W), 5th Workshop on Analysis and Modeling of
        Faces and Gestures (AMFG 2013). Oregon, USA, June 2013.

    [3] C. Sagonas, G. Tzimiropoulos, S. Zafeiriou, M. Pantic.
        300 Faces in-the-Wild Challenge: The first facial landmark localization Challenge.
        Proceedings of IEEE Int’l Conf. on Computer Vision (ICCV-W), 300 Faces in-the-Wild

    """
    url = "https://raw.githubusercontent.com/"
    url += "davisking/dlib-models/master/shape_predictor_68_face_landmarks.dat.bz2"

    folder = pathlib.Path(CACHE_FOLDER) if folder is None else pathlib.Path(folder)
    filepath = folder / "shape_predictor_68_face_landmarks.dat"

    if filepath.is_file():
        return

    if verbose:
        print("Downloading and decompressing {} to {}.".format(url, filepath))

    req = urllib.request.urlopen(url)
    CHUNK = 16 * 1024

    decompressor = bz2.BZ2Decompressor()
    with open(str(filepath), 'wb') as fp:
        while True:
            chunk = req.read(CHUNK)
            if not chunk:
                break
            fp.write(decompressor.decompress(chunk))
    req.close()
