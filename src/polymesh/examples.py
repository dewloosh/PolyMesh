from dewloosh.core.downloads import download_stand
from polymesh import PolyData
from typing import Union


def stand_vtk(read=False) -> Union[str, PolyData]:
    """
    Downloads and optionally reads the stand example as a vtk file.

    Example
    -------
    >>> from polymesh.examples import stand_vtk
    >>> mesh = stand_vtk(read=True)

    """
    vtkpath = download_stand()
    if read:
        return PolyData.read(vtkpath)
    else:
        return vtkpath
