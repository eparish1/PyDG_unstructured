import numpy as np
from evtk.vtk import VtkFile, VtkUnstructuredGrid, VtkTriangle, VtkVertex
def _addDataToFile(vtkFile, cellData, pointData):
    # Point data
    if pointData is not None:
        keys = pointData.keys()
        vtkFile.openData("Point", scalars=keys[0])
        for key in keys:
            data = pointData[key]
            vtkFile.addData(key, data)
        vtkFile.closeData("Point")

    # Cell data
    if cellData is not None:
        keys = list(cellData.keys())
        vtkFile.openData("Cell", scalars=keys[0])
        for key in keys:
            data = cellData[key]
            vtkFile.addData(key, data)
        vtkFile.closeData("Cell")


def _appendDataToFile(vtkFile, cellData, pointData):
    # Append data to binary section
    if pointData is not None:
        keys = pointData.keys()
        for key in keys:
            data = pointData[key]
            vtkFile.appendData(data)

    if cellData is not None:
        keys = cellData.keys()
        for key in keys:
            data = cellData[key]
            vtkFile.appendData(data)


def triangle_faces_to_VTK(filename, x, y, z, faces, point_data, cell_data):
    vertices = (x, y, z)
    x2 = x*1.
    y2 = y*1.
    z2 = z*1.
    vert2 = (x2,y2,z2)
    w = VtkFile(filename, VtkUnstructuredGrid)
    w.openGrid()
    w.openPiece(npoints=len(x), ncells=len(faces))
    w.openElement("Points")
    w.addData("Points", vertices)
    w.closeElement("Points")

    # Create some temporary arrays to write grid topology.
    ncells = len(faces)
    # Index of last node in each cell.
    offsets = np.arange(start=3, stop=3*(ncells + 1), step=3, dtype='uint32')
    # Connectivity as unrolled array.
    connectivity = faces.reshape(ncells*3).astype('int32')
    cell_types = np.ones(ncells, dtype='uint8')*VtkTriangle.tid

    w.openElement("Cells")
    w.addData("connectivity", connectivity)
    w.addData("offsets", offsets)
    w.addData("types", cell_types)
    w.closeElement("Cells")

    _addDataToFile(w, cellData=cell_data, pointData=point_data)

    w.closePiece()
    w.closeGrid()

    w.appendData(vert2)
    w.appendData(connectivity).appendData(offsets).appendData(cell_types)

    _appendDataToFile(w, cellData=cell_data, pointData=point_data)

    w.save()
    return w.getFileName()
