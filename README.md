# PyDG_unstructured
Basic unstructured 2D DG code.
Requires numpy and scipy. Uses a tpl to write solution to .vtk files that can be visualized by paraview

Steps to install tpl

```bash
cd tpls/pauloh-pyevtk-e253ef56687d
python setup.py install
```

Run examples with, e.g.,

```bash
cd examples/shallowWaterEquations
python main.py
```
