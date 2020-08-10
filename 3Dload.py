#%%
import numpy as np
import plotly.graph_objects as go
import scipy.ndimage
import pywavefront
from objects import binvox_rw

folder = 'objects/3Dgraphics/'
# filename = 'teapot.binvox'
filename = 'PotatoSingle.binvox'

with open(folder+filename, 'rb') as f:
    model = binvox_rw.read_as_3d_array(f)


folder_out = 'objects/generated/'
np.save(folder_out+filename.split('.')[0], model.data)

# %%
# Helix equation
data = model.data*100
span = 100
values = data[0:span,0:span,0:span]

import plotly.graph_objects as go

X, Y, Z = np.mgrid[0:span, 0:span, 0:span]


fig = go.Figure(data=go.Volume(
    x=X.flatten(),
    y=Y.flatten(),
    z=Z.flatten(),
    value=values.flatten(),
    isomin=0.1,
    isomax=0.8,
    opacity=0.1,  # needs to be small to see through all surfaces
    surface_count=17,  # needs to be a large number for good volume rendering
))
fig.show()
# %%
