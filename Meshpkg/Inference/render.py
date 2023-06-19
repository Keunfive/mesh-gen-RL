import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import Meshpkg.params as p
from Meshpkg.Inference.graph import graph_plot


" Volume mesh plot "
def render(volume_mesh, episode):
    plt.clf()
    margin = 0.1
    xmin = round(np.min(volume_mesh[-1][:, 0]),1) - margin
    xmax = round(np.max(volume_mesh[-1][:, 0]),1) + margin
    ymin = round(np.min(volume_mesh[-1][:, 1]),1) - margin
    ymax = round(np.max(volume_mesh[-1][:, 1]),1) + margin

    plt.xlim([xmin, xmax])
    plt.ylim([ymin, ymax])

    for i in range(len(volume_mesh)):
      plt.plot(volume_mesh[i][:, 0], volume_mesh[i][:, 1], linewidth = 0.5, color = p.colormap[i % 5])
      plt.plot( [volume_mesh[i][0, 0], volume_mesh[i][-1, 0]] , [volume_mesh[i][0, 1], volume_mesh[i][-1, 1]], 
               linewidth = 0.5, color = p.colormap[i % 5])
      
      if i != len(volume_mesh) - 1:
        for j in range(len(volume_mesh[0])):
          plt.plot([volume_mesh[i][j, 0], volume_mesh[i + 1][j, 0]], [volume_mesh[i][j, 1], volume_mesh[i + 1][j, 1]], 
                   linewidth=0.5, color=p.colormap[i % 5])

    plt.title(f'Inference (Episode: {episode})')

    if episode != None:
      graph_plot().createFolder('Inference')
      plt.savefig(f'Inference/Training_inf_epi_{episode}.jpg', dpi = 350)

    plt.clf()
    plt.close("all")
