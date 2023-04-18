from snake_ai.physim.simulation import Simulation
from typing import List, Optional
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pathlib import Path
from phi import flow

def animate_walker_history(field : flow.CenteredGrid, history : List[flow.Tensor], output : Optional[str] = None):
    fig, ax = plt.subplots()
    ax.imshow(field.values.numpy('y,x'))
    ax.set(xlabel='x', ylabel='y', title='Concentration map + deterministic walkers')
    anim = []
    for point_cloud in history:
        points = point_cloud.numpy('point,vector')
        # anim.append(flow.plot(point_cloud, animate=True))
        anim.append(ax.plot(points[:,0], points[:,1], marker="o", ls="", animated=True, color='orange'))
    anim = animation.ArtistAnimation(fig, anim, blit=True, interval=200)
    if output is not None:
        output = Path(output).resolve()
        anim.save(output)

def animate_simulation_history(simulation : Simulation, output : str = "video.mp4"):
    pass
