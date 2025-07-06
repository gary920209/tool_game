from __future__ import division, print_function
import pymunk as pm
import pygame as pg
import numpy as np
import cv2
from scipy.stats import multivariate_normal as mvnm
from ..world import *
from ..constants import *
from ..object import *
from pygame.constants import *
from PIL import Image
#from .visualize_likelihoods import *
import pdb
__all__ = ['drawWorld', 'demonstrateWorld', 'demonstrateTPPlacement',
           'visualizePath', 'drawPathSingleImage', 'drawPathSingleImageWithTools', 'drawWorldWithTools', 'visualizeScreen', 'drawPathSingleImageBasic',
           'makeImageArray', 'makeImageArrayNoPath','drawTool', '_draw_line_gradient',
           'drawMultiPathSingleImage', 'drawMultiPathSingleImageBasic', 'demonstrateWorld_and_save_video']

COLORS=[(255,0,255,255), (225,225,0, 255),(0, 255, 255, 255)]
WHITE = (255, 255, 255, 255)
def _lighten_rgb(rgba, amt=.2):
    assert 0 <= amt <= 1, "Lightening must be between 0 and 1"
    r = int(255- ((255-rgba[0]) * (1-amt)))
    g = int(255- ((255-rgba[1]) * (1-amt)))
    b = int(255- ((255-rgba[2]) * (1-amt)))
    if len(rgba) == 3:
        return (r, g, b)
    else:
        return (r, g, b, rgba[3])

def _draw_line_gradient(start, end, steps, rgba, surf):
    diffs = np.array(end) - np.array(start)
    dX = (end[0] - start[0]) / steps
    dY = (end[1] - start[1]) / steps

    points = np.array(start) + np.array([[dX,dY]])*np.array([range(0,steps),]*2).transpose()
    cols = [_lighten_rgb(rgba, amt=0.9*step/steps) for step in range(0, steps)]
    for i, point in enumerate(points[:-1]):
        pg.draw.line(surf, cols[i], point, points[i+1], 3)
    return surf

def _filter_unique(mylist):
    newlist = []
    for ml in mylist:
        if ml not in newlist:
            newlist.append(ml)
    return newlist

def _draw_obj(o, s, makept, lighten_amt=0):
    if o.type == 'Poly':
        vtxs = [makept(v) for v in o.vertices]
        col = _lighten_rgb(o.color, lighten_amt)
        pg.draw.polygon(s, col, vtxs)
    elif o.type == 'Ball':
        pos = makept(o.position)
        rad = int(o.radius)
        col = _lighten_rgb(o.color, lighten_amt)
        pg.draw.circle(s, col, pos, rad)
        # Draw small segment that adds a window
        rot = o.rotation
        mixcol = [int((3.*oc + 510.)/5.) for oc in o.color]
        mixcol = _lighten_rgb(mixcol, lighten_amt)
        for radj in range(5):
            ru = radj*np.pi / 2.5 + rot
            pts = [(.65*rad*np.sin(ru) + pos[0], .65*rad*np.cos(ru) + pos[1]),
                   (.7 * rad * np.sin(ru) + pos[0], .7 * rad * np.cos(ru) + pos[1]),
                   (.7 * rad * np.sin(ru+np.pi/20.) + pos[0], .7 * rad * np.cos(ru+np.pi/20.) + pos[1]),
                   (.65 * rad * np.sin(ru+np.pi/20.) + pos[0], .65 * rad * np.cos(ru+np.pi/20.) + pos[1])]
            pg.draw.polygon(s, mixcol, pts)
    elif o.type == 'Segment':
        pa, pb = [makept(p) for p in o.points]
        col = _lighten_rgb(o.color, lighten_amt)
        pg.draw.line(s, col, pa, pb, o.r)
    elif o.type == 'Container':
        for poly in o.polys:
            ocol = col = _lighten_rgb(o.outer_color, lighten_amt)
            vtxs = [makept(p) for p in poly]
            pg.draw.polygon(s, ocol, vtxs)
        garea = [makept(p) for p in o.vertices]
        if o.inner_color is not None:
            acolor = (o.inner_color[0], o.inner_color[1], o.inner_color[2], 128)
            acolor = _lighten_rgb(acolor, lighten_amt)
            pg.draw.polygon(s, acolor, garea)
    elif o.type == 'Compound':
        col = _lighten_rgb(o.color, lighten_amt)
        for poly in o.polys:
            vtxs = [makept(p) for p in poly]
            pg.draw.polygon(s, col, vtxs)
    elif o.type == 'Goal':
        if o.color is not None:
            col = _lighten_rgb(o.color, lighten_amt)
            vtxs = [makept(v) for v in o.vertices]
            pg.draw.polygon(s, col, vtxs)
    else:
        print ("Error: invalid object type for drawing:", o.type)

def _draw_tool(toolverts, makept, size=[90, 90], color=(0,0,0,255)):
    s = pg.Surface(size)
    s.fill(WHITE)
    for poly in toolverts:
        vtxs = [makept(p) for p in poly]
        pg.draw.polygon(s, color, vtxs)
    return s

def drawWorld(world, backgroundOnly=False, lightenPlaced=False):
    s = pg.Surface(world.dims)
    s.fill(world.bk_col)

    def makept(p):
        return [int(i) for i in world._invert(p)]

    for b in world.blockers.values():
        drawpts = [makept(p) for p in b.vertices]
        pg.draw.polygon(s, b.color, drawpts)

    for o in world.objects.values():
        if not backgroundOnly or o.isStatic():
            if lightenPlaced and o.name == 'PLACED':
                _draw_obj(o, s, makept, .5)
            else:
                _draw_obj(o, s, makept)
    return s

def drawTool(tool):

    def maketoolpt(p):
        return [int(p[0] + 45), int(45-p[1])]

    s = _draw_tool(tool, maketoolpt, [90,90])

    return s

def drawWorldWithTools(tp, backgroundOnly=False, worlddict=None):
    if worlddict is not None:
        world = loadFromDict(worlddict)
    else:
        world = loadFromDict(tp._worlddict)
    s = pg.Surface((world.dims[0] + 150, world.dims[1]))
    s.fill(world.bk_col)

    def makept(p):
        return [int(i) for i in world._invert(p)]

    def maketoolpt(p):
        return [int(p[0] + 45), int(45-p[1])]

    for b in world.blockers.values():
        drawpts = [makept(p) for p in b.vertices]
        pg.draw.polygon(s, b.color, drawpts)

    for o in world.objects.values():
        if not backgroundOnly or o.isStatic():
            _draw_obj(o, s, makept)

    for i, t in enumerate(tp._tools.keys()):
        col = COLORS[i]
        newsc = pg.Surface([96, 96])
        newsc.fill(col)
        toolsc = _draw_tool(tp._tools[t], maketoolpt, [90,90])
        newsc.blit(toolsc, [3, 3])
        s.blit(newsc, (630, 137 + 110*i))
    return s

def demonstrateWorld(world, hz = 30.):
    pg.init()
    sc = pg.display.set_mode(world.dims)
    clk = pg.time.Clock()
    sc.blit(drawWorld(world), (0,0))
    pg.display.flip()
    running = True
    tps = 1./hz
    clk.tick(hz)
    dispFinish = True        
    while running:
        world.step(tps)
        sc.blit(drawWorld(world), (0, 0))
        pg.display.flip()
        clk.tick(hz)
        for e in pg.event.get():
            if e.type == pg.QUIT:
                running = False
        if dispFinish and world.checkEnd():
            print("Goal accomplished")
            dispFinish = False
    pg.quit()

from PIL import Image
import pygame as pg
import os

def demonstrateWorld_and_save_video(world, video_filename="simulation.mp4", hz=30., max_frames=5):
    print(f"🎬 Starting MP4 video generation: {video_filename}")
    print(f"   Parameters: hz={hz}, max_frames={max_frames}")
    
    pg.init()
    sc = pg.display.set_mode(world.dims)
    clk = pg.time.Clock()
    tps = 1. / hz
    frames = []

    # Always capture the initial state
    sc.blit(drawWorld(world), (0, 0))
    pg.display.flip()
    raw_data = pg.image.tostring(sc, 'RGB')  
    img = Image.frombytes('RGB', sc.get_size(), raw_data)
    # Convert PIL image to numpy array for OpenCV
    frame_array = np.array(img)
    frames.append(frame_array)
    print(f"   Captured initial frame (total: {len(frames)})")

    # 只錄製指定數量的幀或直到動作結束
    for frame_idx in range(max_frames - 1):  # -1 因為已經有初始幀
        world.step(tps)
        sc.blit(drawWorld(world), (0, 0))
        pg.display.flip()
        clk.tick(hz)

        # screen to PIL Image then to numpy array
        raw_data = pg.image.tostring(sc, 'RGB')
        img = Image.frombytes('RGB', sc.get_size(), raw_data)
        frame_array = np.array(img)
        frames.append(frame_array)
        
        print(f"   Captured frame {len(frames)}/{max_frames}")

        # 檢查動作是否完成
        if world.checkEnd():
            print(f"Goal accomplished at frame {len(frames)}")
            break

    pg.quit()

    # save frames as MP4 video
    if frames and len(frames) > 1:
        height, width, layers = frames[0].shape
        
        # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(video_filename, fourcc, hz, (width, height))
        
        for frame in frames:
            # OpenCV uses BGR, but our frames are RGB, so we need to convert
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(frame_bgr)
        
        out.release()
        print(f"Simulation MP4 saved to {video_filename} with {len(frames)} frames at {hz} fps")
    else:
        print(f"Warning: Only {len(frames)} frame(s) captured. Creating static image instead.")
        if frames:
            frame_pil = Image.fromarray(frames[0])
            frame_pil.save(video_filename.replace('.mp4', '.png'))
            print(f"Static image saved to {video_filename.replace('.mp4', '.png')}")


def _draw_tool(surface, tool_points, topleft, color=(255, 0, 0)):
    """
    Draw a polygon tool on a surface at a specific topleft position.
    tool_points: [[[x1, y1], [x2, y2], ...]]
    """
    # Adjust the topleft position to account for the tool's size
    topleft_adjusted = (topleft[0], topleft[1] + 35)
    
    for poly in tool_points:
        shifted = [[pt[0] + topleft_adjusted[0], pt[1] + topleft_adjusted[1]] for pt in poly]
        pg.draw.polygon(surface, color, shifted, width=0)
        pg.draw.polygon(surface, (0, 0, 0), shifted, width=2)

def saveWorld(world, filename,tools):
    """
    Save the world to a file in JSON format.
    """
    pg.init()
    sc = pg.display.set_mode(world.dims)
    sc.blit(drawWorld(world), (0, 0))
        # If tools are provided, draw them in upper right
    if tools:
        margin = 10
        tool_width = 80
        tool_height = 60
        font = pg.font.SysFont(None, 20)
        i = 0
        for toolname, toolpoints in tools.items():
            x = world.dims[0] - tool_width - margin
            y = margin + i * (tool_height + margin)
            _draw_tool(sc, toolpoints, topleft=(x, y))
            label = font.render(toolname, True, (0, 0, 0))
            sc.blit(label, (x, y + tool_height + 2))
            i += 1

    pg.image.save(sc, filename)
    data = pg.image.tostring(sc, 'RGBA')
    pil_img = Image.frombytes('RGBA', sc.get_size(), data)  
    print(f"World saved to {filename}")
    pg.quit()
    return pil_img

def demonstrateTPPlacement(toolpicker, toolname, position, path, maxtime=20.,
                           noise_dict=None, hz=30.):
    tps = 1./hz
    toolpicker.bts = tps
    if noise_dict:
        pth, ocm, etime, wd = toolpicker.runFullNoisyPath(toolname, position, maxtime, returnDict=True, **noise_dict)
    else:
        pth, ocm, etime, wd = toolpicker.observeFullPlacementPath(toolname, position, maxtime, returnDict=True)
    
    # Handle the case where path is None (invalid action)
    if pth is None or wd is None:
        print("Path or world data is None - creating static image instead")
        # Create a static image of the current world state
        world = loadFromDict(toolpicker.world.genDesc())
        pg.init()
        sc = pg.display.set_mode(world.dims)
        sc.blit(drawWorld(world), (0, 0))
        pg.display.flip()
        pg.image.save(sc, path)
        pg.quit()
        return
    
    world = loadFromDict(wd)
    print (ocm)
    pg.init()
    sc = pg.display.set_mode(world.dims)
    clk = pg.time.Clock()
    sc.blit(drawWorld(world), (0, 0))
    pg.display.flip()
    clk.tick(hz)
    t = 0
    i = 0
    dispFinish = True
    while t < etime and i < len(pth[list(pth.keys())[0]][0]):
        for onm, o in world.objects.items():
            if not o.isStatic() and onm in pth:
                if len(pth[onm]) >= 2 and len(pth[onm][0]) > i:
                    o.setPos(pth[onm][0][i])
                    o.setRot(pth[onm][1][i])
        i += 1
        t += tps
        sc.blit(drawWorld(world), (0,0))
        pg.display.flip()
        for e in pg.event.get():
            if e.type == pg.QUIT:
                pg.quit()
                return
    pg.image.save(sc, path)  
    pg.quit()

def visualizePath(worlddict, path, hz=30.):
    world = loadFromDict(worlddict)
    pg.init()
    sc = pg.display.set_mode(world.dims)
    clk = pg.time.Clock()
    sc.blit(drawWorld(world), (0, 0))
    pg.display.flip()
    clk.tick(hz)
    if len(path[(list(path.keys())[0])]) == 2:
        nsteps = len(path[list(path.keys())[0]][0])
    else:
        nsteps = len(path[list(path.keys())[0]])

    for i in range(nsteps):
        for onm, o in world.objects.items():
            if not o.isStatic():
                if len(path[onm])==2:
                    o.setPos(path[onm][0][i])
                    o.setRot(path[onm][1][i])
                else:
                    o.setPos(path[onm][i][0:2])
                    #o.setRot(path[onm][i][2])
        sc.blit(drawWorld(world), (0,0))
        pg.display.flip()
        for e in pg.event.get():
            if e.type == pg.QUIT:
                pg.quit()
                return
        clk.tick(hz)
    pg.quit()

def makeImageArray(worlddict, path, sample_ratio=1):
    world = loadFromDict(worlddict)
    #pg.init()
    images = [drawWorld(world)]
    if len(path[(list(path.keys())[0])]) == 2:
        nsteps = len(path[list(path.keys())[0]][0])
    else:
        nsteps = len(path[list(path.keys())[0]])

    for i in range(1,nsteps,sample_ratio):
        for onm, o in world.objects.items():
            if not o.isStatic():
                if len(path[onm])==2:
                    o.setPos(path[onm][0][i])
                    o.setRot(path[onm][1][i])
                else:
                    o.setPos(path[onm][i][0:2])
                    o.setRot(path[onm][i][2])
        images.append(drawWorld(world))
    return images

def makeImageArrayNoPath(worlddict, path_length):
    world = loadFromDict(worlddict)
    #pg.init()
    images = [drawWorld(world)]
    nsteps = path_length
    return images*int(nsteps)

def visualizeScreen(tp):
    #pg.init()
    pg.display.set_mode((10,10))
    s = drawWorldWithTools(tp, backgroundOnly=False)
    i = s.convert_alpha()
    pg.image.save(i, 'test.png')
    pg.quit()

def drawPathSingleImageWithTools(tp, path, pathSize=3, lighten_amt=.5, worlddict=None, with_tools=False):
    # set up the drawing
    if worlddict is None:
        worlddict = tp._worlddict
    world = loadFromDict(worlddict)
    #pg.init()
    #sc = pg.display.set_mode(world.dims)
    if not with_tools:
        sc = drawWorld(world, backgroundOnly=True)#, worlddict=worlddict)
    else:
        sc = drawWorldWithTools(tp, backgroundOnly=True, worlddict=worlddict)
    def makept(p):
        return [int(i) for i in world._invert(p)]
    # draw the paths in the background
    for onm, o in world.objects.items():
        if not o.isStatic():
            if o.type == 'Container':
                col = o.outer_color
            else:
                col = o.color
            pthcol = _lighten_rgb(col, lighten_amt)
            if len(path[onm]) == 2:
                poss = path[onm][0]
            else:
                poss = [path[onm][i][0:2] for i in range(0, len(path[onm]))]
            #for p in poss:
            #    pg.draw.circle(sc, pthcol, makept(p), pathSize)
            pts = _filter_unique([makept(p) for p in poss])

            if len(pts) > 1:
                steps = len(pts)
                cols = [_lighten_rgb(col, amt=0.9*step/steps) for step in range(0, steps)]
                for i,pt in enumerate(pts[:-1]):
                    color = cols[i]
                    pg.draw.line(sc, color, pt, pts[i+1], 3)
                    #_draw_line_gradient(pt, pts[i+1], 5, col, sc)
                #pg.draw.lines(sc, pthcol, False, pts, pathSize)
    # Draw the initial tools, lightened
    for onm, o in world.objects.items():
        if not o.isStatic():
            _draw_obj(o, sc, makept, lighten_amt=lighten_amt)
    # Draw the end tools
    for onm, o in world.objects.items():
        if not o.isStatic():
            if len(path[onm])==2:
                o.setPos(path[onm][0][-1])
                o.setRot(path[onm][1][-1])
            else:
                o.setPos(path[onm][-1][0:2])
            _draw_obj(o, sc, makept)

    return sc

def drawPathSingleImage(worlddict, path, pathSize=3, lighten_amt=.5):
    # set up the drawing
    world = loadFromDict(worlddict)
    sc = drawWorld(world, backgroundOnly=True)
    def makept(p):
        return [int(i) for i in world._invert(p)]
    # draw the paths in the background
    for onm, o in world.objects.items():
        if not o.isStatic():
            if o.type == 'Container':
                col = o.outer_color
            else:
                col = o.color
            pthcol = _lighten_rgb(col, lighten_amt)
            if len(path[onm]) == 2:
                poss = path[onm][0]
            else:
                poss = [path[onm][i][0:2] for i in range(0, len(path[onm]))]
            #for p in poss:
            #    pg.draw.circle(sc, pthcol, makept(p), pathSize)
            pts = _filter_unique([makept(p) for p in poss])
            if len(pts) > 1:
                pg.draw.lines(sc, pthcol, False, pts, pathSize)
    # Draw the initial tools, lightened
    for onm, o in world.objects.items():
        if not o.isStatic():
            _draw_obj(o, sc, makept, lighten_amt=lighten_amt)
    # Draw the end tools
    for onm, o in world.objects.items():
        if not o.isStatic():
            if len(path[onm])==2:
                o.setPos(path[onm][0][-1])
                o.setRot(path[onm][1][-1])
            else:
                o.setPos(path[onm][-1][0:2])
            _draw_obj(o, sc, makept)

    return sc


def drawMultiPathSingleImage(worlddict, path_set, pathSize=3, lighten_amt=.5):
    # set up the drawing
    world = loadFromDict(worlddict)

    sc = drawWorld(world, backgroundOnly=True)
    def makept(p):
        return [int(i) for i in world._invert(p)]
    # draw the paths in the background
    for path in path_set:
        for onm, o in world.objects.items():
            if not o.isStatic():
                if o.type == 'Container':
                    col = o.outer_color
                else:
                    col = o.color
                pthcol = _lighten_rgb(col, lighten_amt)
                if len(path[onm]) == 2:
                    poss = path[onm][0]
                else:
                    poss = [path[onm][i][0:2] for i in range(0, len(path[onm]))]

                pts = _filter_unique([makept(p) for p in poss])
                if len(pts) > 1:
                    pg.draw.lines(sc, pthcol, False, pts, pathSize)
    # Draw the initial tools, lightened
    for onm, o in world.objects.items():
        if not o.isStatic():
            _draw_obj(o, sc, makept, lighten_amt=lighten_amt)
    # Draw the end tools
    for path in path_set:
        for onm, o in world.objects.items():
            if not o.isStatic():
                if len(path[onm])==2:
                    o.setPos(path[onm][0][-1])
                    o.setRot(path[onm][1][-1])
                else:
                    o.setPos(path[onm][-1][0:2])
                _draw_obj(o, sc, makept)

    return sc

def drawPathSingleImageBasic(sc, world, path, pathSize=3, lighten_amt=.5):
    # set up the drawing
    def makept(p):
        return [int(i) for i in world._invert(p)]
    # draw the paths in the background
    for onm, o in world.objects.items():
        if not o.isStatic():
            if o.type == 'Container':
                col = o.outer_color
            else:
                col = o.color
            pthcol = _lighten_rgb(col, lighten_amt)
            if len(path[onm]) == 2:
                poss = path[onm][0]
            else:
                poss = path[onm]
            #for p in poss:
            #    pg.draw.circle(sc, pthcol, makept(p), pathSize)
            pts = _filter_unique([makept(p) for p in poss])
            if len(pts) > 1:
                pg.draw.lines(sc, pthcol, False, pts, pathSize)
    # Draw the initial tools, lightened
    for onm, o in world.objects.items():
        if not o.isStatic():
            _draw_obj(o, sc, makept, lighten_amt=lighten_amt)
    # Draw the end tools
    for onm, o in world.objects.items():
        if not o.isStatic():
            if len(path[onm])==2:
                o.setPos(path[onm][0][-1])
                o.setRot(path[onm][1][-1])
            else:
                o.setPos(path[onm][-1])
            _draw_obj(o, sc, makept)
    return sc


def drawMultiPathSingleImageBasic(sc, world, path_set, pathSize=3, lighten_amt=.5):
    # set up the drawing
    def makept(p):
        return [int(i) for i in world._invert(p)]
    # draw the paths in the background
    for path in path_set:
        for onm, o in world.objects.items():
            if not o.isStatic():
                if o.type == 'Container':
                    col = o.outer_color
                else:
                    col = o.color
                pthcol = _lighten_rgb(col, lighten_amt)
                if len(path[onm]) == 2:
                    poss = path[onm][0]
                else:
                    poss = path[onm]
                #for p in poss:
                #    pg.draw.circle(sc, pthcol, makept(p), pathSize)
                pts = _filter_unique([makept(p) for p in poss])
                if len(pts) > 1:
                    pg.draw.lines(sc, pthcol, False, pts, pathSize)
    # Draw the initial tools, lightened
    for onm, o in world.objects.items():
        if not o.isStatic():
            _draw_obj(o, sc, makept, lighten_amt=lighten_amt)
    # Draw the end tools
    for path in path_set:
        for onm, o in world.objects.items():
            if not o.isStatic():
                if len(path[onm]) == 2:
                    o.setPos(path[onm][0][-1])
                    o.setRot(path[onm][1][-1])
                else:
                    o.setPos(path[onm][-1])
                _draw_obj(o, sc, makept)
    return sc

def drawTool(tool, color=(0,0,255), toolbox_size=(90, 90)):
    s = pg.Surface(toolbox_size)
    def resc(p):
        return [int(p[0] +toolbox_size[0]/2),
                int(toolbox_size[1]/2 - p[1])]
    s.fill((255,255,255))
    for poly in tool:
        pg.draw.polygon(s, color, [resc(p) for p in poly])

    s_arr = pg.surfarray.array3d(s)
    return s_arr

def _def_inv(p):
    return(p)

