import os
import sys
import vtk
from vtkPointCloud import VtkPointCloud
import numpy as np
from utils import read_data, get_points, pc2img
from time import time

from PyQt4.QtCore import *
from PyQt4.QtGui import *

'''
craete a gui to open and visualize a3d file.
'''
# comment this line and use your own initial path
global init_path

init_path = os.getcwd()

class BodyScanWrapper:
    def __init__(self, fpath, threshold=0.1):
        # load point cloud
        self.data = read_data(fpath)
        self.data = self.data / np.amax(self.data)
        self.points = get_points(self.data, threshold).astype(np.int16)

        tensity = np.sort([self.data[tuple(p)] for p in self.points])
        self.th1 =tensity[int(len(tensity) * 0.90)]
        #self.th2 = tensity[int(len(tensity) * 0.95)]
        self.th3 = tensity[int(len(tensity) * 0.99)]

        #colors
        self.c1 = np.array([255, 255, 255])# grey
        # self.c2 = np.array([0, 0, 255]) /4  # blue
        # self.c3 = np.array([0, 255, 0]) /2 # green
        self.c4 = np.array([255, 0, 0]) # red

    def render(self):
        tic = time()
        pointCloud = VtkPointCloud()
        for p in self.points:
            intensity = self.data[tuple(p)]
            if intensity > self.th3:
                pointCloud.addPoint(p, self.c4)
            elif intensity > self.th1:
                pointCloud.addPoint(p, self.c1 * intensity)
            else:
                pass


        pointCloud.postProcess()
        toc = time()
        print('rendering takes {}s'.format(toc - tic))
        return pointCloud

    
if __name__ == '__main__':
    bs = BodyScanWrapper('/home/hugh/code/TSA_Segmentation/pc2convex/a3d/a3d/09a8fdd43905244c6626e15385c9ae22.a3d')
    pc = bs.render()

    # Render
    renderer = vtk.vtkRenderer()
    renderer.AddActor(pc.vtkActor)
    renderer.ResetCamera()

    # Render Window
    renderWindow = vtk.vtkRenderWindow()
    renderWindow.AddRenderer(renderer)

    # Interactor
    renderWindowInteractor = vtk.vtkRenderWindowInteractor()
    renderWindowInteractor.SetRenderWindow(renderWindow)

    # Begin Interaction
    renderWindow.Render()
    renderWindowInteractor.Start()

