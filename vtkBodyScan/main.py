import os
import sys
import argparse
import vtk
from vtkPointCloud import VtkPointCloud
import numpy as np
from scipy import io
from utils import read_data, get_points, pc2img
from time import time

from PyQt4.QtCore import *
from PyQt4.QtGui import *

sys.path.append('../npy2mat/')
sys.path.append('../pc2convex')
from npy2mat import mat2npy
import config

'''
craete a gui to open and visualize a3d file.
'''
# comment this line and use your own initial path
global init_path
init_path = os.getcwd()

parser = argparse.ArgumentParser(description='display bodyscan point cloud using vtk. You can use either one setting')
parser.add_argument('--high', action='store_true',
                    help='display points with high intensity value')
parser.add_argument('--label', action='store_true',
                    help='display points colored according to their labels.')
args = parser.parse_args()

class BodyScanWrapper:
    def __init__(self, fpath, threshold=0.1):
        if '.a3d' in fpath:
            # load point cloud
            self.data = read_data(fpath)
            self.data = self.data / np.amax(self.data)
            self.points = get_points(self.data, threshold).astype(np.int16)
        elif '.mat' in fpath:
            with open(fpath, 'r') as fid:
                self.dset = io.loadmat(fid)
                self.points = np.array(self.dset['pts'])
                self.data = mat2npy(self.dset)
        
        if args.high or not args.label:
            #threshold
            self.th0 = threshold
            tensity = np.sort([self.data[tuple(p)] for p in self.points])
            self.th1 =tensity[int(len(tensity) * 0.90)]
            self.th2 = tensity[int(len(tensity) * 0.95)]
            self.th3 = tensity[int(len(tensity) * 0.99)]

            #colors
            self.c1 = np.array([255, 255, 255]) * 0.1 # grey
            self.c2 = np.array([0, 0, 255]) /4  # blue
            self.c3 = np.array([0, 255, 0]) /2 # green
            self.c4 = np.array([255, 0, 0]) # red
        elif args.label:
            if '.mat' not in fpath:
                raise ValueError('to display the label. The input file must be .a3d.')
            self.intensity = np.array(self.dset['intensity']).ravel()
            self.labels = np.array(self.dset['labels']).ravel()      
            assert len(self.labels) == len(self.points)
            
    def render(self):
        tic = time()
        pointCloud = VtkPointCloud()
        if args.high or not args.label:
            for p in self.points:
                if self.data[tuple(p)] > self.th3:
                    pointCloud.addPoint(p, self.c4)
                elif self.data[tuple(p)] > self.th2:
                    pointCloud.addPoint(p, self.c3)
                elif self.data[tuple(p)] > self.th1:
                    pointCloud.addPoint(p, self.c2)
                elif self.data[tuple(p)] > self.th0:
                    #pointCloud.addPoint(p, self.c1)
                    pass
                else:
                    pass
        elif args.label:
            for p, l in zip(self.points, self.labels):
                if l >= 0:
                    pointCloud.addPoint(p, config.colors_plt[l])
                    
        pointCloud.postProcess()
        toc = time()
        print('rendering takes {}s'.format(toc - tic))
        return pointCloud

    
if __name__ == '__main__':
    try:
        # select a3d file
        # Create an PyQT4 application object.
        a = QApplication(sys.argv)
    
        # The QWidget widget is the base class of all user interface objects in PyQt4.
        w = QWidget()
    
        # Set window size. 
        w.resize(320, 240)
        
        # Set window title 
        w.setWindowTitle("vtkBodyScan")
        
        # Get filename using QFileDialog
        
        fileName = QFileDialog.getOpenFileName(w, 'Open File', init_path)
        print(fileName + ' is selected')


        if not os.path.exists(fileName):
            print('Invalid file path')

        bs = BodyScanWrapper(fileName)
        # bs = BodyScanWrapper('/home/hugh/code/TSA_Segmentation/pc2convex/a3d/a3d/09a8fdd43905244c6626e15385c9ae22.a3d')
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
    except KeyboardInterrupt:
        a.exit(0)
