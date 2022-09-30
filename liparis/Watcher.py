from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

import Datacube
class HandleNewFile(FileSystemEventHandler):

    def __init__(self,output_path):
        self.output_path = output_path

    def on_created(self, event):

        cube = Datacube.Datacube(event.src_path)
        print('\n Found new file: ' + event.src_path)
        
        ### PSE Method
        print('Running Power Spectrum Extended algorithm.')
        cube.pse(outDir = self.output_path)
        print('Done with Power Spectrum Extended algorithm on file '+ event.src_path+'.\n')
        
        ### Classic Lucky Imaging
        print('Running classic lucky imaging algorithm.')
        cube.classic(outDir = self.output_path)
        print('Done with classic algorithm on file '+ event.src_path+'.\n')
        
        ### Fourier Lucky Imaging
        print('Running fourier algorithm on file.')
        cube.isfas(outDir = self.output_path)
        print('Done with fourier algorithm on file '+ event.src_path+'.\n')
        



