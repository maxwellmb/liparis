from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

import Datacube

class testHandle(FileSystemEventHandler):
    
    def on_created(self, event):
        cube = Datacube.Datacube(event.src_path)
        print('\n Found new file: ' + event.src_path)
        print('Running Power Spectrum Extended algorithm.')
        cube.pse(outDir = '/Users/grossman/Desktop/testOutDir/')
        print('Done with Power Spectrum Extended algorithm on file '+ event.src_path+'.\n')
        print('Running classic algorithm.')
        cube.classic(outDir = '/Users/grossman/Desktop/testOutDir/')
        print('Done with classic algorithm on file '+ event.src_path+'.\n')
        print('Running fourier algorithm on file.')
        cube.isfas(outDir = '/Users/grossman/Desktop/testOutDir/')
        print('Done with fourier algorithm on file '+ event.src_path+'.\n')
        
observer = Observer()
observer.schedule(testHandle(), "/Users/grossman/Desktop/testInDir", recursive=True)
observer.start()

try:
    while observer.is_alive():
        observer.join(1)
except KeyboardInterrupt:
    observer.stop()
observer.join()
