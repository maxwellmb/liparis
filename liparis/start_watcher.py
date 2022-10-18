import matplotlib
matplotlib.use('QT5Agg')
from watchdog.observers import Observer
from watchdog.observers.polling import PollingObserver
from Watcher import HandleNewFile
from Watcher import PatternHandleNewFile
import matplotlib.pyplot as plt

"""
scan_directory = "/home/orkideng/library/liparis/testInDir_1008/"
output_path = "/home/orkideng/library/liparis/testOutDir_1008/"
"""

#Input and output directories, respectively

scan_directory="/s/sdata1651/221017/"
output_path = "/s/sdata1651/tmp_quicklook/"

#Region of Interest parameters

'''
xCent = 1370
xSize = 50
yCent = 760
ySize = 50

'''
xCent = None
xSize = None
yCent = None
ySize = None

fig,axes = plt.subplots(2,4,figsize=(10,5))
axes = axes.flatten()

axes[0].set_title("Shift and add")
axes[1].set_title("Classic")
axes[2].set_title("PSE")
axes[3].set_title("Stack")

plt.draw()
plt.pause(0.001)

#Set up the observer
observer = PollingObserver()

#observer.schedule(HandleNewFile(output_path, fig, axes), scan_directory, recursive=False)

observer.schedule(HandleNewFile(output_path, fig, axes, xCent, xSize, yCent, ySize), scan_directory, recursive = False)

#observer.schedule(PatternHandleNewFile(output_path,fig,axes), scan_directory)

observer.start()
print("starting the observer")
print("Watching directory: " + scan_directory)

#Keep it going. 
try:
    while observer.is_alive():
        observer.join(1)
        plt.draw()
        #plt.pause(.001)
        fig.canvas.flush_events()
except KeyboardInterrupt:
    observer.stop()
observer.join()
