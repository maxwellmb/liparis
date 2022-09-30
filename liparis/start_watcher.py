from watchdog.observers import Observer
from Watcher import HandleNewFile
import matplotlib.pyplot as plt

plt.switch_backend('QT5Agg')

scan_directory = "/Users/maxwellmb/tmp/input/"
output_path = "/Users/maxwellmb/tmp/output/"


fig,axes = plt.subplots(2,4,figsize=(20,10))
axes = axes.flatten()

axes[0].set_title("Shift and add")
axes[1].set_title("Classic")
axes[2].set_title("PSE")
axes[3].set_title("Fourier")

plt.draw()
plt.pause(0.001)

#Set up the observer
observer = Observer()
# observer.schedule(HandleNewFile(output_path), scan_directory, recursive=False)
observer.schedule(HandleNewFile(output_path,fig,axes), scan_directory)
observer.start()
print("starting the observer")

#Keep it going. 
try:
    while observer.is_alive():
        observer.join(1)
        plt.draw()
        plt.pause(.001)
except KeyboardInterrupt:
    observer.stop()
observer.join()