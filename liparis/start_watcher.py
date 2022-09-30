from watchdog.observers import Observer
from Watcher import HandleNewFile

scan_directory = "/Users/maxwellmb/tmp/"
output_path = "/Users/maxwellmb/tmp/output/"

#Set up the observer
observer = Observer()
observer.schedule(HandleNewFile(output_path), scan_directory, recursive=True)
observer.start()

#Keep it going. 
try:
    while observer.is_alive():
        observer.join(1)
except KeyboardInterrupt:
    observer.stop()
observer.join()