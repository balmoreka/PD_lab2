import time
from watchdog.observers import Observer
from watchdog.observers.polling import PollingObserver
from watchdog.events import FileSystemEventHandler
from batch_predict import predict
import os

input_folder = '/'+os.getenv("INPUT_FOLDER")
output_folder = '/'+os.getenv("OUTPUT_FOLDER")
print(input_folder)
print(output_folder)
class MyHandler(FileSystemEventHandler):
    def on_created(self, event):
        input_file = event.src_path
        output_file = output_folder+'/'+input_file[len(input_file)-input_file[::-1].find("/"):]
        print('created')
        print(input_file)
        print(output_file)
        predict(input_file, output_file)

if __name__ == "__main__":
    event_handler = MyHandler()
    observer = PollingObserver()
    observer.schedule(event_handler, path=input_folder, recursive=True)
    print(f'Waiting for files to predict in {input_folder}')
    observer.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:  # keyboard using ctr + c
        observer.stop()
    observer.join()