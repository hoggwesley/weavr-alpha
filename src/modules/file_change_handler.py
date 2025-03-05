from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer
import time
import os
from modules.persistent_indexing import get_or_create_persistent_index

class FileChangeHandler(FileSystemEventHandler):
    def __init__(self, knowledge_base_dir, reindex_flag):
        self.knowledge_base_dir = knowledge_base_dir
        self.reindex_flag = reindex_flag

    def on_modified(self, event):
        if not event.is_directory:
            # print(f"File modified: {event.src_path}")
            self.set_reindex_flag()

    def on_created(self, event):
        if not event.is_directory:
            # print(f"File created: {event.src_path}")
            self.set_reindex_flag()

    def on_deleted(self, event):
        if not event.is_directory:
            # print(f"File deleted: {event.src_path}")
            self.set_reindex_flag()

    def set_reindex_flag(self):
        # print("Setting reindex flag due to file change...")
        self.reindex_flag.set()

def start_observer(knowledge_base_dir, reindex_flag):
    event_handler = FileChangeHandler(knowledge_base_dir, reindex_flag)
    observer = Observer()
    observer.schedule(event_handler, path=knowledge_base_dir, recursive=True)
    observer.start()
    print("File change observer started.")
    return observer

def stop_observer(observer):
    observer.stop()
    observer.join()
    print("File change observer stopped.")
