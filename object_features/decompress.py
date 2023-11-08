import os
import pickle
import numpy as np
import sys
from tqdm.autonotebook import tqdm
import queue
import threading
import math
import time


def write_files(q, out_folder):
    stop = False
    while not stop:
        if not q.empty():
            entry = q.get()
            if entry == 'finish':
                stop = True
            else:
                filepath = os.path.join(out_folder, entry['fp'])
                np.save(filepath, entry['entry'])

    print('thread has been finished')


start_t = time.time()

input_file = sys.argv[1]
output_folder = sys.argv[2]

num_threads = 20
q = queue.Queue(maxsize=num_threads*5)
threads = []

with open(input_file, 'rb') as fp:
    data = pickle.load(fp)

for t in range(num_threads):
    thread = threading.Thread(target=write_files, args=(q, output_folder))
    thread.start()
    threads.append(thread)


for fp, entry in tqdm(data.items()):
    while q.full():
        time.sleep(0.001)

    q.put(dict(fp=fp, entry=entry))

for _ in threads:
    q.put('finish')

for t in threads:
    t.join()

print(time.time() - start_t)
