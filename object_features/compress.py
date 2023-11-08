import os
import pickle
import numpy as np
import sys
from tqdm.autonotebook import tqdm
import queue
import threading
import math
import time


def read_files(q, file_list, output=False):
    if output:
        generator = tqdm(file_list)
    else:
        generator = file_list

    for fn in generator:
        if '.npy' in fn:
            entry = np.load(os.path.join(input_folder, fn))
            q.put(dict(fn=fn, entry=entry))
    q.put('finish')


start_t = time.time()

input_folder = sys.argv[1]
output_file = sys.argv[2]
num_parts = int(sys.argv[3])
num_threads = 20


directory_list = os.listdir(input_folder)
part_size = math.ceil(len(directory_list) / num_parts)

for part_num in range(num_parts):
    q = queue.Queue()
    data = dict()
    threads = []
    finish_threads = 0
    part_data = directory_list[part_num * part_size: (part_num + 1) * part_size]

    for t in range(num_threads):
        num_entries = math.ceil(len(part_data) / num_threads)
        thread_data = part_data[t * num_entries: (t + 1) * num_entries]
        thread = threading.Thread(target=read_files, args=(q, thread_data, t == 0))
        thread.start()
        threads.append(thread)

    while finish_threads != num_threads:
        if not q.empty():
            entry = q.get()
            if entry == 'finish':
                print('finish thread signal')
                finish_threads += 1
            else:
                data[entry['fn']] = entry['entry']
                # print('.')

    for thread in threads:
        thread.join()

    with open("{}.{}".format(output_file, part_num), 'wb') as out_file:
        pickle.dump(data, out_file, protocol=-1)

print(time.time() - start_t)
