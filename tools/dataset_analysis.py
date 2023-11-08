import json
from matplotlib import pyplot as plt
import numpy as np

with open('../data/Annotations/TemporalAnnotations/abnormal/train.json') as fp:
    ds = json.load(fp)
durations = []
video_names = []
for video_name, actions in ds.items():
    for action in actions:
        durations.append(action['end'] - action['start'])
        video_names.append(video_name)

bins = [120, 280, 1200, 2400, 4800]
bins = [0] + [2**x for x in range(12)]
values = []
d = np.array(durations)
for i in range(1, len(bins)):
    values.append(np.logical_and(d >= bins[i-1], d < bins[i]).sum())
# values.append(np.logical_and(d >= 0, d <= 120).sum())
# values.append(np.logical_and(d > 120, d <= 280).sum())
# values.append(np.logical_and(d > 600, d <= 1200).sum())
# values.append(np.logical_and(d > 1200, d <= 2400).sum())
# values.append(np.logical_and(d > 2400, d <= 4800).sum())
# values.append(np.logical_and(d > 4800, d <= 10000000).sum())

# Creating histogram
fig, ax = plt.subplots(figsize=(20, 7))
ax.hist(d, bins=bins, rwidth=1)

# Show plot
plt.show()

for i in range(1, len(bins)):
    print('{} - {} : {}'.format(bins[i-1], bins[i], values[i-1]))
print(np.array(values).sum())

# print(max(durations))
# print(video_names[durations.index(max(durations))])
