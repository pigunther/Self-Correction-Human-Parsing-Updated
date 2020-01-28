import torch
from matplotlib import pyplot as plt

# t = torch.load('./snapshots/events.out.tfevents.1576098400.Uranus')
# print(t.shape)

# with open(path, 'rb') as file:
#     print(file.readline())


from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
path = './snapshots/events.out.tfevents.1576098400.Uranus'

event_acc = EventAccumulator(path)
event_acc.Reload()
# Show all tags in the log file
print(event_acc.Tags())

# E. g. get wall clock, number of steps and value for a scalar 'Accuracy'
images = event_acc.images
print(images)
print(images.Keys())
items = images.Items('Images/')
preds = images.Items('Preds/')
predsEdges = images.Items('PredEdges/')
print(len(items[0]), len(items[1]))
print(len(predsEdges), len(predsEdges[0]))
img = items[0]
with open('img_1.png', 'wb') as f:
  f.write(img.encoded_image_string)

with open('img_pred_1.png', 'wb') as f:
  f.write(preds[0].encoded_image_string)

with open('img_predEdges_1.png', 'wb') as f:
  f.write(predsEdges[0].encoded_image_string)
# w_times, step_nums, vals = zip(*event_acc.Scalars('images'))