import os
from collections import Counter

base_dir = '/mnt/e/MachineLearning/animal_photos/simple_images'
class_counts = {cls: len(os.listdir(os.path.join(base_dir, cls))) for cls in os.listdir(base_dir)}
print(class_counts)
