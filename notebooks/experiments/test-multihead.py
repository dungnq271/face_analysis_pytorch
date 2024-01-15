# %%
import os
import os.path as osp
import random
import importlib
import time

from PIL import Image
import cv2
import matplotlib.pyplot as plt

import torch
from torchvision import transforms
import predict_custom

importlib.reload(predict_custom)
from reproducible_code.tools import plot
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# %%
genders = ["Female", "Male"]
races = ["Caucasian", "African-American", "Asian", "India", "Other (e.g., hispanic, latino, middle eastern"]

# %%
transform = transforms.Compose(
    [
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ]
)

class Args:
    race_model = "../../weights/Race/final/model_.pth"
    age_model = "../../weights/Age/final/model_.pth"
    gender_model = "../../weights/Gender/final/model_.pth"
    net_mode = "ir_se"
    depth = 50
    drop_ratio = 0.4
    transform = transform
    device = device

args = Args()
predictor = predict_custom.Predictor(
    args.race_model,
    args.gender_model,
    args.age_model,
    args.net_mode,
    args.depth,
    args.drop_ratio,
    args.device,
)

# %%
img_dir = "../../cropped_faces"
img_fns = os.listdir(img_dir)

# %%
img_fn_sample = random.sample(img_fns, 1)[0]
img_fp = osp.join(img_dir, img_fn_sample) 
# face_img = cv2.imread(img_fp)
# face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
face_img = Image.open(img_fp).convert("RGB")
plt.imshow(face_img)

# %%
t1 = time.time()
face_img = transform(face_img).unsqueeze(0)
outputs = predictor.predict(face_img, False)
t2 = time.time()
print(t2-t1)

# %%
races[outputs[0].item()], genders[outputs[1].item()], outputs[2].item()

# %%
# %%
img_fn_sample = random.sample(img_fns, 16)
imgs = []
descs = []

t1 = time.time()

for fn in img_fn_sample:
    img_fp = osp.join(img_dir, fn) 
    # face_img = cv2.imread(img_fp)
    # face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    face_img = Image.open(img_fp).convert("RGB")
    face_img_tensor = transform(face_img).unsqueeze(0)
    outputs = predictor.predict(face_img_tensor, False)

    imgs.append(face_img)
    descs.append(races[outputs[0].item()] + ' ' + genders[outputs[1].item()] + ' ' + str(outputs[2].item()))

t2 = time.time()
print(t2-t1)

plot.display_multiple_images(imgs, titles=descs, axes_pad=1.4)

# %%
