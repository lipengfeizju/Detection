import requests
from io import BytesIO
from PIL import Image
import numpy as np

from maskrcnn_benchmark.config import cfg
from predictor import COCODemo

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
matplotlib.use('agg')

def load(url):
    """
    Given an url of an image, downloads the image and
    returns a PIL image
    """
    response = requests.get(url)
    pil_image = Image.open(BytesIO(response.content)).convert("RGB")
    # convert to BGR format
    image = np.array(pil_image)[:, :, [2, 1, 0]]
    return image

def imshow(img):
    fig, ax = plt.subplots(1,figsize=(15, 8))
    ax.imshow(img[:, :, [2, 1, 0]])
    #plt.axis("off")
    plt.savefig('cache/cache_01.png')
    #plt.close()

def load_local(image_path):
    img = plt.imread(img_path)
    image = np.array(img)[:, :, [2, 1, 0]]
    image = (255*image).astype(np.uint8)
    return image


config_file = "configs/caffe2/e2e_mask_rcnn_R_50_FPN_1x_caffe2.yaml"

# update the config options with the config file
cfg.merge_from_file(config_file)
# manual override some options
cfg.merge_from_list(["MODEL.DEVICE", "cpu"])

coco_demo = COCODemo(
    cfg,
    min_image_size=800,
    confidence_threshold=0.2,
)
#image = load("http://farm3.staticflickr.com/2469/3915380994_2e611b1779_z.jpg")

img_path = "/data/diva/diva_data/0002/VIRAT_S_000203_04_000903_001086/00001.png"
image = load_local(img_path)
predictions = coco_demo.run_on_opencv_image(image)

imshow(predictions)

