from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import pylab

pylab.rcParams['figure.figsize'] = (8.0, 10.0)
def showCocoAnnotations(inputDir, annFile):
    coco = COCO(annFile)
    # display COCO categories and supercategories
    cats = coco.loadCats(coco.getCatIds())
    nms=[cat['name'] for cat in cats]
    print('COCO categories: \n{}\n'.format(' '.join(nms)))

    # nms = set([cat['supercategory'] for cat in cats])
    # print('COCO supercategories: \n{}'.format(' '.join(nms)))

    catIds = coco.getCatIds(catNms=['__background__',
        'guard rail',
        'car',
        'dashed',
        'solid',
        'solid solid',
        'dashed dashed',
        'dashed-solid',
        'solid-dashed',
        'yellow dashed',
        'yellow solid',
        'yellow solid solid',
        'yellow dashed dashed',
        'yellow dashed-solid',
        'yellow solid-dashed',
        'boundary'
                                    ])

    # catIds = coco.getCatIds(catNms=['__background__',
    #                           'ego vehicle',
    #                           'rectification border',
    #                           'out of roi',
    #                           'static',
    #                           'dynamic',
    #                           'ground',
    #                           'road',
    #                           'sidewalk',
    #                           'parking',
    #                           'rail track',
    #                           'building',
    #                           'wall',
    #                           'fence',
    #                           'guard rail',
    #                           'bridge',
    #                           'tunnel',
    #                           'pole',
    #                           'polegroup',
    #                           'traffic light',
    #                           'traffic sign',
    #                           'vegetation',
    #                           'terrain',
    #                           'sky',
    #                           'person',
    #                           'rider',
    #                           'car',
    #                           'truck',
    #                           'bus',
    #                           'caravan',
    #                           'trailer',
    #                           'train',
    #                           'motorcycle',
    #                           'bicycle',])

    # catIds = coco.getCatIds(catNms=['__background__',
    #     "bike",
    #     "bus",
    #     "car",
    #     # "motor",
    #     "person",
    #     "rider",
    #     "traffic light",
    #     "traffic sign",
    #     # "train",
    #     "truck",
    #     "area/alternative",
    #     "area/drivable"])
    # imgIds = coco.getImgIds(imgIds=[0,1,2,3,4], catIds=catIds);
    imgIds = coco.getImgIds(imgIds=[21])
    # imgIds = coco.getImgIds(imgIds=[0])
    img = coco.loadImgs(imgIds[np.random.randint(0, len(imgIds))])[0]



    # load and display image
    # I = io.imread('%s/images/%s/%s'%(dataDir,dataType,img['file_name']))
    # use url to load image
    # I = io.imread(img['coco_url'])
    print ("img['file_name']:" + img['file_name'])
    I = io.imread(inputDir + img['file_name'])
    # plt.axis('off')
    # plt.imshow(I)
    # plt.show()

    # load and display instance annotations
    plt.imshow(I);
    plt.axis('off')
    annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
    anns = coco.loadAnns(annIds)
    coco.showAnns(anns)
    plt.show()

if __name__ == '__main__':
    # inputDir = "/media/administrator/deeplearning/dataset/cityscape/leftImg8bit/val_full/"
    # annFile = "/media/administrator/deeplearning/dataset/cityscape/output/instancesonly_filtered_image_val.json"

    inputDir = "/media/administrator/deeplearning/self-labels/leftImg8bit/train/"
    annFile = "/media/administrator/deeplearning/self-labels/output/instancesonly_filtered_image_train.json"

    # inputDir = "/media/administrator/deeplearning/dataset/test_cityscape/image/train/aachen/"
    # annFile = "/media/administrator/deeplearning/dataset/test_cityscape/output/instancesonly_filtered_image_train.json"
    showCocoAnnotations(inputDir, annFile)
