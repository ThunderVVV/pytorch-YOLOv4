import os
import sys

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def cocoEvaluate(tgtFile, resFile, iouType="bbox", block_print=True):
    """
    Use pycocotools to calculate AP, and it return AP results.
    Args:
        tgtFile(string): The path to annotations json file like "xxx.json".
            Its content format must be same with coco dataset.
            here is an example of annotations json sturcture: 
                {"images": [ 
                    {"file_name": "0",  # [string] no use here
                     "height": xxx,  # [int] image height
                     "width": xxx,  # [int] image width 
                     "id" : xxx,  # [int] image id which must be unique 
                    }, ...],
                 "type": "instances",  # [string] set "instances"
                 "annotations": [
                    {"area": xxx,  # [int] area of bbox
                     "iscrowd": 0,  # [1/0] set 0
                     "image_id": xxx,  # [int] image id which must be unique
                     "bbox": [xmin, ymin, w, h],  # [list] bbox coordinates info 
                     "category_id": xxx,  # [int] category id which should in the range of [0, num_classes - 1]
                     "id": xxx,  # box id which must be unique 
                     "ignore": 0,  # set 0
                     "segmentation": []  # [list] no use here
                    }, ...],
                 "categories": [
                    {"supercategory": "none",  # [string] no use here
                     "id": xxx,  # [int] category id
                     "name": "xxx",  # [string] corresponding category name 
                    }, ...]
                }
        resFile(string): The path to prediction result json file like "xxx.json".
            here is an example of result json sturcture: 
                [{"category_id": xxx,  # [int] category id which should in the range of [0, num_classes - 1] 
                  "image_id": xxx, # [int] image id
                  "bbox_normalized": [xmin_norm, ymin_norm, w_norm, h_norm], # [list] bbox coordinates info normalized
                                                                             # normalization means: 
                                                                             # xmin / width, ymin / height, w / width, h / height
                  "bbox": [xmin, ymin, w, h],  # [list] bbox coordinates info 
                  "score": xxx,  # [float] prediction score
                  "timing": xxx,  # [float] detect time which can be calculated by subtraction of time.time() 
                }, ...]
    Returns:
        status(ndarray): Its shape is (12,). In turn, is AP, AP50, AP75, AP_small, AP_medium, AP_large, 
            AR1, AR10, AR100, AR_small, AR_medium, AR_large

    """
    if block_print:
        sys.stdout = open(os.devnull, 'w')
    try:
        cocoGt = COCO(tgtFile)
        cocoDt = cocoGt.loadRes(resFile)   
        cocoEval = COCOeval(cocoGt, cocoDt, iouType)
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()
        stats = cocoEval.stats
    finally:
        if block_print:
            sys.stdout = sys.__stdout__

    return stats