import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
print(os.getcwd())
print("Executing ...")

import torch, torchvision
print(torch.__version__, torch.cuda.is_available())

import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

import numpy as np
import cv2
import random

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.data.catalog import DatasetCatalog

print(detectron2.__version__)
#register_coco_instances("YourTrainDatasetName", {},"path to train.json", "path to train image folder")

from detectron2.data.datasets import register_coco_instances
subset = "mydata"
register_coco_panoptic_separated(subset+"_"+tr1.backbone_config, {}, 
        join(tr1.dataset_dir,"images/"+subset), 
        join(tr1.dataset_dir,"masks/"+subset), join('/content/mydata/anns', subset+".json"), 
        join(tr1.dataset_dir,"masks/"+subset), join('/content/mydata/anns', subset+".json"))

#register_coco_instances("TRY_train", {}, "/content/TRY/annot/github.json", "/content/TRY/train")
#register_coco_instances("TRY_val", {}, "/content/TRY/train/", "./content/TRY/annot/github.json")
#register_coco_instances("TRY_test", {}, "/content/TRY/train/", "./content/TRY/annot/github.json")

MetadataCatalog.get("mydata").thing_classes = ["g","k","h"]
#MetadataCatalog.get("my_dataset_test").thing_classes = ["Eclosed egg", "Egg", "Larvae", "Lump"]

#visualize training data
my_dataset_train_metadata = MetadataCatalog.get("mydata_train")
dataset_dicts = DatasetCatalog.get("mydata_train")

import random
from detectron2.utils.visualizer import Visualizer

# Our trainer
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator

class CocoTrainer(DefaultTrainer):

  @classmethod
  def build_evaluator(cls, cfg, dataset_name, output_folder=None):

    if output_folder is None:
        os.makedirs("coco_eval", exist_ok=True)
        output_folder = "coco_eval"

    return COCOEvaluator(dataset_name, cfg, False, output_folder)

# Setup model to train
model_yaml_path = "COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"
model_yaml_path = "COCO-Detection/retinanet_R_101_FPN_3x.yaml"
model_yaml_path = "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
model_yaml_path = "COCO-PanopticSegmentation/panoptic_fpn_R_50_3x.yaml"
from detectron2.config import get_cfg

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file(model_yaml_path))
cfg.DATASETS.TRAIN = ("mydata_train",)
cfg.DATASETS.TEST = ("mydata_val",)

cfg.DATALOADER.NUM_WORKERS = 1
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model_yaml_path)  # Let training initialize from model zoo
cfg.SOLVER.IMS_PER_BATCH = 1
cfg.SOLVER.BASE_LR = 0.001

cfg.MODEL.DEVICE='cpu'
cfg.SOLVER.WARMUP_ITERS = 2
cfg.SOLVER.MAX_ITER = 300 #adjust up if val mAP is still rising, adjust down if overfit
cfg.SOLVER.STEPS = (1, 2)
cfg.SOLVER.GAMMA = 0.05

cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 56
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3
cfg.MODEL.RETINANET.NUM_CLASSES = 3

cfg.TEST.EVAL_PERIOD = 2


os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = CocoTrainer(cfg)
trainer.resume_or_load(resume=False)
t = trainer.train()
print(t)
