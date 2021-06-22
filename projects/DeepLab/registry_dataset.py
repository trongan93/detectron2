import os, json, cv2, random, orjson
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import load_coco_json, load_sem_seg


def registry_dataset_semantic_segmentation():
#     ##### Real data
#     name_train = "custom_dataset_train"
#     # 1. register a function which returns dicts
#     image_root = '/mnt/d/RarePlanes/datasets/real/train/PS-RGB_tiled'
#     sem_seg_root = '/mnt/d/RarePlanes/datasets/real/train/trongan_PS-RGB_tiled_mask'
#     DatasetCatalog.register(name_train, lambda: load_sem_seg(sem_seg_root, image_root , gt_ext='png', image_ext='png'))
#     # 2. Optionally, add metadata about this dataset,
#     # since they might be useful in evaluation, visualization or logging
# #     thing_classes=["aircraft"],
#     MetadataCatalog.get(name_train).set(stuff_classes=["aircraft"], ignore_label="0", image_root = image_root, sem_seg_root = sem_seg_root)
    
#     name_test = "custom_dataset_test"
#     # 1. register a function which returns dicts
#     DatasetCatalog.register(name_test, lambda: load_sem_seg('/mnt/d/RarePlanes/datasets/real/test/trongan_PS-RGB_tiled_mask', '/mnt/d/RarePlanes/datasets/real/test/PS-RGB_tiled', gt_ext='png', image_ext='png'))
#     # 2. Optionally, add metadata about this dataset,
#     # since they might be useful in evaluation, visualization or logging
#     MetadataCatalog.get(name_test).set(evaluator_type='sem_seg', stuff_classes=["aircraft"], ignore_label="0", image_root = image_root, sem_seg_root = sem_seg_root)
    
    ##### Synthetic data
    name_train = "custom_dataset_train"
    # 1. register a function which returns dicts
    image_root = '/mnt/d/RarePlanes/datasets/synthetic/train/images'
    sem_seg_root = '/mnt/d/RarePlanes/datasets/synthetic/train/masks'
    DatasetCatalog.register(name_train, lambda: load_sem_seg(sem_seg_root, image_root , gt_ext='png', image_ext='png'))
    # 2. Optionally, add metadata about this dataset,
    # since they might be useful in evaluation, visualization or logging
#     thing_classes=["aircraft"],
    MetadataCatalog.get(name_train).set(stuff_classes=["aircraft"], ignore_label="0", image_root = image_root, sem_seg_root = sem_seg_root)
    
    name_test = "custom_dataset_test"
    # 1. register a function which returns dicts
    DatasetCatalog.register(name_test, lambda: load_sem_seg('/mnt/d/RarePlanes/datasets/synthetic/test/masks', '/mnt/d/RarePlanes/datasets/synthetic/test/images', gt_ext='png', image_ext='png'))
    # 2. Optionally, add metadata about this dataset,
    # since they might be useful in evaluation, visualization or logging
    MetadataCatalog.get(name_test).set(evaluator_type='sem_seg', stuff_classes=["aircraft"], ignore_label="0", image_root = image_root, sem_seg_root = sem_seg_root)
    
    