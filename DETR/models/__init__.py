# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from .detr import build

#build 함수는 detr.py 
def build_model(args):
    return build(args)
