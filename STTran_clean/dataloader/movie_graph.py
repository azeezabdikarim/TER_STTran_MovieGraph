import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import Resize, Compose, ToTensor, Normalize
import random
# from scipy.misc import imread
from imageio import imread
import numpy as np
import pickle
import os
import re
from fasterRCNN.lib.model.utils.blob import prep_im_for_blob, im_list_to_blob

class MG(Dataset):

    def __init__(self, mode, datasize, data_path=None, filter_nonperson_box_frame=True, filter_small_box=False, only_valid = True):

        root_path = data_path
        self.frames_path = os.path.join(root_path, 'frames/')

        # collect the object classes
        self.object_classes = ['__background__']
        with open(os.path.join(root_path, 'annotations/object_classes.txt'), 'r') as f:
            for line in f.readlines():
                line = line.strip('\n')
                self.object_classes.append(line)
        f.close()
        self.object_classes[9] = 'closet/cabinet'
        self.object_classes[11] = 'cup/glass/bottle'
        self.object_classes[23] = 'paper/notebook'
        self.object_classes[24] = 'phone/camera'
        self.object_classes[31] = 'sofa/couch'

        # collect relationship classes
        self.relationship_classes = []
        with open(os.path.join(root_path, 'annotations/relationship_classes.txt'), 'r') as f:
            for line in f.readlines():
                line = line.strip('\n')
                self.relationship_classes.append(line)
        f.close()
        self.relationship_classes[0] = 'looking_at'
        self.relationship_classes[1] = 'not_looking_at'
        self.relationship_classes[5] = 'in_front_of'
        self.relationship_classes[7] = 'on_the_side_of'
        self.relationship_classes[10] = 'covered_by'
        self.relationship_classes[11] = 'drinking_from'
        self.relationship_classes[13] = 'have_it_on_the_back'
        self.relationship_classes[15] = 'leaning_on'
        self.relationship_classes[16] = 'lying_on'
        self.relationship_classes[17] = 'not_contacting'
        self.relationship_classes[18] = 'other_relationship'
        self.relationship_classes[19] = 'sitting_on'
        self.relationship_classes[20] = 'standing_on'
        self.relationship_classes[25] = 'writing_on'

        self.attention_relationships = self.relationship_classes[0:3]
        self.spatial_relationships = self.relationship_classes[3:9]
        self.contacting_relationships = self.relationship_classes[9:]


        # collect valid frames
        self.video_dict = {}
        for video_name in os.listdir(root_path + "frames"):
            path_to_frames = os.path.join(root_path, "frames", video_name)
            if only_valid and 'invalid' not in path_to_frames:
                for frame_path in os.listdir(os.path.join(root_path, "frames", video_name)):

                    if video_name in self.video_dict.keys():
                        self.video_dict[video_name].append(frame_path)
                    else:
                        self.video_dict[video_name] = [frame_path]

        self.video_list = []
        self.video_size = [] # (w,h)

        '''
        filter_nonperson_box_frame = True (default): according to the stanford method, remove the frames without person box both for training and testing
        filter_nonperson_box_frame = False: still use the frames without person box, FasterRCNN may find the person
        '''
#         print(video_dict.keys())
        for i in self.video_dict.keys():
            video = []
            for j in self.video_dict[i]:
                video.append(os.path.join(i, j))

            if len(video) > 2:
                self.video_list.append(video)
            elif len(video) == 1:
                self.one_frame_video += 1


    def __getitem__(self, index):

        frame_names = self.video_list[index]
        processed_ims = []
        im_scales = []

        for idx, name in enumerate(frame_names):
            im = imread(os.path.join(self.frames_path, name)) # channel h,w,3
            im = im[:, :, ::-1] # rgb -> bgr
            im, im_scale = prep_im_for_blob(im, [[[102.9801, 115.9465, 122.7717]]], 600, 1000) #cfg.PIXEL_MEANS, target_size, cfg.TRAIN.MAX_SIZE
            im_scales.append(im_scale)
            processed_ims.append(im)

        blob = im_list_to_blob(processed_ims)
        im_info = np.array([[blob.shape[1], blob.shape[2], im_scales[0]]],dtype=np.float32)
        im_info = torch.from_numpy(im_info).repeat(blob.shape[0], 1)
        img_tensor = torch.from_numpy(blob)
        img_tensor = img_tensor.permute(0, 3, 1, 2)

        gt_boxes = torch.zeros([img_tensor.shape[0], 1, 5])
        num_boxes = torch.zeros([img_tensor.shape[0]], dtype=torch.int64)
        
        match = re.search(r'_(\d{3})_', frame_names[0])
        scene_id = -1
        if match:
            number = match.group(1)
            scene_id = int(number)       

        return img_tensor, im_info, gt_boxes, num_boxes, index, scene_id

    def __len__(self):
        return len(self.video_list)

def cuda_collate_fn(batch):
    """
    don't need to zip the tensor

    """
    return batch[0]
