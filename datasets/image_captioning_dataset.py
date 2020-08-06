import csv
import os
from os import path
import torch
from torch.utils.data import Dataset
from PIL import Image, ImagePalette, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.warnings.simplefilter('ignore')
from .utils import colormap
import datasets.transforms as tf

''' for image captioning datasets where input data are pairs of images and labels
'''
class CaptioningDataset(Dataset):
    CLASSES = [
        'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 
        'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
        'motorbike', 'person', 'potted-plant', 'sheep', 'sofa', 'train',
        'tv/monitor', 'ambiguous']

    CLASS_IDX = {
            'background': 0, 
            'aeroplane': 1,
            'bicycle': 2,
            'bird': 3,
            'boat': 4,
            'bottle': 5,
            'bus': 6,
            'car': 7,
            'cat': 8,
            'chair': 9,
            'cow': 10,
            'diningtable': 11,
            'dog': 12,
            'horse': 13,
            'motorbike': 14,
            'person': 15,
            'potted-plant': 16,
            'sheep': 17,
            'sofa': 18,
            'train': 19,
            'tv/monitor': 20,
            'ambiguous': 255}

    CLASS_IDX_INV = {
            0: 'background', 
            1: 'aeroplane',
            2: 'bicycle',
            3: 'bird',
            4: 'boat',
            5: 'bottle',
            6: 'bus',
            7: 'car',
            8: 'cat',
            9: 'chair',
           10: 'cow',
           11: 'diningtable',
           12: 'dog',
           13: 'horse',
           14: 'motorbike',
           15: 'person',
           16: 'potted-plant',
           17: 'sheep',
           18: 'sofa',
           19: 'train',
           20: 'tv/monitor',
          255: 'ambiguous'}

    NUM_CLASS = 21

    MEAN = (0.485, 0.456, 0.406)
    STD = (0.229, 0.224, 0.225)

    def __init__(self):
        super().__init__()
        self._init_palette()

    def _init_palette(self):
        self.cmap = colormap()
        self.palette = ImagePalette.ImagePalette()
        for rgb in self.cmap:
            self.palette.getcolor(rgb)

    def get_palette(self):
        return self.palette

    def denorm(self, image):

        if image.dim() == 3:
            assert image.dim() == 3, "Expected image [CxHxW]"
            assert image.size(0) == 3, "Expected RGB image [3xHxW]"

            for t, m, s in zip(image, self.MEAN, self.STD):
                t.mul_(s).add_(m)
        elif image.dim() == 4:
            # batch mode
            assert image.size(1) == 3, "Expected RGB image [3xHxW]"

            for t, m, s in zip((0,1,2), self.MEAN, self.STD):
                image[:, t, :, :].mul_(s).add_(m)

        return image


class CaptioningSegmentation(CaptioningDataset):
    def __init__(self, cfg, split, test_mode, root):
        super().__init__()
        self.cfg = cfg
        if root is None:
            root = os.path.expanduser('./data')
        self.root = root
        self.split = split
        self.test_mode = test_mode

        # train/val/test splits are pre-cut
        if self.split == 'train':
            self._split_f = os.path.join(self.root, f'train_{self.dataset_name}.tsv')
        elif self.split == 'train_val':
            self._split_f = os.path.join(self.root, f'trainval_{self.dataset_name}.tsv')
        elif self.split == 'val':
            self._split_f = os.path.join(self.root, f'val_{self.dataset_name}.tsv')
        else:
            raise RuntimeError('Unknown dataset split.')

        assert os.path.isfile(self._split_f), "%s not found" % self._split_f
        
        self.images = []
        self.labels = []
        with open(self._split_f, "r") as f:
            reader = csv.reader(f, delimiter='\t')
            for _img, *labels in reader:
                img_filepath = path.join(self.root, 'images', _img)
                assert os.path.isfile(img_filepath), '%s not found' % img_filepath
                self.images.append(img_filepath)
                self.labels.append([self.CLASS_IDX[l] for l in labels])


        self.transform = tf.Compose([tf.RandResizedCrop(self.cfg.DATASET), \
                                     tf.HFlip(), \
                                     tf.ColourJitter(p = 1.0), \
                                     tf.Normalise(self.MEAN, self.STD)])
        
    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = Image.open(self.images[index]).convert('RGB')
        unique_labels  = torch.tensor(self.labels[index], dtype=torch.long)

        # # ambigious
        # if unique_labels[-1] == self.CLASS_IDX['ambiguous']:
        #     unique_labels = unique_labels[:-1]

        # # background
        # if unique_labels[0] == self.CLASS_IDX['background']:
        #     unique_labels = unique_labels[1:]
        
        unique_labels -= 1 # shifting since no BG class

        assert len(unique_labels) > 0, 'No labels found in %s' % self.masks[index]
        labels = torch.zeros(self.NUM_CLASS - 1)
        labels[unique_labels] = 1

        # general resize, normalize and toTensor
        image = self.transform(image)
        return image, labels, os.path.basename(self.images[index])

    @property
    def pred_offset(self):
        return 0


class COCOSegmentation(CaptioningSegmentation):
    def __init__(self, cfg, split, test_mode, root=None):
        self.cfg = cfg
        if root is None:
            root = os.path.expanduser('./data')
        self.root = root
        self.split = split
        self.test_mode = test_mode

        # train/val/test splits are pre-cut
        if self.split == 'train':
            self._split_f = os.path.join(self.root, 'train_coco.tsv')
        elif self.split == 'train_val':
            self._split_f = os.path.join(self.root, 'trainval_coco.tsv')
        elif self.split == 'val':
            self._split_f = os.path.join(self.root, 'val_coco.tsv')
        else:
            raise RuntimeError('Unknown dataset split.')

        assert os.path.isfile(self._split_f), "%s not found" % self._split_f
        super(COCOSegmentation, self).__init__()



class ConceptualCaptionsSegmentation(CaptioningSegmentation):
    def __init__(self, cfg, split, test_mode, root=None):
        print("here")
        self.cfg = cfg
        if root is None:
            root = os.path.expanduser('./data')
        self.root = root
        self.split = split
        self.test_mode = test_mode

        # train/val/test splits are pre-cut
        if self.split == 'train':
            self._split_f = os.path.join(self.root, 'train_concap.tsv')
        elif self.split == 'train_val':
            self._split_f = os.path.join(self.root, 'trainval_concap.tsv')
        elif self.split == 'val':
            self._split_f = os.path.join(self.root, 'val_concap.tsv')
        else:
            raise RuntimeError('Unknown dataset split.')

        assert os.path.isfile(self._split_f), "%s not found" % self._split_f
        super(ConceptualCaptionsSegmentation, self).__init__()

