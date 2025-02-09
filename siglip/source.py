import torch, os, PIL, numbers
from PIL import Image
import cv2

from transformers.modeling_utils import PreTrainedModel
from transformers.models.siglip.modeling_siglip import SiglipVisionModel
from transformers import AutoConfig, AutoModel, SiglipImageProcessor, SiglipVisionConfig, PretrainedConfig
from typing import Union
import torch.nn.functional as F
import numpy as np


def crop_clip(clip, min_h, min_w, h, w):
    if isinstance(clip[0], np.ndarray):
        cropped = [img[min_h:min_h + h, min_w:min_w + w, :] for img in clip]

    elif isinstance(clip[0], PIL.Image.Image):
        cropped = [
            img.crop((min_w, min_h, min_w + w, min_h + h)) for img in clip
        ]
    else:
        raise TypeError('Expected numpy.ndarray or PIL.Image' +
                        'but got list of {0}'.format(type(clip[0])))
    return cropped


class Normalize(object):
    """Normalize a clip with mean and standard deviation.
    Given mean: ``(M1,...,Mn)`` and std: ``(S1,..,Sn)`` for ``n`` channels, this transform
    will normalize each channel of the input ``torch.*Tensor`` i.e.
    ``input[channel] = (input[channel] - mean[channel]) / std[channel]``
    .. note::
        This transform acts out of place, i.e., it does not mutates the input tensor.
    Args:
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, clip):
        """
        Args:
            clip (Tensor): Tensor clip of size (T, C, H, W) to be normalized.
        Returns:
            Tensor: Normalized Tensor clip.
        """
        return normalize(clip, self.mean, self.std)

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


class CenterCrop(object):
    """Extract center crop at the same location for a list of images
    Args:
    size (sequence or int): Desired output size for the
    crop in format (h, w)
    """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            size = (size, size)

        self.size = size

    def __call__(self, clip):
        """
        Args:
        img (PIL.Image or numpy.ndarray): List of images to be cropped
        in format (h, w, c) in numpy.ndarray
        Returns:
        PIL.Image or numpy.ndarray: Cropped list of images
        """
        h, w = self.size
        if isinstance(clip[0], np.ndarray):
            im_h, im_w, im_c = clip[0].shape
        elif isinstance(clip[0], PIL.Image.Image):
            im_w, im_h = clip[0].size
        else:
            raise TypeError('Expected numpy.ndarray or PIL.Image' +
                            'but got list of {0}'.format(type(clip[0])))
        if w > im_w or h > im_h:
            error_msg = (
                'Initial image size should be larger then '
                'cropped size but got cropped sizes : ({w}, {h}) while '
                'initial image is ({im_w}, {im_h})'.format(
                    im_w=im_w, im_h=im_h, w=w, h=h))
            raise ValueError(error_msg)

        x1 = int(round((im_w - w) / 2.))
        y1 = int(round((im_h - h) / 2.))
        cropped = crop_clip(clip, y1, x1, h, w)

        return cropped


def resize_clip(clip, size, interpolation='bilinear'):
    if isinstance(clip[0], np.ndarray):
        if isinstance(size, numbers.Number):
            im_h, im_w, im_c = clip[0].shape
            # Min spatial dim already matches minimal size
            if (im_w <= im_h and im_w == size) or (im_h <= im_w
                                                   and im_h == size):
                return clip
            new_h, new_w = get_resize_sizes(im_h, im_w, size)
            size = (new_w, new_h)
        else:
            size = size[0], size[1]
        if interpolation == 'bilinear':
            np_inter = cv2.INTER_LINEAR
        else:
            np_inter = cv2.INTER_NEAREST
        scaled = [
            cv2.resize(img, size, interpolation=np_inter) for img in clip
        ]
    elif isinstance(clip[0], PIL.Image.Image):
        if isinstance(size, numbers.Number):
            im_w, im_h = clip[0].size
            # Min spatial dim already matches minimal size
            if (im_w <= im_h and im_w == size) or (im_h <= im_w
                                                   and im_h == size):
                return clip
            new_h, new_w = get_resize_sizes(im_h, im_w, size)
            size = (new_w, new_h)
        else:
            size = size[1], size[0]
        if interpolation == 'bilinear':
            pil_inter = PIL.Image.BILINEAR
        else:
            pil_inter = PIL.Image.NEAREST
        scaled = [img.resize(size, pil_inter) for img in clip]
    else:
        raise TypeError('Expected numpy.ndarray or PIL.Image' +
                        'but got list of {0}'.format(type(clip[0])))
    return scaled


def _is_tensor_clip(clip):
    return torch.is_tensor(clip) and clip.ndimension() == 4


def get_resize_sizes(im_h, im_w, size):
    if im_w < im_h:
        ow = size
        oh = int(size * im_h / im_w)
    else:
        oh = size
        ow = int(size * im_w / im_h)
    return oh, ow


def normalize(clip, mean, std, inplace=False):
    if not _is_tensor_clip(clip):
        raise TypeError('tensor is not a torch clip.')

    if not inplace:
        clip = clip.clone()

    dtype = clip.dtype
    mean = torch.as_tensor(mean, dtype=dtype, device=clip.device)
    std = torch.as_tensor(std, dtype=dtype, device=clip.device)
    clip.sub_(mean[:, None, None, None]).div_(std[:, None, None, None])

    return clip


class Resize(object):
    """Resizes a list of (H x W x C) numpy.ndarray to the final size
    The larger the original image is, the more times it takes to
    interpolate
    Args:
    interpolation (str): Can be one of 'nearest', 'bilinear'
    defaults to nearest
    size (tuple): (widht, height)
    """

    def __init__(self, size, interpolation='nearest'):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, clip):
        resized = resize_clip(
            clip, self.size, interpolation=self.interpolation)
        return resized


class Compose(object):
    """Composes several transforms
    Args:
    transforms (list of ``Transform`` objects): list of transforms
    to compose
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, clip):
        for t in self.transforms:
            clip = t(clip)
        return clip


def convert_img(img):
    """Converts (H, W, C) numpy.ndarray to (C, W, H) format"""
    if len(img.shape) == 3:
        img = img.transpose(2, 0, 1)
    if len(img.shape) == 2:
        img = np.expand_dims(img, 0)
    return img


class ClipToTensor(object):
    """Convert a list of m (H x W x C) numpy.ndarrays in the range [0, 255]
    to a torch.FloatTensor of shape (C x m x H x W) in the range [0, 1.0]
    """

    def __init__(self, channel_nb=3, div_255=True, numpy=False):
        self.channel_nb = channel_nb
        self.div_255 = div_255
        self.numpy = numpy

    def __call__(self, clip):
        """
        Args: clip (list of numpy.ndarray): clip (list of images)
        to be converted to tensor.
        """
        # Retrieve shape
        if isinstance(clip[0], np.ndarray):
            h, w, ch = clip[0].shape
            assert ch == self.channel_nb, "Got {0} instead of 3 channels".format(ch)
        elif isinstance(clip[0], Image.Image):
            w, h = clip[0].size
        else:
            raise TypeError(
                "Expected numpy.ndarray or PIL.Image\
            but got list of {0}".format(
                    type(clip[0])
                )
            )

        np_clip = np.zeros([self.channel_nb, len(clip), int(h), int(w)])

        # Convert
        for img_idx, img in enumerate(clip):
            if isinstance(img, np.ndarray):
                pass
            elif isinstance(img, Image.Image):
                img = np.array(img, copy=False)
            else:
                raise TypeError(
                    "Expected numpy.ndarray or PIL.Image\
                but got list of {0}".format(
                        type(clip[0])
                    )
                )
            img = convert_img(img)
            np_clip[:, img_idx, :, :] = img
        if self.numpy:
            if self.div_255:
                np_clip = np_clip / 255.0
            return np_clip

        else:
            tensor_clip = torch.from_numpy(np_clip)

            if not isinstance(tensor_clip, torch.FloatTensor):
                tensor_clip = tensor_clip.float()
            if self.div_255:
                tensor_clip = torch.div(tensor_clip, 255)
            return tensor_clip


class VisionTowerConfig(PretrainedConfig):
    model_type = "vision_tower"

    def __init__(self, vision_tower_name: str = None, **kwargs):
        super().__init__()
        self.vision_tower_name = vision_tower_name


class ProcessorWrapper:
    def __init__(self, transform=None, processor=None, height=378, width=378, frames_per_clip=1,
                 image_mean=[0.48145466, 0.4578275, 0.40821073]):
        assert transform is not None or processor is not None, "ERROR: you did not define both `transform` and `processor`! You must define either transform or processor"
        assert transform is None or processor is None, "ERROR: you did defined both `transform` and `processor`! You must define only one of: transform or processor"
        self._size = {
            "height": height,
            "width": width,
            "frames_per_clip": frames_per_clip
        }
        self._transforms = transform
        self._processor = processor
        self.image_mean = image_mean

    @property
    def size(self):
        return self._size

    def preprocess(self, image, return_tensors='pt'):
        # Ensure image is a PIL Image
        output = {}
        if self._transforms is not None:
            output['pixel_values'] = [self._transforms(image)]

        else:
            output = self._processor(image, return_tensors='pt')
        return output

    def save_pretrained(self, save_path):
        if self._transforms is not None:
            transform_dict = transform_to_dict(self._transforms)
            transform_dict["image_processor_type"] = "transforms"
            with open(os.path.join(save_path, 'preprocessor_config.json'), 'w') as f:
                json.dump(transform_dict, f, indent=4)
        else:
            self._processor.save_pretrained(save_path)
        return


class VisionTower(PreTrainedModel):
    config_class = VisionTowerConfig

    def __init__(self, model_name_or_path: str, config: PretrainedConfig, vision_config: VisionTowerConfig = None):
        super().__init__(vision_config)
        self.vision_tower_name = model_name_or_path
        self.vision_config = vision_config
        self.select_layer = getattr(config, "mm_vision_select_layer", -2)
        self.select_feature = getattr(config, "mm_vision_select_feature", "patch")
        self.encode_batch_size = getattr(config, "encode_batch_size", 0) // 2
        self.num_encode_batch = getattr(config, "num_encode_batch", 0) // 2
        self.temporal_tubelet_size = getattr(vision_config, "tubelet_size", 1)

    def feature_select(self, image_features):
        if self.select_layer is not None:
            image_features = image_features.hidden_states[self.select_layer]
            
        if self.select_feature == "patch":
            image_features = image_features[:, 1:]
        elif self.select_feature == "cls_patch":
            image_features = image_features
        else:
            raise ValueError(f"Unexpected select feature: {self.select_feature}")
            
        return image_features

    def vision_tower_forward(self, image):
        image_feature = self.vision_tower(image, output_hidden_states=True)
        return image_feature
    
    def _forward(self, images, out_T=1):
        if type(images) is list:
            image_features = []
            for image in images:
                image_feature = self.vision_tower_forward(image.to(device=self.device, dtype=self.dtype).unsqueeze(0))
                image_feature = self.feature_select(image_feature).to(image.dtype)
                image_feature = image_features.reshape(image_feature.shape[0], self.W, self.H, self.D)
                image_features.append(image_feature)
        else:
            original_shape = images.shape
            if len(original_shape) == 5 and self.T == 1:
                # downsample temporally if needed, and reshape from (B, T, C, W, H) to (B*T, C, W, H).
                images = images[:, ::original_shape[1] // out_T, ...]
                original_shape = images.shape
                images = images.view(-1, *original_shape[2:])

            image_features = self.vision_tower_forward(images.to(device=self.device, dtype=self.dtype))
            image_features = self.feature_select(image_features).to(images.dtype)
            # Reshape back to (B, T, ...) if necessary
            if len(original_shape) == 5 and self.T == 1:
                # Assuming the feature dimension does not change, adapt the following line if it does
                new_shape = list(image_features.shape[:-2]) + [self.W, self.H, self.hidden_size]
                image_features = image_features.reshape(new_shape)
                feature_size = image_features.shape[1:]
                image_features = image_features.view(original_shape[0], original_shape[1], *feature_size)
                
            else:
                image_features = image_features.reshape(image_features.shape[0], self.T, self.W, self.H, self.hidden_size)
                
        return image_features
    
    def forward(self, images):
        return self._forward(images)

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2


class InternVideoTower(VisionTower):
    def __init__(self, model_name_or_path: str, config: PretrainedConfig, vision_config: PretrainedConfig = None):
        if vision_config is None:
            vision_config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)

        super().__init__(model_name_or_path, config, vision_config)
        self.vision_config = vision_config
        normalize = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

        print('loading: ', model_name_or_path)
        model = AutoModel.from_pretrained(model_name_or_path, trust_remote_code=True)
        self.vision_tower = model.to(dtype=eval(config.model_dtype))

        transform = Compose([
            Resize(self.vision_config.img_size, interpolation='bilinear'),
            CenterCrop(size=(self.vision_config.img_size, self.vision_config.img_size)),
            ClipToTensor(),
            Normalize(mean=normalize[0], std=normalize[1])
        ])

        self.vision_processor = ProcessorWrapper(transform=transform,
                                                 height=self.vision_config.img_size,
                                                 width=self.vision_config.img_size,
                                                 frames_per_clip=self.vision_config.num_frames,
                                                 image_mean=normalize[0])

        self.W = self.H = vision_config.img_size // vision_config.patch_size
        self.T = self.vision_config.num_frames // self.vision_config.tubelet_size
        self.num_frames = self.vision_config.num_frames
        self.hidden_size = vision_config.d_model
        self.vision_select_layer=self.select_layer
        self.select_layer=None

    def vision_tower_forward(self, video):
        if video.shape[-3] < self.num_frames:
            video = video.repeat_interleave(self.num_frames, dim=-3)
        elif video.shape[-3] > self.num_frames:
            video = video[:, :, ::video.shape[-3] // self.num_frames, ...]

        video_feature = self.vision_tower(video.to(device=self.device, dtype=self.dtype),
                                          x_vis_return_idx=self.vision_select_layer, x_vis_only=True)
        
        return video_feature

    @property
    def device(self):
        return self.vision_tower.pos_embed.device


class SiglipVisionTower(VisionTower):
    def __init__(self, model_name_or_path: str, config: PretrainedConfig, vision_config: PretrainedConfig = None):
        if vision_config is None:
            vision_config = SiglipVisionConfig.from_pretrained(model_name_or_path)

        super().__init__(model_name_or_path, config, vision_config)
        self.vision_config = vision_config
        self.vision_tower_name = model_name_or_path
        self.vision_processor = SiglipImageProcessor.from_pretrained(self.vision_tower_name)

        print('loading: ', model_name_or_path)
        self.vision_tower = SiglipVisionModel.from_pretrained(self.vision_tower_name)

        self.hidden_size = self.vision_config.hidden_size
        self.W = self.H = self.vision_config.image_size // self.vision_config.patch_size
        self.T = 1
        self.select_feature = "cls_patch"


class ApolloVisionTower(PreTrainedModel):
    def __init__(self, config, vision_tower_cfg):
        super(ApolloVisionTower, self).__init__(config, vision_tower_cfg)
        self.model_name_or_path = vision_tower_cfg._name_or_path
        self.vision_towers = vision_tower_cfg.vision_towers
        self._config = vision_tower_cfg

        for vision_tower_name in self.vision_towers:
            if 'internvideo' in vision_tower_name.lower():
                vision_tower = InternVideoTower(os.path.join(vision_tower_cfg._name_or_path, vision_tower_name), config)
            elif 'siglip' in vision_tower_name.lower():
                vision_tower = SiglipVisionTower(os.path.join(vision_tower_cfg._name_or_path, vision_tower_name),
                                                 config)

            setattr(self, vision_tower_name, vision_tower)

        self.vision_processor = [getattr(self, vt).vision_processor for vt in self.vision_towers]
        self.num_vision_encoders = len(self.vision_towers)
        self.W = self.H = max([getattr(self, vt).W for vt in self.vision_towers])
        self.T = max([getattr(self, vt).T for vt in self.vision_towers])
        self.max_tubelet_size = max(
            [getattr(getattr(self, vt).vision_config, 'tubelet_size', 1) for vt in self.vision_towers])
        
        self._hidden_size = sum([getattr(self, vt).hidden_size for vt in self.vision_towers])
        self.token_output_shape = (self.T, self.W, self.H)
        self.config.num_vision_encoders = self.num_vision_encoders
        self.config.vision_towers = self.vision_towers
        self.config.token_output_shape = self.token_output_shape

    def forward(self, x):
        output_features = []
        for x_s, vision_tower_name in zip(x, self.vision_towers):
            vision_tower = getattr(self, vision_tower_name)
            features = vision_tower._forward(x_s, out_T=self.T)

            if len(features.shape) != len(self.token_output_shape) + 2:
                features = features.unsqueeze(1)

            if features.shape[-len(self.token_output_shape) - 1:-1] != self.token_output_shape:
                features = features.permute(0, 4, 1, 2, 3).contiguous()  # shape [B, D, T, W, H]
                features = F.interpolate(features.to(torch.float32), size=self.token_output_shape, mode='trilinear',
                                         align_corners=False).to(features.dtype)
                features = features.permute(0, 2, 3, 4, 1).contiguous()

            output_features.append(features)

        output_features = torch.cat(output_features, dim=-1)
        output_features = torch.flatten(output_features, start_dim=1, end_dim=-2)
        return output_features

    def save_pretrained(
            self,
            save_directory: Union[str, os.PathLike],
            state_dict=None,
            **kwargs,
    ):
        if state_dict is None:
            state_dict = self.state_dict()

        for vision_tower_name in self.vision_towers:
            vision_tower = getattr(self, vision_tower_name)
            vision_tower_state_dict = OrderedDict(
                {k.split(f"vision_tower.{vision_tower_name}.vision_tower.")[-1]: v for k, v in state_dict.items() if
                 vision_tower_name in k}
            )
            vision_tower.vision_tower.save_pretrained(os.path.join(save_directory, vision_tower_name),
                                                      state_dict=vision_tower_state_dict, **kwargs)
            vision_tower.vision_processor.save_pretrained(os.path.join(save_directory, vision_tower_name))

        config = self.config
        config.configs = {}
        config.save_pretrained(save_directory)

    @property
    def patch_size(self):
        return self._patch_size

    @property
    def image_size(self):
        return self._image_size

    @property
    def hidden_size(self):
        return self._hidden_size
