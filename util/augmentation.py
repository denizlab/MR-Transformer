import torch
import numpy as np
import cv2
import h5py
import random
from scipy import ndimage as nd
from scipy.ndimage import rotate



def read_t2mapping_hdf5(file_name,  max_num_slices=36):
    data = h5py.File(file_name, 'r')
    h, w, num_slices, num_tes = data['data'].shape
    
    image_array = np.zeros((1, max_num_slices, h, w), dtype=np.float32)

    # if not enough number of slices, then load all
    load_slice_idx_begin = 0
    load_slice_idx_end = num_slices

    # limit num_slices
    if num_slices > max_num_slices:
        diff = num_slices - max_num_slices 
        load_slice_idx_begin = diff // 2
        load_slice_idx_end = load_slice_idx_begin + max_num_slices
        num_slices = max_num_slices

    # calculate slice-indices to load the image to
    zstart = int(np.ceil((max_num_slices-num_slices)/2))

    image_array[0,zstart:zstart+num_slices,:,:] = np.array(data['data'][:,:,load_slice_idx_begin:load_slice_idx_end,0]).transpose(2,0,1)

    data.close()

    return image_array


def read_iw_tse_hdf5(file_name,  max_num_slices=36):
    data = h5py.File(file_name, 'r')
    h, w, num_slices = data['data'].shape

    image_array = np.zeros((1, max_num_slices, h, w), dtype=np.float32)
    
    # if not enough number of slices, then load all
    load_slice_idx_begin = 0
    load_slice_idx_end = num_slices

    # limit num_slices
    if num_slices > max_num_slices:
        diff = num_slices - max_num_slices
        load_slice_idx_begin = diff // 2
        load_slice_idx_end = load_slice_idx_begin + max_num_slices
        num_slices = max_num_slices

    # calculate slice-indices to load the image to
    zstart = int(np.ceil((max_num_slices-num_slices)/2))

    image_array[0,zstart:zstart+num_slices,:,:] = np.array(data['data'][:,:,load_slice_idx_begin:load_slice_idx_end]).transpose(2,0,1)

    data.close()

    return image_array


def read_MRNet_hdf5(file_name,  max_num_slices=36):
    data = h5py.File(file_name, 'r')
    num_slices, h, w = data['image'].shape

    image_array = np.zeros((1, max_num_slices, h, w))

    # if not enough number of slices, then load all
    load_slice_idx_begin = 0
    load_slice_idx_end = num_slices
    # limit num_slices
    if num_slices > max_num_slices:
        diff = num_slices - max_num_slices
        load_slice_idx_begin = diff // 2
        load_slice_idx_end = load_slice_idx_begin + max_num_slices
        num_slices = max_num_slices

    # calculate slice-indices to load the image to
    zstart = int(np.ceil((max_num_slices-num_slices)/2))
    image_array[0,zstart:zstart+num_slices,:,:] = np.array(data['image'][()][load_slice_idx_begin:load_slice_idx_end,:,:])

    data.close()
    return image_array



class RandomRescale2D(object):

    '''
    Randomly Rescale 3D image in two dimensions
    '''

    def __init__(self,scale=(0.9,1.3)):
        self.scale=scale
        
    def __call__(self, image):
        scale=np.random.uniform(low=self.scale[0],high=self.scale[1])
        scaled_image = nd.zoom(image, zoom=[1,1,scale,scale], order=1) 
        return scaled_image

    def __repr__(self):
        return self.__class__.__name__ + '()'

    
class RandomRescale3D(object):

    '''
    Randomly Rescale 3D image in three dimensions
    '''

    def __init__(self,scale=(0.9,1.3)):
        self.scale=scale
        
    def __call__(self, image):
        scale=np.random.uniform(low=self.scale[0],high=self.scale[1])
        scaled_image = nd.zoom(image, zoom=[1,scale,scale,scale], order=1)
        return scaled_image
    
    
    
def to_shape(a, shape):
    z_, y_, x_ = shape
    z, y, x = a.shape
    z_pad = (z_-z)
    y_pad = (y_-y)
    x_pad = (x_-x)
    return np.expand_dims(np.pad(a,
                  ((z_pad//2, z_pad//2 + z_pad%2),
                      (y_pad//2, y_pad//2 + y_pad%2),
                     (x_pad//2, x_pad//2 + x_pad%2)),
                  mode = 'constant'),0)



class RandomCrop(object):

    '''
    Randomly Crop 3D image
    '''

    def __init__(self, cropdim=(36,384,384)):
        self.cropdim=cropdim

    def __call__(self, image):
        """
        Args:
            tensor: Tensor to be repeated.

        Returns:
            Tensor: repeated tensor.
        """
        _, d, h, w = image.shape
        cropdim_d, cropdim_h, cropdim_w = self.cropdim
        
        
        # add padding to maintain for cropping
        if d < cropdim_d:
            image = to_shape(image[0],shape=(cropdim_d,h,w))
            _, d, h, w = image.shape
        if h < cropdim_h:
            image = to_shape(image[0],shape=(d,cropdim_h,w))
            _, d, h, w = image.shape
        if w < cropdim_w:
            image = to_shape(image[0],shape=(d,h,cropdim_w))
            _, d, h, w = image.shape
        
        
        crop_d = random.randint(0, d-self.cropdim[0])
        crop_h = random.randint(0, h-self.cropdim[1])
        crop_w = random.randint(0, w-self.cropdim[2])
        cropped_image = image[
            :,
            crop_d:crop_d+self.cropdim[0],
            crop_h:crop_h+self.cropdim[1],
            crop_w:crop_w+self.cropdim[2]
        ]
        return cropped_image

    def __repr__(self):
        return self.__class__.__name__ + '()'

    
    
    
class CenterCrop(object):

    '''
    CenterCrop Images
    '''

    def __init__(self, cropdim=(36,384,384)):
        self.cropdim=cropdim

    def __call__(self, image):
        """
        Args:
            tensor: Tensor to be repeated.

        Returns:
            Tensor: repeated tensor.
        """
        _, d, h, w = image.shape
        cropdim_d, cropdim_h, cropdim_w = self.cropdim
        
        # add padding to maintain for cropping
        if d < cropdim_d:
            image = to_shape(image[0],shape=(cropdim_d,h,w))
            _, d, h, w = image.shape
        if h < cropdim_h:
            image = to_shape(image[0],shape=(d,cropdim_h,w))
            _, d, h, w = image.shape
        if w < cropdim_w:
            image = to_shape(image[0],shape=(d,h,cropdim_w))
            _, d, h, w = image.shape
        
        
        crop_d = int((d-self.cropdim[0])/2)
        crop_h = int((h-self.cropdim[1])/2)
        crop_w = int((w-self.cropdim[2])/2)
        return image[
            :,
            crop_d:crop_d+self.cropdim[0],
            crop_h:crop_h+self.cropdim[1],
            crop_w:crop_w+self.cropdim[2]
        ]

    def __repr__(self):
        return self.__class__.__name__ + '()'



    
class RandomFlip(object):
    '''
    Randomly flipped images
    Only horizontal flip is applied
    '''

    def __init__(self, p=0.5):
        self.p = p

    def horizontal_flip(self, image):
        '''
        :param p: probability of flip
        :return: randomly horizontaly flipped image
        '''

        integer = random.randint(0, 1)
        if integer <= self.p:
            output_image = np.flip(image, 3)
        else:
            output_image = image

        return output_image

    def vertical_flip(self, image):

        '''
        :param p: probability of flip
        :return: randomly vertically flipped image
        '''

        integer = random.randint(0, 1)
        if integer <= self.p:
            output_image = np.flip(image, 2)
        else:
            output_image = image

        return output_image

    def __call__(self, image):
        """
        Args:
            tensor: Tensor to be repeated.

        Returns:
            Tensor: repeated tensor.
        """
        # image = self.vertical_flip(image)
        image = self.horizontal_flip(image)
        if self.p > 0:
            #  if indices orders are changed, make it contiguous
            return np.ascontiguousarray(image)
        return image


    def __repr__(self):
        return self.__class__.__name__ + '()'
    
    
    
class SimpleRotate(object):

    def __init__(self, degrees=10, p=0.5):
        self.degrees = degrees
        self.p = p

    def __call__(self, array):
        """
        Args:
            array: numpy ndarray

        Returns:
            Tensor: repeated tensor.
        """
        integer = random.randint(0, 1)
        if integer <= self.p:
            output_image = rotate(array, random.randint(-self.degrees, self.degrees), axes=(3,2), reshape=False)
        else:
            output_image = array
        return output_image

    
    def __repr__(self):
        return self.__class__.__name__ + '()'
    
    
    
class Standardize(object):
    """
    Turn image into 0 mean unit variance.
    """

    def __init__(self):
        pass

    def __call__(self, array):
        """
        Args:
            array: numpy ndarray

        Returns:
            array: numpy ndarray
        """
        array -= np.mean(array)
        array /= np.maximum(np.std(array), 10**(-5))
        return array

    def __repr__(self):
        return self.__class__.__name__ + '()'

    
class RepeatChannels(object):

    def __init__(self, channels):
        self.channels = channels

    def __call__(self, tensor):
        """
        Args:
            tensor: Tensor to be repeated.

        Returns:
            Tensor: repeated tensor.
        """
        return tensor.repeat(self.channels, 1, 1, 1)


    def __repr__(self):
        return self.__class__.__name__ + '()'
    
    
class GaussianBlur(object):    
    """
    Implements Gaussian blur as described in the SimCLR paper
    """
    
    def __init__(self, kernel_size=35, min=0.1, max=2.0):
        self.min = min
        self.max = max
        # kernel size is set to be 10% of the image height/width
        self.kernel_size = kernel_size

    def __call__(self, sample):
        # blur the image with a 50% chance
        prob = np.random.random_sample()

        if prob < 0.5:
            sigma = (self.max - self.min) * np.random.random_sample() + self.min
            sample = cv2.GaussianBlur(sample[0,:,:,:].transpose(1,2,0), (self.kernel_size, self.kernel_size), sigma)
            sample = np.expand_dims(sample.transpose(2,0,1), 0)

        return sample
    
    
    
def numpy_to_torch(obj):
    """
    Convert to tensors all Numpy arrays inside a Python object composed of the
    supported types.
    Args:
        obj: The Python object to convert.
    Returns:
        A new Python object with the same structure as `obj` but where the
        Numpy arrays are now tensors. Not supported type are left as reference
        in the new object.
    Example:
        .. code-block:: python
            >>> from poutyne import numpy_to_torch
            >>> numpy_to_torch({
            ...     'first': np.array([1, 2, 3]),
            ...     'second':[np.array([4,5,6]), np.array([7,8,9])],
            ...     'third': 34
            ... })
            {
                'first': tensor([1, 2, 3]),
                'second': [tensor([4, 5, 6]), tensor([7, 8, 9])],
                'third': 34
            }
    """
    fn = lambda a: torch.from_numpy(a) if isinstance(a, np.ndarray) else a
    return _apply(obj, fn)

def _apply(obj, func):
    if isinstance(obj, (list, tuple)):
        return type(obj)(_apply(el, func) for el in obj)
    if isinstance(obj, dict):
        return {k: _apply(el, func) for k, el in obj.items()}
    return func(obj)




class ToTensor(object):

    def __init__(self):
        pass

    def __call__(self, array):
        """
        Args:
            array: numpy ndarray

        Returns:
            Tensor: repeated tensor.
        """
        return numpy_to_torch(array)


    def __repr__(self):
        return self.__class__.__name__ + '()'
    