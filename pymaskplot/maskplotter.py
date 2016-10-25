import nibabel as nb
import scipy as sp
import numpy as np
from scipy import ndimage
import seaborn as sns
import matplotlib.pyplot as plt


def plot_mask_straight_edges(mask, **kwargs):
    
    l = None
    
    color = kwargs.get('c', 'black')
    
    xs, ys = np.where(np.diff(mask, axis=1))    
    
    for x, y in zip(xs, ys):
        plt.plot([x-.5, x+.5], [y+.5, y+.5], **kwargs)
        

    xs, ys = np.where(np.diff(mask, axis=0) != 0)    
    
    for x, y in zip(xs, ys):
        l = plt.plot([x+.5, x+.5], [y-.5, y+.5], **kwargs)        
        
    if l:        
        return l[0]

class MaskPlotter(object):
    
    orientations = ['sagittal', 'coronal', 'axial']
    crop_margin = 15
    
    def __init__(self, data, masks, mask_labels=None, palette=None, threshold=0.0, check_affines=True):
        
        self.masks = [nb.load(mask) for mask in masks]
        self.masks_ = [mask.get_data() > threshold for mask in self.masks]
        
        
        self.data = nb.load(data)
        self.data_ = self.data.get_data()

        if check_affines:
            assert(np.all([(mask.affine == self.masks[0].affine).all() for mask in self.masks]))
            assert((self.data.affine == self.masks[0].affine).all())
        else:
            print '***WARNING*** affines are not checked'
        
        if not mask_labels:
            self.mask_labels = ['mask %d' % i for i in range(0+1, len(self.masks)+1)]
        else:
            self.mask_labels = mask_labels
            
            
        if palette == None:
            self.palette = sns.color_palette('husl', len(masks))
        else:
            self.palette = palette


            
        self._xlim = None
        self._ylim = None
        self._zlim = None
        
        
    def plot_masks(self, orientation='coronal', s=None, center_on_mask=None, crop=True, imshow_kwargs={}, plotting_procedure='contour', **kwargs):
       

 
        
        if orientation not in self.orientations + ['all']:
            raise NotImplementedError('{orientation} orientation not implemented'.format(orientation))        

        if len(self.masks) > 0 and not center_on_mask:
            center_on_mask = self.mask_labels[0]

        if orientation == 'all':
            plt.subplot(131)
            self.plot_masks('coronal', center_on_mask=center_on_mask, crop=crop, imshow_kwargs=imshow_kwargs, **kwargs)
            plt.subplot(132)
            self.plot_masks('sagittal', center_on_mask=center_on_mask, crop=crop, imshow_kwargs=imshow_kwargs, **kwargs)
            plt.subplot(133)
            l = self.plot_masks('axial', center_on_mask=center_on_mask, crop=crop, imshow_kwargs=imshow_kwargs, **kwargs)
            return l

        
        if not s:
            if len(self.masks) > 0:
                s = ndimage.center_of_mass(self._get_mask(center_on_mask))[self.orientations.index(orientation)]
            else:
                raise Exception('No mask to get center-of-mass, please provide slice number')
            
            
        idx = [slice(None), slice(None), slice(None)]
        idx[self.orientations.index(orientation)] = s
        
        cmap = imshow_kwargs.pop('cmap', plt.cm.gray) 
        plt.imshow(self.data_[idx].T, origin='lower', interpolation='nearest', cmap=cmap, **imshow_kwargs)
        
        if plotting_procedure == 'contour':
            for i, mask in enumerate(self.mask_labels):
                l = plt.contour(self._get_mask(mask)[idx].T, levels=[0, 1], colors=[self.palette[i]], **kwargs) 
        elif plotting_procedure == 'straight_edges':
            for i, mask in enumerate(self.mask_labels):
                l = plot_mask_straight_edges(self._get_mask(mask)[idx], c=self.palette[i], **kwargs) 
            
        

        plt.axis('off')

        if crop:
            if orientation == 'coronal':
                plt.xlim(self.xlim[0] - self.crop_margin, self.xlim[1] + self.crop_margin)
                plt.ylim(self.zlim[0] - self.crop_margin, self.zlim[1] + self.crop_margin)
            if orientation == 'sagittal':
                plt.xlim(self.ylim[0] - self.crop_margin, self.ylim[1] + self.crop_margin)
                plt.ylim(self.zlim[0] - self.crop_margin, self.zlim[1] + self.crop_margin)
            if orientation == 'axial':
                plt.xlim(self.xlim[0] - self.crop_margin, self.xlim[1] + self.crop_margin)
                plt.ylim(self.ylim[0] - self.crop_margin, self.ylim[1] + self.crop_margin)

            

        plt.title('Slice %d (%s)' % (s, orientation))
        return l
            
            
    def _get_mask(self, mask_label):
        
        return self.masks_[self.mask_labels.index(mask_label)]

    def get_limits(self):

        _, xs, ys, zs = np.where(self.masks_)

        self._xlim = np.min(xs), np.max(xs)
        self._ylim = np.min(ys), np.max(ys)
        self._zlim = np.min(zs), np.max(zs)


    @property
    def xlim(self):

        if self._xlim is None:
            self.get_limits()

        return self._xlim

    @property
    def ylim(self):

        if self._ylim is None:
            self.get_limits()

        return self._ylim

    @property
    def zlim(self):

        if self._zlim is None:
            self.get_limits()

        return self._zlim
        
        
