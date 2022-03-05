import h5py
import numpy as np

class Mask():
    def __init__(self):
        self.arr = None
        self.name = None
        self.crop_start = None
        self.crop_size = None
        self.size = None

    @classmethod
    def load_masks_from_file(cls, f):
        masks = {}
        with h5py.File(f, 'r') as fd:
            for g, d in fd.items():
                mask = cls()
                try:
                    name = d.attrs['name'].decode('utf-8')
                except:
                    name = str(d.attrs['name'])
                mask.name = name
                mask.arr = d['mask'][()]
                props = d['ArrayProps'].attrs
                mask.crop_start = props['crop_start'][()]
                mask.crop_size  = props['crop_size'][()]
                mask.size       = props['size'][()]
                masks[name] = mask
        return masks

    @classmethod
    def save_masks_to_file(cls, f, masks):
        if isinstance(masks, dict):
            masks = list(masks.values())

        with h5py.File(f, 'w') as fd:
            for ii, mask in enumerate(masks):
                maskgroup = fd.create_group('/'+str(mask.name))
                maskgroup.attrs.create('name', str(mask.name))
                maskgroup.attrs.create('index', ii, dtype=np.uint16)
                props = maskgroup.create_group('ArrayProps')
                props.attrs.create('crop_size', mask.crop_size, dtype=np.uint16)
                props.attrs.create('crop_start', mask.crop_start, dtype=np.uint16)
                props.attrs.create('size', mask.size, dtype=np.uint16)
                maskgroup.create_dataset('mask', data=mask.arr, dtype=np.uint8)

    def pad_to_original_size(self):
        arr = np.zeros(self.size[::-1])
        subslice = tuple([slice(self.crop_start[ii], self.crop_start[ii]+self.crop_size[ii]) for ii in range(3)][::-1])
        arr[subslice] = self.arr
        return arr
