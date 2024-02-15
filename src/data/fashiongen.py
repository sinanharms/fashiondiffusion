import h5py
from torch.utils.data import Dataset, Subset
from tqdm import tqdm


class FashionGenDataset(Dataset):
    def __init__(self, filename):
        self.data = None
        self.filename = filename

    def import_data(self):
        with h5py.File(self.filename, "r") as f:
            self.data = f

    def get_images(self):
        return self.data["images"]


def traverse_datasets(hdf_file):
    """Traverse all datasets across all groups in HDF5 file."""

    import h5py

    def h5py_dataset_iterator(g, prefix=""):
        for key in g.keys():
            item = g[key]
            path = "{}/{}".format(prefix, key)
            if isinstance(item, h5py.Dataset):  # test for dataset
                yield (path, item)
            elif isinstance(item, h5py.Group):  # test for group (go down)
                yield from h5py_dataset_iterator(item, path)

    with h5py.File(hdf_file, "r") as f:
        for path, dset in h5py_dataset_iterator(f):
            print(path, dset)

    return None
