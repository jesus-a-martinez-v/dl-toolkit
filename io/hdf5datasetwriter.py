import os

import h5py


class HDF5DatasetWriter(object):
    def __init__(self, dimensions, output_path, data_key='images', buffer_size=1000):
        if os.path.exists(output_path):
            raise ValueError(
                'The supplied "output_path" already exists and cannot be overwritten. Manually delete the file before '
                'continuing.',
                output_path)

        self.db = h5py.File(output_path, 'w')
        self.data = self.db.create_dataset(data_key, dimensions, dtype='float')
        self.labels = self.db.create_dataset('labels', (dimensions[0],), dtype='int')

        self.buffer_size = buffer_size
        self.buffer = {'data': [], 'labels': []}
        self.index = 0

    def add(self, rows, labels):
        self.buffer['data'].extend(rows)
        self.buffer['labels'].extend(labels)

        if len(self.buffer['data']) >= self.buffer_size:
            self.flush()

    def flush(self):
        i = self.index + len(self.buffer['data'])
        self.data[self.index: i] = self.buffer['data']
        self.labels[self.index: i] = self.buffer['labels']
        self.index = i
        self.buffer = {'data': list(), 'labels': list()}

    def store_class_labels(self, class_labels):
        dt = h5py.special_dtype(vlen=str)
        label_set = self.db.create_dataset('label_names', (len(class_labels),), dtype=dt)
        label_set[:] = class_labels

    def close(self):
        if len(self.buffer['data']) > 0:
            self.flush()

        self.db.close()
