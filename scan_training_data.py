from os import listdir
import SimpleITK
import os.path
import pickle
import numpy

def process_scan(scan):

    img =  numpy.transpose(
        SimpleITK.GetArrayFromImage(scan).astype(dtype=numpy.float),
        [2, 1, 0]
    )

    print(img.shape)

    return img




def iterate_folder(folder):
    for filename in sorted(listdir(folder)):
        if filename.endswith('.mhd') and not filename.endswith('_segmentation.mhd'):
            absolute_filename = os.path.join(folder, filename)
            segmentation_absolute_filename = absolute_filename[:-4] + '_segmentation.mhd'
            if os.path.exists(segmentation_absolute_filename):
                print(filename)
                yield absolute_filename, segmentation_absolute_filename


def load_data(folder):
    input_filenames, label_filenames = zip(*list(iterate_folder(folder)))
    
    X = numpy.array([process_scan(SimpleITK.ReadImage(f)) for f in input_filenames])
    y = numpy.array([process_scan(SimpleITK.ReadImage(f)) for f in label_filenames])
    
    return X, y > 0.5

X, y = load_data('/export/shared/uiuc/promise2012/training_data/')
X.shape, y.shape, y.mean() 

with open('/export/shared/uiuc/promise2012/training_data/train_data_raw.p3', 'wb') as f:
    pickle.dump([X, y], f)

