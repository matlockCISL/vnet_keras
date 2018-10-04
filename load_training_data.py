from os import listdir
import SimpleITK
import os.path
import pickle
import numpy

volSize = numpy.array((128,128,64), numpy.int32)
dstRes  = numpy.array((1,1,1.5))
normDir = False
method  = SimpleITK.sitkLinear

def process_scan(scan):
    ret = numpy.zeros(volSize, dtype=numpy.float32)
    factor = numpy.asarray(scan.GetSpacing()) / dstRes

    factorSize = numpy.asarray(scan.GetSize() * factor, dtype=numpy.float)

    newSize = numpy.max([factorSize, volSize], axis=0)

    newSize = newSize.astype(dtype=numpy.int32)

    T=SimpleITK.AffineTransform(3)
    T.SetMatrix(scan.GetDirection())

    resampler = SimpleITK.ResampleImageFilter()
    resampler.SetReferenceImage(scan)
    resampler.SetOutputSpacing(dstRes)
    resampler.SetSize(newSize.tolist())
    resampler.SetInterpolator(method)
    if normDir:
        resampler.SetTransform(T.GetInverse())

    imgResampled = resampler.Execute(scan)


    imgCentroid = numpy.asarray(newSize, dtype=numpy.float) / 2.0

    imgStartPx = (imgCentroid - numpy.array(volSize) / 2.0).astype(dtype=int)

    regionExtractor = SimpleITK.RegionOfInterestImageFilter()
    regionExtractor.SetSize(volSize.astype(dtype=numpy.int32).tolist())
    regionExtractor.SetIndex(imgStartPx.tolist())

    imgResampledCropped = regionExtractor.Execute(imgResampled)

    return numpy.transpose(
        SimpleITK.GetArrayFromImage(imgResampledCropped).astype(dtype=numpy.float),
        [2, 1, 0]
    )


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

with open('/export/shared/uiuc/promise2012/training_data/train_data.p3', 'wb') as f:
    pickle.dump([X, y], f)

