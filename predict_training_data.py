import sys
sys.path.append('/export/software/tensorflow-1.3.0-rc2/python_modules/')

from keras.models import load_model
import keras.backend as K

from os import listdir
import SimpleITK
import os.path
import pickle
import numpy

import ntpath

volSize = numpy.array((128,128,64), numpy.int32)
dstRes  = numpy.array((1,1,1.5))
normDir = False
method  = SimpleITK.sitkLinear

def generate_segmentation(f, model, output_path):
    print(f)
    scan = SimpleITK.ReadImage(f)
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

    X = SimpleITK.GetArrayFromImage(imgResampledCropped).astype(dtype=numpy.float)
    print(X.shape)

    X =  numpy.array( [numpy.transpose(X, [2, 1, 0])] )

    # predict
    # y_pred = model.predict(X)
    y_pred = X[0, :, :, :] > 0.5

    # reverse crop
    pred = numpy.zeros(imgResampled.GetSize(), dtype=numpy.float)

    # convert back
    # print(y_pred.shape)
    pred[ imgStartPx[0]:(imgStartPx[0]+y_pred.shape[0]), imgStartPx[1]:(imgStartPx[1]+y_pred.shape[1]), imgStartPx[2]:(imgStartPx[2]+y_pred.shape[2]) ] = y_pred
    pred = numpy.transpose(pred, [2, 1, 0])

    # print(pred.shape)
    # print(imgResampled.GetSize())

    mask = SimpleITK.GetImageFromArray(pred)
    mask.SetOrigin( scan.GetOrigin() )
    mask.SetDirection( scan.GetDirection() )
    mask.SetSpacing( scan.GetSpacing() )

    # print(mask.GetSize())

    T=SimpleITK.AffineTransform(3)
    T.SetMatrix(mask.GetDirection())

    resampler = SimpleITK.ResampleImageFilter()
    resampler.SetReferenceImage(mask)
    resampler.SetOutputSpacing(scan.GetSpacing())
    resampler.SetSize(scan.GetSize())
    resampler.SetInterpolator(method)
    if normDir:
        resampler.SetTransform(T.GetInverse())

    imgReverseResampled = resampler.Execute(mask)

    print(scan.GetSize())
    print(imgReverseResampled.GetSize())

    if not output_path.endswith('/'):
        output_path = output_path + '/'

    filename = ntpath.basename(f)
    SimpleITK.WriteImage(SimpleITK.Cast(imgReverseResampled, SimpleITK.sitkUInt8), output_path+filename[:-4]+'_segmentation.mhd')

    # print('-----------')


def iterate_folder(folder):
    for filename in sorted(listdir(folder)):
        if filename.endswith('.mhd') and not filename.endswith('_segmentation.mhd'):
            absolute_filename = os.path.join(folder, filename)
            yield absolute_filename

def dice_coef(y_true, y_pred, smooth = 1e-5):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    union = K.sum(y_true * y_true) + K.sum(y_pred * y_pred)
    return (2. * intersection + smooth) / (union + smooth)

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


def process_data(folder):
    input_filenames = list(iterate_folder(folder))

    model = load_model('vnet_impl_3.h5', custom_objects={'dice_coef_loss': dice_coef_loss, 'dice_coef': dice_coef})
    
    for f in input_filenames:
        generate_segmentation(f, model, '/export/shared/uiuc/promise2012/testing_results/')
    

process_data('/export/shared/uiuc/promise2012/testing_data/')




