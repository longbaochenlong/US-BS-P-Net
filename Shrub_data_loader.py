from osgeo import gdal
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np
from sklearn.decomposition import PCA


class ShrubCLSData():
    img_filename = './data/hyperspectral_image.dat'
    gr_filename = './data/ground_truth.tif'
    palette_filename = './palette/shrub_palette.csv'

    def __init__(self, width, band_idx=None, is_std=True, is_pca=False):
        if band_idx is None:
            band_idx = []
        self.raster_data = gdal.Open(ShrubCLSData.img_filename)
        self.image = self.scale(band_idx, is_std)
        if is_pca:
            self.image = self.applyPCA(numComponents=80)
        self.ground_truth = gdal.Open(ShrubCLSData.gr_filename)
        self.class_df = pd.read_csv(ShrubCLSData.palette_filename)
        margin = int((width - 1) / 2)
        self.padded_data = self.padWithZeros(margin)
        self.colorList = self.get_colors()

    def scale(self, band_idx=None, is_std=True):
        if band_idx is None:
            band_idx = []
        scaler = StandardScaler()
        raster_array = self.raster_data.ReadAsArray()
        raster_array = raster_array.transpose((1, 2, 0))
        if len(band_idx) > 0:
            raster_array = raster_array[:, :, band_idx]
        h, w, b = raster_array.shape
        if is_std:
            raster_array = scaler.fit_transform(raster_array.reshape((h * w, b)))
        return raster_array.reshape(h, w, b)

    def applyPCA(self, numComponents=5):
        newX = np.reshape(self.image, (-1, self.image.shape[2]))
        pca = PCA(n_components=numComponents, whiten=True)
        newX = pca.fit_transform(newX)
        newX = np.reshape(newX, (self.image.shape[0], self.image.shape[1], numComponents))
        return newX

    def get_gt(self, pixel=None):
        gt_data = self.ground_truth.ReadAsArray()
        if pixel:
            return gt_data[pixel[0], pixel[1]]
        else:
            return gt_data.reshape(gt_data.shape[0] * gt_data.shape[1])

    def padWithZeros(self, margin=2):
        newX = np.zeros((self.image.shape[0] + 2 * margin, self.image.shape[1] + 2 * margin, self.image.shape[2]))
        x_offset = margin
        y_offset = margin
        newX[x_offset:self.image.shape[0] + x_offset, y_offset:self.image.shape[1] + y_offset, :] = np.copy(self.image)
        return newX

    def getPatch(self, index, windowSize=5):
        row = int(index / self.image.shape[1])
        col = int(index - row * self.image.shape[1])
        return self.padded_data[row:row + windowSize, col:col + windowSize, :]

    def getPatches(self, indexes, windowSize=5):
        patchData = np.zeros([len(indexes), windowSize, windowSize, self.image.shape[2]], dtype=np.float32)
        for i, index in enumerate(indexes):
            row = int(index / self.image.shape[1])
            col = int(index - row * self.image.shape[1])
            patchData[i] = self.padded_data[row:row + windowSize, col:col + windowSize, :]
        return patchData

    def getPatches_bs(self, indexes, band_idx, windowSize=5):
        patchData = np.zeros([len(indexes), windowSize, windowSize, len(band_idx)], dtype=np.float32)
        for i, index in enumerate(indexes):
            row = int(index / self.image.shape[1])
            col = int(index - row * self.image.shape[1])
            patchData[i] = self.padded_data[row:row + windowSize, col:col + windowSize, band_idx]
        return np.squeeze(patchData)

    def getPatches_new(self, indexes, windowSize=5):
        width = self.raster_data.RasterXSize
        height = self.raster_data.RasterYSize
        band_num = self.raster_data.RasterCount
        patchData = np.zeros([len(indexes), windowSize, windowSize, band_num], dtype=np.float32)
        for i, index in enumerate(indexes):
            row = int(index / width)
            col = int(index - row * width)
            read_xoff = min(max(col, 0), width - 1)
            read_yoff = min(max(row, 0), height - 1)
            read_xsize = windowSize
            read_ysize = windowSize
            start_row = 0
            start_col = 0
            if col < 0:
                read_xsize = windowSize + col
                start_col = -col
            if col > width - windowSize:
                read_xsize = min(windowSize, width - col)
            if row < 0:
                read_ysize = windowSize + row
                start_row = -row
            if row > height - windowSize:
                read_ysize = min(windowSize, height - row)
            data = self.raster_data.ReadAsArray(xoff=read_xoff, yoff=read_yoff, xsize=read_xsize, ysize=read_ysize)
            patchData[i, start_row:start_row+read_ysize, start_col:start_col+read_xsize, :] = data.transpose((1, 2, 0))
        return patchData

    def get_image_size(self):
        return self.raster_data.RasterYSize, self.raster_data.RasterXSize, self.raster_data.RasterCount

    def get_colors(self):
        numberList = self.class_df['Number'].values
        colorList = np.zeros([len(numberList), 3], dtype=int)
        for i in numberList:
            row = self.class_df[self.class_df['Number'] == i]
            colorList[i, 0] = row['R'].values[0]
            colorList[i, 1] = row['G'].values[0]
            colorList[i, 2] = row['B'].values[0]
        return colorList

    def get_color_by_index(self, index):
        # index = int(index) + 1
        if index < 0:
            index = 0
        return self.colorList[index, 0], self.colorList[index, 1], self.colorList[index, 2]