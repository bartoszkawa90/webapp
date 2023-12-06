## IMPORTS
# import skimage.morphology
from .variables import *
from time import time
from cv2 import imread, imwrite, imshow, namedWindow, resize, resizeWindow, waitKey, destroyWindow, threshold, morphologyEx, INTER_AREA
from cv2 import split, THRESH_BINARY, bitwise_and, getStructuringElement, MORPH_ELLIPSE, MORPH_OPEN, MORPH_CLOSE, COLOR_BGRA2GRAY
from cv2 import findContours, RETR_TREE, CHAIN_APPROX_SIMPLE, boundingRect, WINDOW_NORMAL, drawContours, cvtColor, resize
from cv2 import setMouseCallback, EVENT_RBUTTONDOWN, FONT_HERSHEY_SIMPLEX, destroyAllWindows
from cv2 import Canny as CVCanny
from sys import exit
from copy import deepcopy
from os import listdir
from sys import version
from functools import lru_cache

# from scipy.signal import convolve2d
# from scipy.ndimage import convolve



## CLASSES

class Set:
    def __init__(self, first, second):
        self.first = first
        self.second = second

    def get_firsts(self):
        return self.first

    def get_seconds(self):
        return self.second

    def print(self):
        print(f'first {self.first} second {self.second} \n')


class Parameters:
    def __init__(self, img_path, thresholdRange=36, thresholdMaskValue=20, CannyGaussSize=3, CannyGaussSigma=0.6, CannyLowBoundry=0.1,
         CannyHighBoundry=10.0, CannyUseGauss=True, CannyPerformNMS=False, contourSizeLow=8,
         contourSizeHigh=500, whiteCellBoundry=186,  returnOriginalContours=False):
        self.img_path = img_path
        self.thresholdRange = thresholdRange
        self.thresholdMaskValue = thresholdMaskValue
        self.CannyGaussSize = CannyGaussSize
        self.CannyGaussSigma = CannyGaussSigma
        self.CannyLowBoundry = CannyLowBoundry
        self.CannyHighBoundry = CannyHighBoundry
        self.CannyUseGauss = CannyUseGauss
        self.CannyPerformNMS = CannyPerformNMS
        self.contourSizeLow = contourSizeLow
        self.contourSizeHigh = contourSizeHigh
        self.whiteCellBoundry = whiteCellBoundry
        self.returnOriginalContours = returnOriginalContours


class SegmentationResult:
    def __init__(self, cells=[], coordinates=[], contours=(), image=[]):
        self.cells = cells
        self.coordinates = coordinates
        self.contours = contours
        self.image = image



## FUNCTIONS / KEYWORDS
def plot_photo(title='None', image=None, height=1500, widht=1500):
    ''' Plots photo is given  resolution
        title - title of ploted photo
        image - image to plot
        height - height of ploted photo
        width - width of ploted photo
    '''
    # while True:
    if isinstance(title, str) == False:
        image = title
        title = "None"
    namedWindow(title, WINDOW_NORMAL)
    resizeWindow(title, height, widht)
    imshow(title, image)
    waitKey(0)
    destroyAllWindows()
    exit()


def printArr(*args):
    '''
    arg :  array which max min and shape will be printed
    '''
    for arr in args:
        print(" Array name ::   {}\n Array shape : {} \n {} \n Max : {} \n Min : {} \n ".format('ada', arr.shape, arr,
                                                                                                arr.max(), arr.min()))

# @njit
def preprocess(img, xmin=0, xmax=None, ymin=0, ymax=None):
    '''
    :param xmin: ->| cuts from left side
    :param xmax:  |<- cuts from right side
    :param ymin:  cuts from the top   // should be 800 for central photos and ~2000 for the one situated on the bottom
    :param ymax:  cuts from the bottom
    '''
    image = resize(img, (3000, 3000), INTER_AREA)
    if ymax == None: ymax = img.shape[0]
    if xmax == None: xmax = img.shape[1]
    new = image[ymin:ymin + ymax, xmin:xmin + xmax]

    return new


def contoursProcessing(contours, lowBoundry=15, highBoundry=500, RETURN_ADDITIONAL_INFORMATION=0):
    '''
    Function for finding smallest and largest contours and removing too small and too large contours
    :param contours: contours given to a function for processing
    :param lowBoundry: low limit of contour size
    :param highBoundry: high limit of contour size
    :param RETURN_ADDITIONAL_INFORMATION: if set to 1, the function returns additional information ::
            // returns smallest and largest contour and their IDs
    :return: tuple of selected contours with correct size
    '''
    # Filter contours according to size of contours
    conts = tuple([con for con in contours if con.shape[0] > lowBoundry and con.shape[0] < highBoundry])

    # Additional Data
    # if RETURN_ADDITIONAL_INFORMATION == 1:
    #     contours = conts
    #     SIZE_MAX = contours[0].shape[0]
    #     size_min = contours[0].shape[0]
    #     id_min = 0
    #     ID_MAX = 0
    #     count = 0
    #
    #     for con in contours:
    #         if con.shape[0] < size_min:
    #             size_min = con.shape[0]
    #             id_min = count
    #         if con.shape[0] > SIZE_MAX:
    #             SIZE_MAX = con.shape[0]
    #             ID_MAX = count
    #         count += 1
    #
    #     largest, smallest = contours[ID_MAX], contours[id_min]
    #     return conts, smallest, largest, id_min, ID_MAX

    return conts


def filterWhiteAndBlackCells(contours, img, whiteCellsBoundry=193):
    '''
    :param contours: contours to filter
    :param img: image on which the contours will be applied
    :return: tuple of contours with wrong contours removed
            // wrong contour = contour inside which cell is white
    '''

    conts = []

    for con in contours:
        # extract cells
        x_min, y_min, x_max, y_max = boundingRect(con)
        cell = img[y_min:y_min + y_max, x_min:x_min + x_max]

        # filter cells according to mean blue value
        blue = split(cell)[2]
        if np.mean(blue) < whiteCellsBoundry:
            conts.append(con)

    return tuple(conts)


def removeContour(contours, contourToRemove):
    '''
    :param contours:  tuple of contours // contour == numpy.ndarray
    :param contourToRemove: contour to remove
    :return: tuple of contours after deleting wrong contour
    '''
    newConts = []
    for con in contours:
        if con.shape != contourToRemove.shape:
            newConts.append(con)
        else:
            if not (con == contourToRemove).all():
                newConts.append(con)
    return newConts


def find_image(mainImg, img):
    im = np.atleast_3d(mainImg)
    tpl = np.atleast_3d(img)
    Z = mainImg.shape[2]
    h, w = img.shape[:2]

    # Integral image and template sum per channel
    sat = im.cumsum(1).cumsum(0)
    tplsum = np.array([tpl[:, :, i].sum() for i in range(Z)])

    # Calculate lookup table for all the possible windows
    a, b, c, d = sat[:-h, :-w], sat[:-h, w:], sat[h:, :-w], sat[h:, w:]
    lookup = d - b - c + a
    # Possible matches
    possible_match = np.where(np.logical_and.reduce([lookup[..., i] == tplsum[i] for i in range(Z)]))

    # Find match
    for y, x in zip(*possible_match):
        if np.all(im[y+1:y+h+1, x+1:x+w+1] == tpl):
            return True#(y+1, x+1)

    return False


def filterRepetitions(contours, img):
    '''
    :param contours: (tuple of ndarrays) contours to filter repetitions and wrong cells
    :param img: img on which the contours where found
    :return: tuple of contours after removing wrong cells
    '''
    count = 0
    # filter cells repetitions
    for con in contours:
        for con2 in contours:
            cell1 = extractCell(con, img).get_seconds()
            cell2 = extractCell(con2, img).get_seconds()
            if cell1.shape == cell2.shape:
                if (cell1 == cell2).all():
                    count += 1
        if count >= 2:
            contours = removeContour(contours, con)
        count = 0
    print(len(contours), "liczba konturów po odfiltrowaniu duplikatów")

    for con in contours:
        for con2 in contours:
            cell1 = extractCell(con, img).get_seconds()
            cell2 = extractCell(con2, img).get_seconds()

            if cell1.shape[0] > cell2.shape[0] \
                    and cell1.shape[1] >= cell2.shape[1] \
                    and find_image(cell1, cell2):
                contours = removeContour(contours, con)

            elif cell1.shape[0] >= cell2.shape[0] \
                    and cell1.shape[1] > cell2.shape[1] \
                    and find_image(cell1, cell2):
                contours = removeContour(contours, con)

    print(len(contours), "liczba konturów po odfiltrowaniu zlepek komórek")
    return tuple(contours)


# Version for colors
def extractCell(contour=None, img=None):
    x_min, y_min, x_max, y_max = boundingRect(contour)
    cell = img[y_min:y_min + y_max, x_min:x_min + x_max]
    # cell_dict = {[x_min, x_max, y_min, y_max]: cell}
    cell_set = Set([x_min, x_max, y_min, y_max], cell)

    return cell_set


def MAC(M1, M2):    # Multiply-accumulate function
    """
    :param M1: first array
    :param M2: second array
    :return: returns product of multiply and accumulate operation
    """
    return np.sum(M1 * M2)


def Convolution2D(x, h, mode="full", returnUINT8=False):
    """
    :param x: Input array
    :param h: kernel
    :param mode: mode of convolution ( determine how the output will look like
    :param returnUINT8: if True => returned result will be type np.uint8
    :return result: returns product of 2D convolution ==>  x * h
    """
    # if h.shape[0] != h.shape[1]:
    #     # raise ValueError('Kernel must be square matrix.')

    # if h.shape[0] % 2 == 0 or h.shape[1] % 2 == 0:
    #     raise ValueError('Kernel must have odd number of elements so it can have a center.')

    h = h[::-1, ::-1]
    # shape[0]  -  | num of rows
    # shape[1]  -  - num of columns
    y_shift = h.shape[0] // 2
    x_shift = h.shape[1] // 2

    if mode == "same":
        # zeros is x copy in bigger array and we work on it
        zeros = np.zeros((x.shape[0] + h.shape[0] - 1, x.shape[1] + h.shape[1] - 1))
        zeros[y_shift:y_shift + x.shape[0], x_shift:x_shift + x.shape[1]] = x
        # result is the array final values
        result = zeros.copy()
        # size corection which is essential to extract only final values
        endSizeCorrection = [int((i-j)/2) for i, j in zip(result.shape, x.shape)]

        for i in range(y_shift, y_shift + x.shape[0]):
            for j in range(x_shift, x_shift + x.shape[1]):
                # print(h)
                # print(zeros[i - y_shift: i + y_shift + 1, j - x_shift:j + x_shift + 1])
                # print(h * zeros[i - y_shift: i + y_shift + 1, j - x_shift:j + x_shift + 1])
                result[i, j] = MAC(h, zeros[i - y_shift: i + y_shift + 1, j - x_shift:j + x_shift + 1])
        if returnUINT8:
            return result[endSizeCorrection[0]:result.shape[0]-endSizeCorrection[0],
               endSizeCorrection[1]:result.shape[1]-endSizeCorrection[1]].astype(np.uint8)
        else:
            return result[endSizeCorrection[0]:result.shape[0]-endSizeCorrection[0],
               endSizeCorrection[1]:result.shape[1]-endSizeCorrection[1]]

    elif mode == "full":
        # zeros is x copy in bigger array and we work on it
        zeros = np.zeros((x.shape[0] + h.shape[0] + 1, x.shape[1] + h.shape[1] + 1))
        # result is the array final values
        result = zeros.copy()
        zeros[y_shift + 1:y_shift + x.shape[0] + 1, x_shift + 1:x_shift + x.shape[1] + 1] = x

        for i in range(y_shift, result.shape[0]-1):
            for j in range(x_shift, result.shape[1]-1):
                # print(h)
                # print(zeros[i - y_shift: i + y_shift + 1, j - x_shift:j + x_shift + 1])
                # print(h * zeros[i - y_shift: i + y_shift + 1, j - x_shift:j + x_shift + 1])
                result[i, j] = MAC(h, zeros[i - y_shift: i + y_shift + 1, j - x_shift:j + x_shift + 1])
        if returnUINT8:
            return result[1:result.shape[0]-1, 1:result.shape[1]-1].astype(np.uint8)
        else:
            return result[1:result.shape[0]-1, 1:result.shape[1]-1]


@lru_cache(maxsize=None)
def gaussKernelGenerator(size=3, sigma=1):
    '''
    NOTE: to filter whole image 2x 2DConvolution is required
    :param size: size of gauss kernel ( shape will be (size,1) )
    :param sigma: parameter used to calculate gauss kernel
    :return: returns gauss kernel
    '''
    x = np.arange(size)
    x = x - x[x.shape[0]//2]
    e = (1/((2*np.pi*sigma)**0.5)) #(1/np.sqrt(2*np.pi*sigma))
    temp = [e*np.exp((-i**2)/(2*sigma**2)) for i in x]
    return np.array(temp).reshape(size, 1)


@lru_cache(maxsize=None)
def gaussianFilterGenerator(size=3, sigma=1):
    X = np.zeros((size, size))
    Y = np.zeros((size, size))
    for i in range(2*size):
        if i < size:
            X[0, i] = Y[i, 0] = -1
        else:
            X[size-1, i-size-1] = Y[i-size-1, size-1] = 1
    result = (1/(2*np.pi*sigma*sigma)) * np.exp(  (-1*(np.power(X, 2) + np.power(Y, 2))) / (2*sigma*sigma))
    return result


def scale(arr, newMax):
    return ((arr-arr.min()) / (arr.max() - arr.min()))*newMax


def Canny(grayImage=None, gaussSize=3, gaussSigma=1, mask_x=mask_x, mask_y=mask_y, lowBoundry=1.0, highBoundry=10.0,
          performNMS=False, useGaussFilter=True):
    '''
    :param grayImage: input image in gray scale
    :param mask_x: vertical kernel
    :param mask_y: horizontal kernel
    :param lowBoundry: low limit for thresholding
    :param highBoundry: high limit for thresholding
    :param extractMore: determines values for gauss kernel 1 -> 5, 2.1 0 -> 3, 1.4
        // higher values makes cells less accurate but selects more of them
    :return: image with marked edges
    '''
    if grayImage is None:
        print("You have to give at least one argument")
        return
    if len(grayImage.shape) == 3:
        grayImage = cvtColor(grayImage, COLOR_BGRA2GRAY)

    # # zastosowanie filtru Gaussa w celu ograniczenia szumów
    # gaussKernel = gaussKernelGenerator(5, 1)
    # # convolution with gaussian kernel 2 times(rows and columns) to blure whole image
    # gImage = Convolution2D(Convolution2D(grayImage, gaussKernel, mode='same'), gaussKernel.T, mode="same")

    if useGaussFilter:
        gaussKernel = gaussKernelGenerator(gaussSize, gaussSigma)
        # gImage = convolve(convolve(grayImage, gaussKernel, mode='constant'), gaussKernel, mode='constant')
        gImage = Convolution2D(Convolution2D(grayImage, gaussKernel, mode='same'), gaussKernel, mode='same')
    else:
        gImage = grayImage

    Gx = Convolution2D(gImage, mask_x, mode='same')
    Gy = Convolution2D(gImage, mask_y, mode='same')
    # Gx = convolve(gImage, mask_x, mode='constant')
    # Gy = convolve(gImage, mask_y, mode='constant')

    ## gradient magnitude and angle(direction)
    GMag = (Gx**2 + Gy**2)**0.5 #np.sqrt(Gx**2 + Gy**2)
    Gangle = np.arctan2(Gy, Gx) * (180/np.pi)  ## angle in deg not in radians

    ## Non-maximum Suppression   ######  IN THESE SITUATION  'NMS' MAY GIVE WORSE RESULTS
    if performNMS == True:
        print("Performing NMS")
        rowNum, colNum = GMag.shape
        result = np.zeros((rowNum, colNum))
        # we want to consider 3x3 matrixes so we do not teke first and last
        for row in range(1, rowNum-1):
            for col in range(1, colNum-1):
                angle = Gangle[row, col]
                if (angle>=0 and angle<=22.5) or (angle<0 and angle>=-22.5) or (angle>=157.5 and angle<=180) \
                    or (angle>=-180 and angle<=-157.5):
                    edge1 = GMag[row-1, col]
                    edge2 = GMag[row+1, col]
                elif (abs(angle)<112.5 and abs(angle)>67.5):
                    edge1 = GMag[row, col - 1]
                    edge2 = GMag[row, col + 1]
                elif (angle>22.5 and angle<=67.5) or (angle>-157.5 and angle<=-112.5):
                    edge1 = GMag[row + 1, col - 1]
                    edge2 = GMag[row - 1, col + 1]
                elif (angle<-22.5 and angle>=-67.5) or (angle>=112.5 and angle<157.5):
                    edge1 = GMag[row - 1, col - 1]
                    edge2 = GMag[row + 1, col + 1]
                else:
                    print("Something went wrong with Non-maximum Suppression")
                    return
                # sprawdzamy po kątach w którą stone idzie nasza krawędz ale do ostatecznego wyniku
                # idą tylko najwyzsze wartosci zeby zostawic cienką krawędz
                if GMag[row, col] >= edge1 and GMag[row, col] >= edge2:
                    result[row, col] = GMag[row, col]
    else:
        result = GMag

    ## Thresholding
    # chodzi o to ze jest granica górna i dolna i :\
    #     jesli wartosc pixeli jest wieksza niz górna granica to na pewno mamy krawędź
    #     jesli wartość pixeli jest nizsza niz dolna granica to na pewno nie jest to krawędź
    #     jesli wartosc jest pomiedzy granicami to aby byc uznana za czesc krawedzi musi sie
    #     łączyć z pixelami o wartości powyzej górnej granicy czyli z pewną krawędzią

    np.where(result < lowBoundry, result, 0.0)
    np.where(result > highBoundry, result, 255.0)
    neighborPixels = np.zeros((3, 3))
    for i in range(1, result.shape[0] - 1):
        for j in range(1, result.shape[1] - 1):
            if result[i, j] != 0 and result[i, j] != 255:
                neighborPixels = result[i-1:i+1, j-1:j+1]
                if np.any(neighborPixels >= highBoundry):
                    result[i, j] = 255
                else:
                    result[i, j] = 0


    return scale(result, 255).astype(np.uint8)#scale(result, 255).astype(np.uint8)#scale(GMag, 255).astype(np.uint8)


def imageThreshold(grayImage, localNeighborhood=51, lowLimitForMask=20):
    '''
    :param image: input image which will be thresholded
    :param lcoalNeighborhood: size of part of image which will be considered for threshold
    :return: image after thresholding
    '''

    # what if given image is not in gray scale
    if len(grayImage.shape) >= 3:
        image = cvtColor(grayImage, COLOR_BGRA2GRAY)

    result = np.zeros_like(grayImage)   # zeros_like creates copy of given array and filled with zeros

    # filter background and making a mask
    ret, mask = threshold(grayImage, lowLimitForMask, 255, THRESH_BINARY)
    grayImage = bitwise_and(grayImage, grayImage, mask=mask)

    # iteration through every pixel on image
    for row in range(grayImage.shape[0]):
        for col in range(grayImage.shape[1]):
            # Calculate the size of neighborhood
            min_row = max(0, row - localNeighborhood // 2)
            max_row = min(grayImage.shape[0], row + localNeighborhood // 2 + 1)
            min_col = max(0, col - localNeighborhood // 2)
            max_col = min(grayImage.shape[1], col + localNeighborhood // 2 + 1)

            # Extract the neighborhood part of image
            neighborhood = grayImage[min_row:max_row, min_col:max_col]

            # Calculate the local threshold using Gaussian weighted average
            # np.std function to calculate standard deviation(odchylenie standardowe) , equation of
            # np.std() is  np.sqrt(np.mean(abs(a - a.mean())**2))
            # std maybe useful but it is not necessary

            local_threshold = np.mean(neighborhood)

            ## Use previously calculated local threshold
            if grayImage[row, col] > local_threshold:
                result[row, col] = 255
            else:
                result[row, col] = 0

    # apply morphology -- get getStructuringElement składa nam maciez o zadanych wymiarach która bedzie nam potrzebna
    #   -- morphologyEx pozwala wyłapać kontur : MORPH_OPEN czysci tło ze smieci a MORPH_CLOSE czysci kontury komórek
    #      ze smieci
    kernel = getStructuringElement(MORPH_ELLIPSE, (5, 5))
    thresh = morphologyEx(result, MORPH_OPEN, kernel)
    kernel = getStructuringElement(MORPH_ELLIPSE, (12, 12))
    result = morphologyEx(thresh, MORPH_CLOSE, kernel)

    return result


def split_on_lists(cell=None):
    red = [rgb[0] for rgb in cell]
    green = [rgb[1] for rgb in cell]
    blue = [rgb[2] for rgb in cell]

    return red, green, blue


@lru_cache(maxsize=None)
def main(params):
    '''
    '''
    # Reading an image in default mode
    if isinstance(params.img_path, str):
        img = imread(params.img_path)
    else:
        img = params.img_path

    # preprocessing
    # img = preprocess(img, xmin=500, xmax=1000, ymin=500, ymax=1000)
    img = preprocess(img)
    print(f" Image after preprocessing {img.shape}")

    # get blue value
    blue = split(img)[2]

    ## apply adaptive threshold
    # Oficjalnie najlepsza wartość threshold dla obrazu przyciętego i resized na (3000, 3000) to 51
    if params.thresholdRange == None and params.thresholdMaskValue == None:
        blob = imageThreshold(blue)
    elif  params.thresholdMaskValue == None and params.thresholdRange != None:
        blob = imageThreshold(blue, localNeighborhood=params.thresholdRange)
    elif params.thresholdMaskValue != None and params.thresholdRange == None:
        blob = imageThreshold(blue, lowLimitForMask=params.thresholdMask)
    else:
        blob = imageThreshold(blue, localNeighborhood=params.thresholdRange, lowLimitForMask=params.thresholdMaskValue)

    # Finding edges
    # edged = Canny(blob, gaussSize=params.CannyGaussSize, gaussSigma=params.CannyGaussSigma,
    #               lowBoundry=params.CannyLowBoundry, highBoundry=params.CannyHighBoundry,
    #               useGaussFilter=params.CannyUseGauss, performNMS=params.CannyPerformNMS)
    edged = CVCanny(blob, 70, 200, 5, L2gradient=False)

    contours, hierarchy = findContours(edged, RETR_TREE, CHAIN_APPROX_SIMPLE)
    print("Number of contours at first {}".format(len(contours)))

    # Filtering cells by size
    conts = []
    if params.contourSizeLow != None and params.contourSizeHigh != None: conts = contoursProcessing(contours,
                                                                                                    lowBoundry=params.contourSizeLow,
                                                                                                    highBoundry=params.contourSizeHigh)
    elif params.contourSizeLow == None and params.contourSizeHigh != None: conts = contoursProcessing(contours, highBoundry=params.contourSizeHigh)
    elif params.contourSizeLow != None and params.contourSizeHigh == None: conts = contoursProcessing(contours, lowBoundry=params.contourSizeLow)
    elif params.contourSizeLow == None and params.contourSizeHigh == None: conts = contoursProcessing(contours)

    print("Number of contours after size filtering : ", len(conts))

    # filtering cells by color and removing duplicats
    if params.whiteCellBoundry == None:
        goodConts = filterWhiteAndBlackCells(contours=conts, img=img)
    else:
        goodConts = filterWhiteAndBlackCells(contours=conts, img=img, whiteCellsBoundry=params.whiteCellBoundry)

    # filter contours repetitions
    finalConts = filterRepetitions(goodConts, img)

    image_copy = deepcopy(img)
    if params.returnOriginalContours:
        cells = [extractCell(c, image_copy) for c in contours]
        coordinates = [cell.get_firsts() for cell in cells]#list(cells_dicts.keys())
        cells = [cell.get_seconds() for cell in cells]#list(cells_dicts.values())
        return SegmentationResult(cells, coordinates, contours, img)
    else:
        cells = [extractCell(c, image_copy) for c in finalConts]
        coordinates = [cell.get_firsts() for cell in cells]#list(cells_dicts.keys())
        cells = [cell.get_seconds() for cell in cells]#list(cells_dicts.values())
        return SegmentationResult(cells, coordinates, finalConts, img)



def save_cells(cells, coordinates, dir='Cells', name_addition=''):
    # SAVE Cells in ./Cells
    print('SAVING CELLS')
    iter = 0
    if coordinates != None:
        for cell, coordiante in zip(cells, coordinates):
            imwrite(f'{dir}/xmin_{coordiante[0]} xmax_{coordiante[1]} ymin_{coordiante[2]} ymax_{coordiante[3]} cell{iter}{name_addition}.jpg',
                        cell)
            iter += 1
    else:
        iter = 0
        for cell in cells:
            imwrite(f"{dir}/cell"+str(iter)+".jpg", cell)
            iter += 1


def get_coordinates_from_filename(path):
    return tuple([int(cor[0]) for cor in [ele.split(' ') for ele in path.split('_')][1:5]])


# def test2():
#     print('Hello \n')
#     print(sys.version)
#     return 2
# test2()
#
#
# test()


## test LoG / Canny  -----------------------------------------------------------

# img = imread('spodnie.jpeg')
# img = imread('Wycinki/resized_wycinek_4_67nieb_82czar.jpg')
# img = imread('zdj_z_arykułu.png')

# gray = cvtColor(img, COLOR_BGRA2GRAY)
# Canny(gray)
# plot_photo("From Canny", LoG(gray))
# plot_photo("From Canny", Canny(gray, lowBoundry=1.0, highBoundry=10.0))
# plot_photo("cv2 Canny", Canny(gray, 100, 200, 10, L2gradient=True))

# edge = Canny(gray, 1, 10)
# contours, hierarchy = findContours(edge, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE)
# conts = delete_incorrect_contours(contours)
# plot_photo("From Canny", edge)
# drawContours(img, conts, -1, (0, 255, 0), 3)
# plot_photo("From Canny", img)


# test Canny 2
# img = imread('Wycinki/resized_wycinek_4_67nieb_82czar.jpg)
# gray = cvtColor(img, COLOR_BGRA2GRAY)
# plot_photo("From Canny", Canny(gray))


# ## test COnvolution
# mask_x = np.zeros((3, 1))
# mask_x[0] = -1
# mask_x[2] = 1
# mask_y = mask_x.T
# printArr(mask_x, mask_y)

# # non square kernel
# print("mine   \n", Convolution2D(exampleArray, mask_y, mode="same"))
# # print("scipy.signal  \n", sig.convolve2d(exampleArray, mask_y, mode='same').astype(np.uint8))
# # print("scipy.ndimage   \n", scipy.ndimage.convolve(exampleArray, mask_y, mode="constant"))
# # print("cv2 filter \n", filter2D(exampleArray, -1, mask_x))
# # print("corelation \n ", scipy.ndimage.correlate(exampleArray, mask_x))
# # print("\n")

# # square input image
# print("mine   \n", Convolution2D(MALAexample, mask_y, mode="same"))
# # print("scipy.signal  \n", sig.convolve2d(MALAexample, mask_y, mode='same').astype(np.uint8))
# # print("scipy.ndimage   \n", scipy.ndimage.convolve(MALAexample, mask_y, mode="constant"))
# # print("cv2 filter \n", filter2D(MALAexample, -1, mask_x))
# # print("\n")

# # non square input image
# print("mine   \n", Convolution2D(exampleArray, exampleKernel, mode="same"))
# # print("scipy.signal  \n", sig.convolve2d(exampleArray, exampleKernel, mode='same').astype(np.uint8))
# # print("scipy.ndimage   \n", scipy.ndimage.convolve(exampleArray, exampleKernel, mode="constant"))
# # print("\n")



# print(scipy.ndimage.filters.convolve(MALAexample, gauss))   # to samo co wyzej







