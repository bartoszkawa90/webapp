from django.shortcuts import render, HttpResponse
from django.conf import settings
from .forms import ImageForm
from .models import Image
import os
from .resources import *
from .classifiers import *
from WebApp.settings import STATIC_URL

acceptable_extensions = ['jpg', 'jpeg', 'pdf', 'png']
parameters = ['threshold_range', 'threshold_mask', 'cell_low_size', 'cell_high_size', 'white_cells_boundry']
default_params = [31, 20, 8, 500, 187]
default_params = {'threshold_range': 31, 'threshold_mask': 20, 'cell_low_size': 8, 'cell_high_size': 500, 'white_cells_boundry': 187}
context = {
    'image_name': '',
    'form': '',
    'image': '',
    'cur_params': default_params,
    'state': 'start',
    'new_image': False,
    'cells_differences': '',
    'segmentation_results': [],
    # 'Kmeans': 0.0,
    'KNN': 0.0,
    # 'CNN': 0.0,
    'find_contours_time': 0.0,
    'classification_time': 0.0
}


# Create your views here.
def home(request, parameters=parameters):
    # print post get data
    print(f' \n\n  Home request \n')
    if request.method == 'POST':
        print(f'POST {request.POST}')
        print(f'FILES {request.FILES}')
    if request.method == 'GET':
        print(f'GET {request.GET}')
    print("\n\n")

    if request.method == 'POST' and 'Upload' in request.POST:
        # GET IMAGE AND ITS NAME AND SAVE TO DIR AND DATABASE AFTER CLEARING PREVIOUS IMAGES-------------------------
        form = ImageForm(request.POST, request.FILES)
        if form.is_valid():
            ## CLEAR DATABASE AND IMAGES DIR TO SAVE NEW IMAGE CORRECTLY
            os.system('python manage.py flush --no-input')
            try:
                for file in os.listdir('images'):
                    os.remove(os.path.join('images/', file))
            except:
                print('[ERROR] some error with images in images occured')
            form.save()
        # CREATE CONTEXT AND SAVE IMAGE NAME
        try:
            image_name = request.FILES['image'].name
            images = Image.objects.all()
        except:
            image_name = ''
            images = [None]
        context['image_name'] = image_name
        context['form'] = form
        context['image'] = [images[0]]
        context['state'] = 'image_uploaded'
        context['new_image'] = True

    elif request.method == 'POST' and 'Find contours' in request.POST and context['state'] == 'image_uploaded':
        try:
            print('--- Find contours ---')
            contours_start_time = time()
            saved_image_name = os.listdir("images")[0]
            if context['new_image']:
                try:
                    print("--- Original image saved ---")
                    imwrite('images/original_image.jpg', imread(os.path.join('images/', saved_image_name)))
                    context['new_image'] = False
                except:
                    print("--- Something wrong original image ---")
            saved_image = imread('images/original_image.jpg')
            # saved_image = imread(os.path.join('images/', saved_image_name))

            print(f"--- Context Parameters before segmentation: {context['cur_params']}")
            parameters = Parameters(img_path=saved_image, thresholdRange=context['cur_params']['threshold_range'],
                                    thresholdMaskValue=context['cur_params']['threshold_mask'],
                                    CannyGaussSize=3, CannyGaussSigma=0.6, CannyLowBoundry=0.1, CannyHighBoundry=10.0,
                                    CannyUseGauss=True, CannyPerformNMS=False,
                                    contourSizeLow=context['cur_params']['cell_low_size'],
                                    contourSizeHigh=context['cur_params']['cell_high_size'],
                                    whiteCellBoundry=context['cur_params']['white_cells_boundry'])
            segmentation_results = main(parameters)
            print(f"--- Segmentation completed ---\n--- Have {len(segmentation_results.cells)} cells after segmentation ---")
            drawContours(segmentation_results.image, segmentation_results.contours, -1, (0, 255, 0), 3)

            imwrite(os.path.join('images/', saved_image_name), segmentation_results.image)
            context['segmentation_results'] = segmentation_results
            context['find_contours_time'] = round((time() - contours_start_time), 3)
        except:
            print("Something went wrong with finding contours ")


    # CALCULATE THE PROCENT OF BLACK CELLS WITH 3 METHODS -------------------------------------------------------------
    elif request.method == 'POST' and 'Calculate' in request.POST and context['state'] == 'image_uploaded':
        try:
            print('--- Calculate ---')
            segmentation_results = context['segmentation_results']
            class_start_time = time()

            black_path = "app/Reference/black/"
            blue_path = "app/Reference/blue/"

            blackKNN, blueKNN = KNN(segmentation_results.cells, black_path, blue_path,
                                    load_reference_coordinates_path_black='app/static/KNN_black_reference_coordicates.json',
                                    load_reference_coordinates_path_blue='app/static/KNN_blue_reference_coordicates.json',
                                    working_state='load data')
            #
            # # Unsupervised methods
            # # We use Kmeans two times it gives best results
            # blackKmeans, blueKmeans, centroids = kMeans(num_of_clusters=2, cells=segmentation_results.cells)

            ## IF THERE ARE VISIBLE DIFFERENCES BETWEEN TWO GROUPS OF CELLS USE KMEANS ONLY ONES
            # if context['cells_differences'] == 'low':
            #     kblack, kblue, cent = kMeans(num_of_clusters=2, cells=blueKmeans)
            #     blueKmeans = kblue
            #     blackKmeans = blackKmeans + kblack

            print(f" KNN :: Black {len(blackKNN)} and blue {len(blueKNN)}  /n Finale result of algorithm is  ::  "
                  f"{len(blackKNN)/(len(blueKNN) + len(blackKNN))*100} % \n")
            # print(f" CNN :: Black {len(blackCNN)} and blue {len(blueCNN)}  /n Finale result of algorithm is"
            #       f"  ::  {len(blackCNN)/(len(blueCNN) + len(blackCNN))*100} % \n")
            # print(f" Kmeans :: Black {len(blackKmeans)} and blue {len(blueKmeans)}  /n Finale result of algorithm is  ::  "
            #       f"{len(blackKmeans)/(len(blueKmeans) + len(blackKmeans))*100} % \n")

            print("--- %s seconds ---" % (time() - class_start_time), ' time after algorithm ')

            # context['Kmeans'] = round(len(blackKmeans)/(len(blueKmeans) + len(blackKmeans))*100, 3)
            context['KNN'] = round(len(blackKNN)/(len(blueKNN) + len(blackKNN))*100, 3)
            # context['CNN'] = round(len(blackCNN)/(len(blueCNN) + len(blackCNN))*100, 3)
            context['classification_time'] = round((time() - class_start_time), 3)
        except:
            print("Something went wrong  while calculating ( contours might have not been found ")

    # GET PARAMETERS  ------------------------------------------------------------------------------
    if request.method == 'GET' and 'threshold_range' in request.GET:
        try:
            cur_params = {}
            for idx in range(len(parameters)):
                if request.GET.get(parameters[idx]) == '' or request.GET.get(parameters[idx]) == None:
                    cur_params[parameters[idx]] = request.GET.get(default_params[idx])
                else:
                    cur_params[parameters[idx]] = int(request.GET.get(parameters[idx]))
            context['cur_params'] = cur_params
            context['cells_differences'] = request.GET.get('cells_differences')
            print(f'Current parameters  {cur_params} and cells differences {context["cells_differences"]}')
        except:
            print("Something went wrong with sending parameters")

    return render(request, 'homepage.html', {'context': context})


def doc(request):
    return render(request, 'documentation.html')

# def terminal(request):
#     return render(request, 'terminal.html')
