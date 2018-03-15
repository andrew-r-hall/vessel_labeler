import sys
import numpy as np
import nibabel as nib
from sklearn.cluster import KMeans
#import matplotlib.pyplot as plt

'''Loads an image from a filename and returns a numpy array of voxel values'''
def load_image(filename):
    img = nib.load(filename)
    img = np.asarray(img.get_data())
    return(img)

'''Essentially does the opposite of load_image. Takes numpy array and filename
   as args, and writes a nifti with filename 'prefix' '''
def write_image(array , prefix):
    new = nib.Nifti1Image(array , np.eye(4))
    nib.save(new , prefix)
    return()
'''Writes out a new image with voxels labelled as their percentile value.
   Can be used to threshold the image as well. If undesired, pass 0 as threshold'''
def percentile(img):
    pct = np.zeros_like(img)
    percentiles = np.percentile(img[img>=1] , range(100))
    for i in range(np.shape(img)[0]):
        if(i % 10 == 0):
            sys.stdout.flush()
            sys.stdout.write('\r'+'...percentizing... '+str(i / np.shape(img)[0] * 100)+' %          ')
        for j in range(np.shape(img)[1]):
                for k in range(np.shape(img)[2]):
                    if(img[i,j,k] > 0):
                        temp = np.asarray( np.append(percentiles , img[i,j,k]) )
                        temp.sort()
                        if(np.where(temp == img[i,j,k])[0][-1] > 0):
                            pct[i,j,k] = np.where(temp == img[i,j,k])[0][-1]
                        else:
                            pct[i,j,k] = 0
    return(pct)


'''z-score scaler'''
def std_scale( array ):
    mean = np.mean(array)
    std  = np.std(array)
    if(std == 0):
        scaled = np.zeros_like(array)
    else:
        scaled = np.divide(np.subtract(array , mean),std)
    return(scaled)

'''does kmeans clustering in a smaller field of view, iterated over entire image'''
def small_fov_cluster(img , n , clusters=2):
    labeled = np.zeros_like(img)
    shape = [n,n]
    for k in range(np.shape(img)[2]):
        for i in range(0,np.shape(img)[0],n):
            for j in range(0,np.shape(img)[1],n):
                scaled = std_scale(img[i:i+n,j:j+n,k]).flatten()
                if(len(np.unique(scaled))==1):
                    labels = np.zeros_like(scaled)
                    labels = np.asarray(labels.reshape(shape))
                else:
                    X=[]
                    for x in scaled:
                        X.append([x])
                    labels = KMeans(n_clusters=clusters , random_state=0).fit_predict(X)
                    labels = np.asarray(labels.reshape(shape))
                labeled[i:i+n , j:j+n , k] = labels
        if(k % 5 == 0):
            sys.stdout.write('\r'+'...clustering... '+str(int(k / np.shape(img)[2] * 100))+' %')
    return(labeled)

'''labels each image with the average value of an nxn kernel around given point in the (0,1) plane.
   This will hopefully help distinguish between vessels and other high intensity
   structures, like WM or CSF, as they are larger stuctures'''
def blur(img , n):
    blurred_img = np.zeros_like(img)
    for k in range(np.shape(img)[2]):
        for i in range(0,np.shape(img)[0],n):
            for j in range(0,np.shape(img)[1],n):
                blurred_img[i:i+n,j:j+n,k] = np.mean(img[i:i+n,j:j+n,k])
        if(k % 10 == 0):
            sys.stdout.flush()
            sys.stdout.write('\r'+'...blurring... '+str(i / np.shape(img)[0] * 100)+' %          ')
    return(blurred_img)

'''Returns a kmeans model to be used for segmentation'''
def clusterize(pct,blurred,clusters):
    X = []
    for i in range(np.shape(pct)[0]):
        if(i % 10 == 0):
            sys.stdout.flush()
            sys.stdout.write('\r'+'...preparing to model... '+str(i / np.shape(pct)[0] * 100)+' %          ')
        for j in range(np.shape(pct)[1]):
            for k in range(np.shape(pct)[2]):
                #X.append([pct[i,j,k] , blurred[i,j,k]])
                X.append([pct[i,j,k]])

    sys.stdout.write('...building model...')
    kmeans = KMeans(n_clusters=clusters).fit(X)
    return(kmeans)

'''Segments using each point's percentile and blurred value.'''
def segment(model , pct , blurred):
    labels = np.empty_like(pct)
    for i in range(np.shape(pct)[0]):
        if(i % 10 == 0):
            sys.stdout.flush()
            sys.stdout.write('\r'+'...segmenting... '+str(i / np.shape(img)[0] * 100)+' %          ')
        for j in range(np.shape(pct)[1]):
            for k in range(np.shape(pct)[2]):
                if(pct[i,j,k] or blurred[i,j,k] != 0):
                    labels[i,j,k] = model.predict([[pct[i,j,k],blurred[i,j,k]]])
                else:
                    labels[i,j,k] = 0
    return(labels)

'''Runs through an example usage of the whole thing, taking a filename,
   kernel size, and number of clusters as args. Writes a nifti with labels.'''
def master(filename , n , clusters , threshold):
    sys.stdout.write('loading image')
    img = load_image(filename)
    pct = percentile(pct , 0)
    blurred = blur(img , n)
    model = clusterize(pct , blurred , clusters)
    labeled = segment(model , pct , blurred)
    outname = filename+'_labeled.nii'
    write_image(labeled , outname)
    return()
