# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 15:44:01 2013

@author: Andrew Kondrath
"""

import numpy as np, cv2

class RoadColorSegmentation(object):
    
    def __init__(self):
        
        # opening file
        movieName = 'Video Sep 05, 9 46 59 AM.mov'
        self.vc = cv2.VideoCapture(movieName)
        if not self.vc.isOpened():
            print 'Error opening video file'
        
        # skipping the first few frames (to where the car is on the road)
        for ii in range(500):
            self.vc.grab()
            
        # reading and sub-sampling the image 
        (ret, self.imgColor) = self.vc.read()
        self.subSampling = 8
        self.imgColor = self.imgColor[::self.subSampling, ::self.subSampling, :].copy()
        
        # size of the image        
        self.m, self.n, o = np.shape(self.imgColor)  
        
        cv2.namedWindow('Image', cv2.CV_WINDOW_AUTOSIZE)
        cv2.imshow('Image', self.imgColor)
        
        cv2.namedWindow('Labels', cv2.CV_WINDOW_AUTOSIZE)
        cv2.namedWindow('Labels Colored', cv2.CV_WINDOW_AUTOSIZE)
     
        self.alpha = 0
        self.U = np.array([[-0.8012116,  0.2644258,  0.53678574],
                           [ 0.15724705, -0.77249313,  0.61524608]])
     
        # point in image assumed to be part of the road
        self.x = self.m // 3 * 2
        self.y = self.n // 2
        self.wid = 5
        
        
    def setupDistanceCost(self, ellipseRatio = 3, penalty = 0.02):

        # It is assumed that the road tends to only occupy the lower, middle 
        # section of the image. This matrix is set up to penalize pixels for
        # being farther away from the assumed road point (self.x, self.y).
        # There is a constant, higher constraint on the upper third of the 
        # image.
        xv = np.array(range(self.m), dtype = 'float32')
        xv[xv < self.m//3] = 0.
        yv = np.array(range(self.n), dtype = 'float32')
        xv = xv - self.x
        yv = yv - self.y
        self.distCost = [[penalty * (ellipseRatio**2 * xi**2 + yi**2)**0.5 for yi in yv] for xi in xv]        
        self.distCost = np.array(self.distCost, dtype = 'float32')
     
    def getPixelMahalanobisDistance(self):
        # http://en.wikipedia.org/wiki/Mahalanobis_distance        
        
        # Finding the squared Mahalanobis distance between each pixel and the 
        # sample mean, weighted by the inverse covariance of the sample.
        img = np.reshape(self.imgColor.astype(np.float32), (self.m*self.n, 3))
        imgCentered = img - self.sampleMean
        temp = np.dot(imgCentered, self.sampleInf)
        distance = np.sum(temp * imgCentered, axis = 1)
        distance = np.reshape(distance, (self.m, self.n))
        
        return distance
        
    def colorParams(self):
        # http://en.wikipedia.org/wiki/Sample_mean_and_sample_covariance
        
        # resetting the label image
        self.labels = np.zeros((self.m, self.n), dtype = 'uint8')

        # taking samples around the assumed raod pixel
        samples = self.imgColor[self.x-2*self.wid:self.x+2*self.wid+1, self.y-self.wid:self.y+self.wid+1, :].astype(np.float32)
        samples = np.reshape(samples, ((4*self.wid + 1) * (2*self.wid+1), 3))
        
        # computing the mean and covariance of the RGB (BGR in OpenCV) values 
        # of the samples
        self.sampleMean = np.mean(samples, axis = 0)
        self.sampleInf = np.linalg.inv(3 * np.cov(samples.T))

        
    def colorSearch(self, cutoff = 5):
        
        distance = self.getPixelMahalanobisDistance()
        distance = np.sqrt(distance)
        distance += self.distCost

        # Pixels that are similar enough to the sample and located close to the
        # assumed road pixel are labeled as road pixels.
        self.labels[distance < cutoff] = 255
           
           
    def displayLabeled(self, runType = 0):
        
        # Giving a red tint to the pixels labeled as road.
        self.imgColorLabeled = self.imgColor.copy()
        self.imgColorLabeled[self.labels == 255, 2] = 255
        
        # Showing where the samples were taken from in the image.
        self.imgColorLabeled[self.x-2*self.wid : self.x+2*self.wid, self.y-self.wid: self.y+self.wid, 1] = 255
        
        cv2.imshow('Labels Colored', self.imgColorLabeled)
        cv2.imshow('Image', self.imgColor)
        cv2.imshow('Labels', self.labels)
        if runType == 1:
            cv2.namedWindow('Alpha Image', cv2.CV_WINDOW_AUTOSIZE)
            cv2.imshow('Alpha Image', self.imgAlpha)
            
        cv2.waitKey(10)
         
        
    def closeFile(self):
        self.vc.release()
        
        
    def getFrame(self):
        retval, self.imgColor = self.vc.read()
        self.imgColor = self.imgColor[::self.subSampling, ::self.subSampling, :].copy()
        
        
    def convertToLogChromaticity(self):

        # This is part of an implementation from "Road Detection Based on 
        # Illumination Invariance" by Alvarez and Lopez.
        # It maps chromacities similar to "Intrinsic Images by Entropy 
        # Minimization" byFinlayson, et al.   
 
        imgColor32 = self.imgColor.astype(np.float32)
        imgColor32[imgColor32 == 0] = 1.0
        imgLog = np.log(imgColor32)
        sumLog = np.sum(imgLog, 2) / 3.
        chromaticities = imgLog - sumLog[:,:,np.newaxis]
           
        u1 = self.U[0,:]
        u2 = self.U[1,:]
        x = np.sum(u1[np.newaxis, np.newaxis, :] * chromaticities, axis = 2)
        y = np.sum(u2[np.newaxis, np.newaxis, :] * chromaticities, axis = 2)
        self.rhos = np.zeros((np.shape(self.imgColor)[0], np.shape(self.imgColor)[1], 2))
        self.rhos[:,:,0] = x
        self.rhos[:,:,1] = y       

        
    def computeAlphaImage(self, alpha = -1):
        
        if not alpha == -1:
            self.alpha = alpha

        self.imgAlpha = self.rhos[:,:,0] * np.cos(self.alpha) + self.rhos[:,:,1] * np.sin(self.alpha)
                
        self.imgAlpha = self.imgAlpha - np.amin(self.imgAlpha)
        self.imgAlpha = (255 * self.imgAlpha / np.amax(self.imgAlpha)).astype(np.uint8)
        
        
    def calculateEntropy(self):
        # http://en.wikipedia.org/wiki/Image_histogram
        # http://en.wikipedia.org/wiki/Entropy_(information_theory)
        
        # The image histogram tells the number of pixels in an image that take 
        # on a specific range of values. For example, in this case, it tells 
        # how many pixels are between 0-3,4-7,ect. By dividing by the total 
        # number of pixels, the normalized histogram gives an extimate of the 
        # probability that a pixel will fall within a range of values 
        # (0-3, 4-7, etc.).
        
        # The entropy tells how much information is in an image. The way it 
        # is estimated here, as a larger number of pixels fall into a smaller 
        # set of ranges, the infomation is reduced. When the same number of 
        # pixels falls in each range, the entropy is maximized. 
        
        H = cv2.calcHist([self.imgAlpha], [0], None, [64], [0, 255])
        H = H[H > 0]
        H /= np.prod(np.shape(self.imgAlpha))
        
        entropy = - np.dot(H.transpose(), np.log(H))
        return entropy
        
        
    def estimateMinimumEntropyAngle(self, numAngles = 180):
        
        minEntropy = 1000.
        minEntropyAngle = 0.
        for inc in range(numAngles):
            
            angle = inc * np.pi / float(numAngles)
            self.computeAlphaImage(angle)
            entropy = self.calculateEntropy()
            if entropy < minEntropy:
                minEntropy = entropy
                minEntropyAngle = angle

        return minEntropyAngle, minEntropy
        
        
    def estimateRoadLikelihood(self, costScale = 1., thresh = 1):
        
        # Taking samples around the assumed raod pixel and determining the
        # probability that a pixel color belongs to the road.
        
        samples = self.imgAlpha[self.x-2*self.wid:self.x+2*self.wid+1, self.y-self.wid:self.y+self.wid+1]
        Hs = cv2.calcHist([samples], [0], None, [256], [0, 255])
        Hs /= np.prod(np.shape(samples))
        
        self.labels = np.zeros(np.shape(self.imgAlpha), dtype = 'uint8')
        likelihoods = np.zeros(np.shape(self.imgAlpha), dtype = 'float32')
        
        for ii in range(256):
            if Hs[ii] > 0:
                
                likelihoods[self.imgAlpha == ii] = -costScale * np.log(Hs[ii])

        # Adding an additional constraint based on pixel distance from the 
        # sample.
        likelihoods += self.distCost    
        
        self.labels[likelihoods < thresh] = 255
    
        
    def setupIlluminationInvariantSearch(self, numAngles = 180):
        
        self.convertToLogChromaticity()
        self.alpha, minE = self.estimateMinimumEntropyAngle(numAngles)
        

    def setupRecursiveBayesianColorEstimator(self):
        
        # Estimating the transformation on the color image that minimizes the 
        # entropy of the image.
        
        self.prior = np.copy(self.distCost)
        
        
    def recursiveBayesionColorEstimator(self, thresh = 4.5):
        # http://en.wikipedia.org/wiki/Recursive_Bayesian_estimation
        
        likelihood = self.getPixelMahalanobisDistance()
        
        likelihood += self.prior
        likelihood += self.distCost
        
        likelihood -= 35
        
        self.prior = likelihood
        self.labels[self.prior <  thresh] = 255
        
        # making sure the prior doesn't get too big, allowing pixels to become 
        # raod more quickly
        self.prior[self.prior > 65.] = 65.
        
        
    def setUp(self, runType = 0, distParams = [], parameters = []):
        # types are
        # 0 - parametric rgb-based estimation        
        # 1 - non-parametric illumination-invariant estimation
        # 2 - parameteric, recursive bayesian rgb-color based estimation
        
        if len(distParams) > 0:
            self.setupDistanceCost(distParams[0], distParams[1])
                
        if runType == 1:
            if len(parameters) > 0:
                self.setupIlluminationInvariantSearch(parameters[0])
            else:
                self.setupIlluminationInvariantSearch()
                
        if runType == 2:
            self.setupRecursiveBayesianColorEstimator()
        
        
    def runEstimation(self, runType = 0, parameters = []):
        
        self.getFrame()
        
        if runType == 0:
            if len(parameters) > 0:
                self.colorParams()
                self.colorSearch(parameters[0])
            else:
                self.colorParams()
                self.colorSearch()
                
        elif runType == 1:
            if len(parameters) > 0:
                self.convertToLogChromaticity()
                self.computeAlphaImage()
                self.estimateRoadLikelihood(parameters[0], parameters[1])
            else:
                self.convertToLogChromaticity()
                self.computeAlphaImage()
                self.estimateRoadLikelihood()
                
        elif runType == 2:
            if len(parameters) > 0:
                self.colorParams()
                self.recursiveBayesionColorEstimator(parameters[0])
            else:
                self.colorParams()
                self.recursiveBayesionColorEstimator()
                
        self.displayLabeled(runType)
                
                
if __name__ == '__main__':
    
    # 0 for parameteric, rgb-color based estimation example
    # 1 for non-parametric, illumination-invariant based estimation example 
    # 2 for parameteric, recursive bayesian rgb-color based estimation example
    method = 2
    # method = 1 still needs some work
    
    distanceParams = [[4, 0.02], [4, 0.005], [6, 0.05]]
    searchParams = [180] #only used when method = 1
    filterParams = [[8], [1.0, 3.0], []]
    
    RS = RoadColorSegmentation()
    
    RS.setUp(method, distanceParams[method], searchParams)
    for ii in range(2000):
        RS.runEstimation(method, filterParams[method])
            
    RS.closeFile()