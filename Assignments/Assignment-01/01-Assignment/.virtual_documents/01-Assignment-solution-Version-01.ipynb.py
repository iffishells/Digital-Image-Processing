# importing lib 

import numpy as np
from skimage import data
import matplotlib.pyplot as plt

from skimage.filters.rank import entropy
from skimage.color import rgb2hsv, rgb2gray, rgb2yuv
import cv2
from skimage.feature import greycomatrix
from matplotlib.colors import NoNorm
from skimage import io 
from skimage.color import rgb2gray



def show_image(image,title ="image", cmap_type = 'gray'):
    plt.imshow(image , cmap = cmap_type)
    plt.title(title)
    plt.axis('off')
    plt.show()
def image_convert_to_grayscale(image):
    return rgb2gray(image)
    
def image_size(image):
    return  (image.shape, image.size)
def image_dim(image):
    return image_dim
def signaltonoise(a, axis=0, ddof=0):
    a = np.asanyarray(a)
    m = a.mean(axis)
    sd = a.std(axis=axis, ddof=ddof)
    return np.where(sd == 0, 0, m/sd)
def image_dept(image):
    return image.shape


# reading image
image_path = "Images/ramdom1.jpg"
image  = cv2.imread(image_path)



def Question_01(image):
    image = image_convert_to_grayscale(image)
    show_image(image) # display image
    print(image.shape)
    print("image Dim and size : {}\nImage Depth : {}".format(image_size(image),
                                                             image_dept(image)))
    print(signaltonoise(image))

# Question_01(image)
def Question_02(image):
    # converting into grayscale
    image  =  image_convert_to_grayscale(image)
    print(image)
#     glcm = np.squeeze(greycomatrix(image, distances=[1], 
#                                angles=[0], symmetric=True, 
#                                normed=True))
#     print(glcm)
    
    
# Question_02(image)
    


from skimage import color
import numpy as np
import math

class Assignmet01:
    
    def __init__(self,image,title):
        self.image = image
        self.title = title
        
        
    def show_image(self,image,title ="image"  ,grayscale = None):
        if grayscale == True:
            plt.imshow(image , cmap=plt.cm.gray )
            plt.title(title)
            plt.show()
        elif grayscale == False:
            plt.imshow(image )
            plt.title(title)
            plt.show()


            
    def image_depth(self,image):
        
        one_dim = image.flatten()
        max_shade =  max(one_dim)
        print(max_shade)
        if max_shade <=1:
            return 0
        elif max_shade <=2 :
            return 1
        elif max_shade <=7 :
            return 3
        elif max_shade <=15 :
            return 4
        elif max_shade <=31 :
            return 5
        elif max_shade <=63 :
            return 6
        elif max_shade <=127 :
            return 7
        elif max_shade <=255 :
            return 8
        
        
        
#         2-0 --> 1
#         2-1 --> 2
#         2-2 --> 4
#         2-3 --> 8
#         2-4 --> 16
#         2-5 --> 32
#         2-6 --> 64
#         2-7 --> 128
#         2-8 --> 256
    def image_channel(self,image):
        
        
    
        self.show_image(image , title="Original image" ,grayscale=False)
        self.hist(image , title="Original image")
        
        
        
        red = image[:,:,0]
        self.show_image(red , title="Red image" ,grayscale = True)
        self.hist(red , title="red")
        plt.show()
        
        green = image[:,:,1]
        self.show_image(green,title = "Green image" ,grayscale=False)
        self.hist(green , title="green")
        
        blue = image[:,:,2]
        self.show_image(blue,title="Blue image" ,grayscale=False)
        self.hist(blue,title="Blue")
        
    def hist(self,image ,title = "title"):
        
        plt.hist(image.ravel(),bins=256)
        plt.title(title)
        plt.show()
    def brightness(self,image):
        Brightness = image.mean()
#         print("Brightness : ",Brightness)

        return Brightness
        
    def signaltonoise(self,a, axis=0, ddof=0):
        a = np.asanyarray(a)
        m = a.mean(axis)
        sd = a.std(axis=axis, ddof=ddof)
        return np.where(sd == 0, 0, m/sd)
        
    def conversion_to_grayscale(self,image):
        grayscale = rgb2gray(image)
    
#         print("representation of GrayScale : ", grayscale)
#         print(grayscale)
        rgb = color.gray2rgb(image)
#         print("representation of RGB : ",rgb)
        return grayscale
    def Question_01(self,image):
        # depth is missing
        image = self.conversion_to_grayscale(image)
        
#         display image 
        self.show_image(image)
        
        # display Dimension
        print("Image Dimension :{}\nImage size:{}\nSignal to Noise Ratio :{}".format(image.shape,image.size,
                                                                                    self.signaltonoise(image)))
    def Question_02(self,image):
        # 1D representation of image
        oneD_image = image.flatten()
        total_len = len(oneD_image)
        (unique, counts) = np.unique(oneD_image, return_counts=True)
#         print(unique, counts)
        unique_element_len = len(unique)
        entropy = 0
        for index in range(0,(unique_element_len)-1):
#             print(index)
            entropy +=  -(counts[index]/total_len)* (math.log2((counts[index]/total_len)))
        
        print(entropy)
        return entropy
        
    def Question_03(self,image):
        # Flip the image in up direction
        vertically_flipped = np.flipud(image)
        self.show_image(vertically_flipped)
        
    def Question_04(self,image):
        
        pass
    def log_transformation(self,value):
        value =math.log(1+value)
        return value
        
    def Question_09(self,image):
        """ log transformation formula is s = loge(1+r)  r is the pixel value
        """
        #grayscale 
#         image = self.conversion_to_grayscale(image)
        print(image.shape)
        log_image = np.uint8(np.log1p(image))
        threshold = 20
        img_3 = cv2.threshold(log_image , threshold ,255, cv2.THRESH_BINARY)[1]
        cv2.imshow("input image" ,image)
        cv2.imshow("transform image ",img_3)
        cv2.waitKey(100000)
        cv2.destroyAllWindows()
    def Question_05(self,image):
        
        neg_image = 255- image
        
        cv2.imshow("Neg image ",neg_image)
        cv2.imshow("orginal Image ",image)
        cv2.waitKey(10000)
        cv2.destroyAllWindows()
        
    def Question_10(self,image,gamma=2):
        
        for gamma in range(0,255):
            trans_img = np.power(image,gamma)
#             cv2.imshow("Input image : ",image)
#             cv2.imshow("Transform Image :",trans_img)
#             cv2.waitKey(10000)
#             cv2.destroyAllWindows()
            self.show_image(trans_img)

    def Question_07(self,image):
        
        beta = self.brightness(image) # average value
        print("beta : ",beta)
        
        # to redure the compilation of over loop or inner loop  to became computauional cost
#         one_D_image = image.flatten()
#         print("lenght of 1 d image : ",len(one_D_image))
#         print("one D image : ",one_D_image)
#         M_X_N = len(image)
        
#         count = 0
#         for each_pixel_value in one_D_image:
#             count += ((each_pixel_value - beta)**2 )
            
#         contrast = math.sqrt(count/M_X_N)
#         print("Contrast value ",contrast)
        
        contrast = image.std()
        print(contrast)
        return  contrast
            
        
    def Question_11(self,image):
        
        """ l(m,n) = (orignalPixel-image_min) * ((lmax-lmin)/image_max-image_min))+ lmin"""
        min_pixel_value = image.min()
        
        max_pixel_value = image.max()
        
        print(min_pixel_value , max_pixel_value)
        
        ratio = ((max_pixel_value-min_pixel_value) / (200 - 100)) + min_pixel_value 
        plt.figure(figsize=(12,4))
        plt.subplot(1,3,1)
        plt.imshow(image , cmap="gray" , norm=NoNorm())
        
        plt.subplot(1,3,2)
        trans_image = (image-np.min(image)) * ratio
        plt.imshow(trans_image, cmap="gray" , norm=NoNorm())
        
                
if __name__ == '__main__':
    
    
    import matplotlib.pyplot as plt

    from skimage import data
    from skimage.color import rgb2gray
    
    image = data.astronaut()
#     grayscale = rgb2gray(original)
#     ax[0].imshow(original)
#     ax[0].set_title("Original")
#     ax[1].imshow(grayscale, cmap=plt.cm.gray)
#     ax[1].set_title("Grayscale")

#     fig.tight_layout()
#     plt.show()
    
    
    object_Assignment_01 = Assignmet01(image ,title = "Image")
#     image = object_Assignment_01.conversion_to_grayscale(image)
    #orgianl image 
#     object_Assignment_01.image_channel(grayscale)
#     object_Assignment_01.Question_01(image)
#     object_Assignment_01.Question_02(image)
#     object_Assignment_01.Question_03(image)
#     object_Assignment_01.Question_04(image)
#     object_Assignment_01.image_depth(image)
#     object_Assignment_01.Question_04(image)
#     object_Assignment_01.Question_09(image)
#     object_Assignment_01.Question_10(image)
#     object_Assignment_01.Question_11(image)

    object_Assignment_01.Question_07(image)





2**3






