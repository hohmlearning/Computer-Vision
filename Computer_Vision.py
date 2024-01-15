import numpy as np

def otsu (img_np):
    '''
    Calculate the Otsu's threshold for an image.
    
    This function implements Otsu's method for automatic thresholding. It finds the 
    threshold that maximizes the inter-class variance and minimizes the intra-class 
    variance, i.e., the variance within each class of pixels (those above and those below the threshold). 
    The function is designed to work with images represented as 2D NumPy arrays of unsigned integer 
    type. It raises a TypeError if the image is not of an unsigned integer type. If the 
    image contains only one unique intensity, the function returns this value as the 
    threshold.
    
    Parameters
    ----------
    img_np : np.ndarray
        A 2D NumPy array representing the image. The array should be of an unsigned integer type.
    
    Raises
    ------
    TypeError
        Raised if `img_np` is not of an unsigned integer type.
    
    Returns
    -------
    threshold_otsu : int or np.uint
        The computed Otsu's threshold. This is an intensity value that separates the image
        into two classes (foreground and background) in a way that minimizes the intra-class 
        variance.
    '''
    if np.issubdtype(img_np.dtype, np.unsignedinteger) == False:
        string = 'Image type = {}. Convert the image in unsigned integer!'.format(img_np.dtype)
        raise TypeError (string)
    unique_intensivity, frequencies = np.unique(img_np, return_counts=True)
    
    if unique_intensivity.shape[0] == 1:
        threshold_otsu = unique_intensivity[0]
        return (threshold_otsu)
        
    else:
        N, M = img_np.shape
        p = frequencies / (N*M)
        mu_total = (unique_intensivity * p).sum()
    
        p_0_t = 0
        w_0_t = 0
        w_0_t_new = 0
        mu_0_t = 0
        
        var_between_max = 0
        threshold_otsu = 0
        for n, threshold in enumerate(unique_intensivity[:-1]):
            p_0_t = p[n]
            w_0_t_new = w_0_t + p_0_t
            mu_0_t = ((mu_0_t * w_0_t) + threshold * p_0_t) / w_0_t_new
            w_0_t = w_0_t_new
            mu_1_t = (mu_total - w_0_t * mu_0_t) / (1-w_0_t)
            var_between = (w_0_t * (1 - w_0_t)) * (mu_0_t - mu_1_t)**2
            
            if var_between > var_between_max:
                var_between_max = var_between 
                threshold_otsu = threshold
    return (threshold_otsu)
        
    
    
    
    
    
    
    
    
    
    
    
    
    

