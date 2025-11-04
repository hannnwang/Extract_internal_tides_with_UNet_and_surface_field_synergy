from netCDF4 import Dataset
import numpy as np
from scipy.signal import convolve2d, convolve

def load_data_from_nc(data_dir):
    files_train = ['wp50.nc', 'wp60.nc', 'wp75.nc', 'wp80.nc']
    file_test = 'wp90.nc'
    nctrains = [Dataset(data_dir + f, 'r') for f in files_train]
    nctest = Dataset(data_dir + file_test, 'r')
    return nctrains, nctest

def load_variable(ncdata, ncindex, variable):
    return np.squeeze(ncdata[ncindex].variables[variable])


#Changed by HW
def gaussian_kernel(decaylength): 
    """Generates a Gaussian kernel."""
    #decaylength is in the unit of grid resolution (4km in Aurelien's data.) So in physical units, the decay lenght would be decaylength*(4 km).
    size=int(2*decaylength)
    sigma=decaylength/(2*np.sqrt(2*np.log(2))) #Interpretting decaylength as the FWHM of Gaussian
    kernel = np.fromfunction(
        lambda x, y: (1 / np.sqrt(2 * np.pi * sigma ** 2)) * 
                      np.exp(-((x- size/2)**2 + (y-size/2)**2) / (2 * sigma ** 2)),
        (size, size)  
    ) #Creating a kernel with 
    return kernel / np.sum(kernel)  # Normalize the kernel
    
#Changed by HW    
def degrade_space_gaussian(field, decaylength):
    nt, nx, ny = np.shape(field)
    kernel = gaussian_kernel(decaylength)
    filtered_field = np.empty([nt, nx, ny])

    for i in range(nt):
        filtered_field[i, : ,:] = convolve2d(field[i, : ,:], kernel, mode = 'same', boundary='symm')#,  fillvalue = np.average(field[i, : ,:]))
    return filtered_field




def degrade_space_uniform(field, ker_size):
    nt, nx, ny = np.shape(field)
    kernel = np.ones([ker_size, ker_size])/ker_size**2
    filtered_field = np.empty([nt, nx, ny])

    for i in range(nt):
        filtered_field[i, : ,:] = convolve2d(field[i, : ,:],\
             kernel, mode = 'same', boundary='fill',\
                fillvalue = np.average(field[i, : ,:]))
    return filtered_field

def degrade_time_uniform(field, ker_size):
    nt, nx, ny = np.shape(field)
    kernel = np.ones(ker_size)/ker_size
    filtered_field = np.empty([nt, nx, ny])
    for xi in range(nx):
        for yi in range(ny):
            filtered_field[:, xi ,yi] = convolve(field[:, xi, yi], \
                kernel, mode = 'same')
    return filtered_field

if __name__ == '__main__':
    root_dir = '/home/jeff/kaushik_pytorch/data/' 
    nctrains, nctest = load_data_from_nc(root_dir)
    print (nctest.variables)
    time = nctest.variables['time_centered']
    print (time[1:] - time[:-1])
    # 2 day averages
    
    space = nctest.variables['nav_lat_rho']
    print (space[1:] - space[:-1])    
    print (space[:, 1:] - space[:, :-1]) 
    #4 km spacing
    
    
    
    
#     s = 3
#     arr = np.arange(3*s**2).reshape([3, s, s])
#     print (arr)
#     kernal = np.ones(3)
#     time_filt = degrade_time(arr, 3)
#     print (time_filt)
    # space_filt = degrade_space(arr, 3)
    # print (space_filt)