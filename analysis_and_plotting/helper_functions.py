from netCDF4 import Dataset
import numpy as np



def load_data_from_nc(data_dir):
    files_train = ['wp50.nc', 'wp60.nc', 'wp75.nc', 'wp80.nc']
    file_test = 'wp90.nc'
    nctrains = [Dataset(data_dir + f, 'r') for f in files_train]
    nctest = Dataset(data_dir + file_test, 'r')
    return nctrains, nctest

#20241204: load the test set as list too.
def load_data_from_nc_as_lists(data_dir):
    files_train = ['wp50.nc', 'wp60.nc', 'wp75.nc', 'wp80.nc']
    files_test = ['wp90.nc']
    nctrains = [Dataset(data_dir + f, 'r') for f in files_train]
    nctest = [Dataset(data_dir + f, 'r') for f in files_test]
    return nctrains, nctest


# def load_variable(ncdata, ncindex, variable, rec_slice, yslice, xslice): #OUT OF ORDER
#     # print (np.shape(ncdata[ncindex].variables[variable][0, rec_slice, yslice, xslice]),'shape of T')
#     if variable == 'T_xy_ins':
#         # print (np.shape(ncdata[ncindex].variables[variable][rec_slice, 0, yslice, xslice]),'shape of T')
#         return ncdata[ncindex].variables[variable][rec_slice, 0,  yslice, xslice]
#     elif variable == 'u_xy_ins':
#         return ncdata[ncindex].variables[variable][rec_slice, 0,  yslice, xslice]
#     else:
#         return ncdata[ncindex].variables[variable][rec_slice, yslice, xslice]
#     # return ncdata[ncindex].variables[variable][rec_slice, yslice, xslice]

    
def load_variable_from_nc(data_dir, energy_level, var_name):
    '''
    energy level takes in 'wp50.nc', 'wp60.nc', 'wp75.nc', 'wp80.nc', 'wp90.nc'
    variable takes in ssh_ins, ssh_lof, ssh_cos/sin, u_xy_ins,..., T_xy_ins
    '''
    ytest_slice = slice(0, 720)
    xtest_slice = slice(0, 256)
    rectest_slice = slice(0, 150)
    ncvar = Dataset(data_dir + energy_level, 'r')
    data_squeezed = np.squeeze(ncvar.variables[var_name])
    return data_squeezed[rectest_slice, ytest_slice, xtest_slice]
