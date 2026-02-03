import numpy as np
from pathlib import Path
import sys
import os
import scipy.io
from scipy.signal import butter, sosfilt
import mne

# For suppressing print statements within a function.  Place function after - with HiddenPrints(): From https://stackoverflow.com/questions/8391411/how-to-block-calls-to-print
class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    sos = butter(order, [low, high], analog=False, btype='band', output='sos')
    return sos

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    sos = butter_bandpass(lowcut, highcut, fs, order=order)
    y = sosfilt(sos, data)
    return y

# https://stackoverflow.com/questions/58065055/floor-and-ceil-with-number-of-decimals
def my_ceil(a, precision=0):
    return np.true_divide(np.ceil(a * 10**precision), 10**precision)

def my_floor(a, precision=0):
    return np.true_divide(np.floor(a * 10**precision), 10**precision)

def non_overlapping_averaged_windows(data, window_size):
    """
    Computes non-overlapping averaged windows of a time series.
    
    Parameters:
    -----------
    data: numpy.ndarray
        The time series data.
    window_size: int
        The size of the window.
    
    Returns:
    --------
    numpy.ndarray
        An array of averaged window values.
    """
    num_windows = len(data) // window_size
    windowed_data = data[:num_windows*window_size].reshape(num_windows, window_size)
    return np.mean(windowed_data, axis=1)

def stc_aligner(stc,stc_est_region):
    """ MNE-Python Helper Function.
    Code which outputs vertices which match the original sources only.
    stc = original sources, stc_est_region = estimated sources.
    Alternatively, if the original sources' labels are available, just use stc_est_region.in_label(label).
    Code will bug out if one of the hemispheres (lh or rh) is empty in stc.   

    Parameters:
    -----------
    stc : mne.SourceEstimate
        The original source time series.
    stc_est_region : mne.SourceEstimate
        The estimated source time series.
    
    Returns:
    --------
    trial : mne.SourceEstimate
        The estimated source time series with the same vertices as the original source time series.
    """    
    # mat_index is the index of the vertices in the stc.data matrix.
    mat_index = []
    # v_number is the vertex number
    v_number = []
    for hemisphere in np.arange(2):
        list = []
        list3 = []    
        for vertex in stc_est_region.vertices[hemisphere]:
            if vertex in stc.vertices[hemisphere]:
                # Record the vertex number
                list.append(vertex)       
                # Record the index of the vertex in the stc_est_region
                list3.append(np.where(stc_est_region.vertices[hemisphere] == vertex)[0][0]) 
        v_number.append(np.array(list))
        mat_index.append(np.array(list3))

    trial = stc_est_region.copy()

    # Keep only the relevant rows in the stc.data matrix
    # stc.data.shape = (dipoles, datapoints). When it is a single array, the left hemisphere is stored in data[:len(vertices[0])] and the right hemisphere is stored in data[-len(vertices[1]):].

    # Initialise new data array
    data = np.zeros((len(stc.vertices[0]) + len(stc.vertices[1]), stc_est_region.data.shape[1]))

    # Fill in the data array using the lh_data and rh_data attributes.  
    # For left hemisphere:
    # data[:len(v_number[0])] = trial.lh_data[mat_index[0]]
    # For right hemisphere:
    # data[-len(v_number[1]):] = trial.rh_data[mat_index[1]]
    # Alternative which also works and doesn't rely on attributes:
    data[:len(v_number[0])] = trial.data[mat_index[0]]
    data[-len(v_number[1]):] = trial.data[mat_index[1]+len(stc_est_region.vertices[0])]

    # Check that all vertices are populated
    print(np.sum(trial.data==0))

    # Replace the vertices (though v_number is equivalent to stc.vertices) - This needs to occur last to not affect the lh_data and rh_data properties which are called. But before the data is replaced.
    trial.vertices = v_number
    # Replace the data in the stc object
    trial.data = data

    # Checks
    # print(trial.get_peak(hemi='lh'))
    # print(trial.get_peak(hemi='rh'))
    # print(stc_est_region.get_peak(hemi='lh'))
    # print(stc_est_region.get_peak(hemi='rh'))
    # # Proof that the peak is one of the vertices
    # print(np.where( trial.vertices[1] == stc_est_region.get_peak(hemi='rh')[0]))
    # # Can also check with the metrics

    return trial

def map_data_to_full_src(stc, src):
    """
    Map the data from stc to the full source space defined in src.
    
    Parameters:
    stc : SourceEstimate
        The source estimate object containing the data to be mapped.
    src : MNE source space
        The source space object containing the full source space information.
        
    Returns:
    wholestc : SourceEstimate
        The source estimate object with the data mapped to the full source space by inserting rows of zeros.
    """

    # Create a new array with zeros for the desired length
    wholesrc_data = np.zeros((src[0]['vertno'].shape[0] + src[1]['vertno'].shape[0], stc.data.shape[1]))

    # Map the data from stc.data to the new array - Left Hemisphere
    vertex_map = {vertex: idx for idx, vertex in enumerate(stc.vertices[0])}
    for idx, vertex in enumerate(src[0]['vertno']):
        if vertex in vertex_map:
            wholesrc_data[idx] = stc.data[vertex_map[vertex]]

    # Map the data from stc.data to the new array - Right Hemisphere
    vertex_map = {vertex: idx for idx, vertex in enumerate(stc.vertices[1])}
    for idx, vertex in enumerate(src[1]['vertno']):
        if vertex in vertex_map:
            wholesrc_data[idx+src[0]['vertno'].shape[0]] = stc.data[vertex_map[vertex]]

    # Generate new stc object with the full source space data
    wholestc = stc.copy()
    wholestc.vertices = [src[0]['vertno'], src[1]['vertno']]
    wholestc.data = wholesrc_data

    return wholestc

def counts_hists(array, hist_rang=None, num_bins=100, density=True):
    """ MNE-Python Helper Function.
    Helper function for getting the counts for each row of a 2D array as plt.hist groups it in the other dimension.
    This is to then plot the activity distribution of each dipole source.
    
    Parameters:
    -----------
    array : numpy.ndarray
        The 2D array.
    hist_rang : tuple
        The range of the histogram.
    num_bins : int
        The number of bins.
    density : bool
        If True, the histogram is normalised.

    Returns:    
    --------
    arr : numpy.ndarray
        The histogram counts for each row.
    bins : numpy.ndarray
        The bin edges.
    """
    num_rows, num_cols = array.shape

    # Set the number of bins for the histogram
    #num_bins = 100

    # Set the range for the histogram
    if hist_rang is None:
        hist_rang = (np.nanmin(array), np.nanmax(array))

    arr = []
    # Iterate over each row of the array
    for i in range(num_rows):
        # if not np.isnan(array[i]).all():
        # Calculate the histogram for the current row
        hist, bins = np.histogram(array[i], bins=num_bins, range=hist_rang, density=density)
        arr.append(hist)

    arr = np.array(arr)

    return arr, bins

def empty_stc_remover(stc):
    """ MNE-Python Helper Function.
    This function removes sources which are inactive (values = 0 for the entire time period) from the source space.  Can combo with the thresholding function.
    
    Parameters:
    -----------
    stc : mne.SourceEstimate
        The source time series.
    
    Returns:
    --------
    stc : mne.SourceEstimate
        The source time series with the empty sources removed.
    """
    stc = stc.copy()
    indices = stc.data.sum(axis=1) != 0
    l_vertices = len(stc.vertices[0])
    stc.vertices = [stc.vertices[0][indices[:l_vertices]], stc.vertices[1][indices[l_vertices:]]]
    # Data needs to be adjusted after vertices
    stc.data = stc.data[indices]

    return stc

def apply_solver(solver, evoked, forward, noise_cov, loose=0.2, depth=0.8, pick_ori=None, process_gain=True, **solver_kwargs):
    """ MNE-Python Helper Function. 
    Call a custom solver on evoked data.  Adjusted from original code slightly.

    This function does all the necessary computation:

    - to select the channels in the forward given the available ones in the data
    - to take into account the noise covariance and do the spatial whitening
    - to apply loose orientation constraint as MNE solvers
    - to apply a weigthing of the columns of the forward operator as in the
      weighted Minimum Norm formulation in order to limit the problem of depth bias.

    Parameters
    ----------
    solver : callable
        The solver takes 3 parameters: data M, gain matrix G, number of
        dipoles orientations per location (1 or 3). A solver shall return
        2 variables: X which contains the time series of the active dipoles
        and an active set which is a boolean mask to specify what dipoles are
        present in X.
    evoked : instance of mne.Evoked
        The evoked data
    forward : instance of Forward
        The forward solution.
    noise_cov : instance of Covariance
        The noise covariance.
    loose : float in [0, 1] | 'auto'
        Value that weights the source variances of the dipole components
        that are parallel (tangential) to the cortical surface. If loose
        is 0 then the solution is computed with fixed orientation.
        If loose is 1, it corresponds to free orientations.
        The default value ('auto') is set to 0.2 for surface-oriented source
        space and set to 1.0 for volumic or discrete source space.
    depth : None | float in [0, 1]
        Depth weighting coefficients. If None, no depth weighting is performed.
    pick_ori : None | 'normal' | 'vector'
        Method to pool vector orientations into one.  Irrelevant for fixed orientation.  If None, take the magnitude of the vector.
        If 'normal', take the normal component of the estimated vector to the surface.  If 'vector', take the vector itself.
    process_gain : bool
        If True, the forward and gain matrix is processed to take into account the
        noise covariance (whiten) and the depth weighting. If False, the gain matrix is not processed.
    solver_kwargs : dict
        Additional keyword arguments to pass to the solver based on the custom solver used.

    Returns
    -------
    stc : instance of SourceEstimate
        The source estimates.
    """
    # Import the necessary private functions
    from mne.inverse_sparse.mxne_inverse import (
        _make_sparse_stc,
        _prepare_gain,
        _reapply_source_weighting,
        is_fixed_orient,
    )

    if loose == 0:
        pick_ori = None
        print('Warning: Fixed orientation is not supported for pick_ori = "normal".  Adjusted to pick_ori = None')

    if process_gain:
    # Process the gain matrix to take into account the noise covariance and depth weighting using MNE's _prepare_gain function
        all_ch_names = evoked.ch_names
        # Handle depth weighting and whitening (here is no weights)
        forward, gain, gain_info, whitener, source_weighting, mask = _prepare_gain(
            forward,
            evoked.info,
            noise_cov,
            pca=False,
            depth=depth,
            loose=loose,
            weights=None,
            weights_min=None,
            rank=None,
        )

        # Select channels of interest
        sel = [all_ch_names.index(name) for name in gain_info["ch_names"]]
        M = evoked.data[sel]

        # Gain should be already whitened from _prepare_gain(), but keeping as per original MNE-Python code
        M = np.dot(whitener, M)
    else:
    # If for some reason you want to use the forward as-is.
        M = evoked.data
        gain = forward['sol']['data']

    n_orient = 1 if is_fixed_orient(forward) else 3
    X, active_set, var_exp, w = solver(M, gain, n_orient, **solver_kwargs)
    if process_gain:
        # This is necessary to reapply the source weighting after the whitener dot product is applied a few lines above.
        X = _reapply_source_weighting(X, source_weighting, active_set)  

    # I think pick_ori == 'normal' works but have not tested properly
    if pick_ori == 'normal':
        # Take only the normal component which is the first one out of the 3
        X = X[0::3, :]
        active_set = active_set[0::3]
        stc = _make_sparse_stc(
            X, active_set, mne.convert_forward_solution(forward, force_fixed=True, copy=True), tmin=evoked.times[0], tstep=1.0 / evoked.info["sfreq"], pick_ori=None,
        )
    else:
        stc = _make_sparse_stc(
            X, active_set, forward, tmin=evoked.times[0], tstep=1.0 / evoked.info["sfreq"], pick_ori=pick_ori,
        )
    return stc, var_exp, w

def log_exp_var(data, res, prefix="    "):
    """Log the explained variance of a signal as taken from MNE-Python. 
    Note, this metric assumes the data is already baseline corrected.
    
    Paramters:
    ----------
    data : numpy.ndarray
        The original data.
    res : numpy.ndarray
        The residuals of the data. 
    prefix : str
        The prefix for the log message.

    Returns:
    --------
    var_exp : float
        The explained variance.
    """
    var_exp = 1 - ((res * res.conj()).sum().real / (data * data.conj()).sum().real)
    var_exp *= 100
    print(f"{prefix}Explained {var_exp:5.1f}% variance")
    return var_exp

def find_each_nearest_vertex(arr1, arr2):
    """ Find the nearest vertex in arr2 for each vertex (row) in arr1.

    Parameters:
    -----------
    arr1 : numpy.ndarray
        The first array of vertices.
    arr2 : numpy.ndarray
        The second array of vertices.

    Returns:
    --------
    indices : numpy.ndarray
        The indices of the nearest vertices in arr2 for each vertex in arr1.
    distances : numpy.ndarray
        The distances to the nearest vertices in arr2 for each vertex in arr1.
    """

    from scipy.spatial import cKDTree

    # Build a KDTree for arr2 vertices
    tree = cKDTree(arr2)

    # Query the nearest neighbor for each row 
    distances, indices = tree.query(arr1)

    return indices, distances