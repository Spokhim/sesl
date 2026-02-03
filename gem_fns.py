import numpy as np
from scipy import linalg
import scipy.sparse
import scipy.sparse.linalg
import lapy
import brainspace.mesh as mesh
import mne

from useful_fns import apply_solver, log_exp_var, find_each_nearest_vertex
from dynsim_fns import map_data_to_full_src, centre_loader, generate_com_labels

def get_downsampled_tris(trias, vertno):
    """ MNE-Python Helper Function.
    This function is used to obtain the triangles in the downsampled source space without reference to the higher-sampled MRI mesh.  Sometimes this is required when the MNE source space
    did not produce the expected results for src[i]['use_tris'] and instead outputted it in terms of vertices in the original higher-sampled mesh. 

    Parameters
    ----------
    trias : array (no. triangles x 3)
        The triangles with indices in terms of the original higher resolution mesh from MRI.  Obtainable from src[i]['use_tris']
    vertno: array (no.vertices x 1)
        Array containing the vertex positions of each vertex used in the down-sampled source space.  Obtainable from src[i]['vertno']
    
    Returns
    -------
    new_trias : array
        3 by N array of the triangles in the downsampled source space. 
    """

    # Create an array of zeros with the same shape as trias
    new_trias = np.zeros_like(trias)

    # Create a dictionary where the keys are the elements in vertno and the values are their corresponding indices
    index_dict = {value: index for index, value in enumerate(vertno)}

    # Iterate over trias and replace each element with its index in vertno
    for index, value in np.ndenumerate(trias):
        if value in index_dict:
            new_trias[index] = index_dict[value]

    return new_trias

def get_tria_from_src(src, hemi, downsample=True):
    """MNE-Python Helper Function.
    This function extracts the points and triangles from the source space.
    
    Parameters
    ----------
    src : MNE source space
        The source space.
    hemi : int
        The hemisphere to extract the points and triangles from.  Either 0 (L) or 1 (R).
    downsample : bool
        If True, extract points and trias from the MNE source space (which should be downsampled).  If False, extract points and trias from the MRI dense surface.
    
    Returns
    -------
    points : ndarray
        The points of the source space.
    trias : ndarray
        The triangles of the source space.
    """

    if downsample:
        # If downsample is true, extract points and trias from the inputted MNE source space (which should be downsampled)
        points = src[hemi]['rr'][src[hemi]['inuse']==1]
        trias = src[hemi]['use_tris']
        # If the triangle vertices are out of index, it means they are the vertices of the higher-sampled MRI mesh and not the down-sampled src
        if trias.max() > src[hemi]['nuse']: 
            print('Downsampling the triangles')
            vertno = src[hemi]['vertno']
            trias = get_downsampled_tris(trias, vertno)
    else:
        # If downsample is false, extract points and trias from the MRI dense surface
        points = src[hemi]['rr']
        trias = src[hemi]['tris']
    
    return points, trias

def calc_eig(tria, num_modes):
    """Calculate the eigenvalues and eigenmodes of a surface.  Code taken from: https://doi.org/10.1038/s41586-023-06098-1 

    Parameters:
    -----------
    tria : lapy compatible object
        Loaded vtk object corresponding to a surface triangular mesh
    num_modes : int
        Number of eigenmodes to be calculated

    Returns:
    -------
    evals : array (num_modes x 1)
        Eigenvalues
    emodes : array (number of surface points x num_modes)
        Eigenmodes
    """
    
    fem = lapy.Solver(tria)
    evals, emodes = fem.eigs(k=num_modes)
    
    return evals, emodes

def get_indices(surface_original, surface_new):
    """Extract indices of vertices of the two surfaces that match.  Code taken from: https://doi.org/10.1038/s41586-023-06098-1 

    Parameters
    ----------
    surface_original : brainspace compatible object
        Loaded vtk object corresponding to a surface triangular mesh
    surface_new : brainspace compatible object
        Loaded vtk object corresponding to a surface triangular mesh

    Returns
    ------
    indices : array
        indices of vertices
    """

    indices = np.zeros([np.shape(surface_new.Points)[0],1])
    for i in range(np.shape(surface_new.Points)[0]):
        indices[i] = np.where(np.all(np.equal(surface_new.Points[i,:],surface_original.Points), axis=1))[0][0]
    indices = indices.astype(int)
    
    return indices

def calc_surface_eigenmodes(points, trias, mask, num_modes):
    """Calculate the eigenvalues and eigenmodes of a surface mesh with application of a cortical mask (to remove the medial wall).

    Parameters
    ----------
    points : array (number of surface points , 3)
        The vertices of the surface mesh.
    trias : array (number of triangles x 3)
        The triangles of the surface mesh.
    mask: array (number of surface points x 1)
        Mask to be applied on the surface (e.g., cortex without medial wall, values = 1 for mask and 0 elsewhere)
    num_modes : int
        Number of eigenmodes to be calculated

    Returns
    ------
    evals : array (num_modes x 1)
        Eigenvalues
    emodes : array (number of surface points x num_modes)
        Eigenmodes
    """

    # load surface (as a brainspace object)
    surface_orig = mesh.mesh_creation.build_polydata(points, trias)
    # create temporary suface based on mask
    surface_cut = mesh.mesh_operations.mask_points(surface_orig, mask)

    # new method: replace v and t of surface_orig with v and t of surface_cut
    # load surface (as a lapy object)
    tria = lapy.TriaMesh(points,trias)

    tria.v = surface_cut.Points
    tria.t = np.reshape(surface_cut.Polygons, [surface_cut.n_cells, 4])[:,1:4]

    # calculate eigenvalues and eigenmodes
    evals, emodes = calc_eig(tria, num_modes)
    
    # get indices of vertices of surface_orig that match surface_cut
    indices = get_indices(surface_orig, surface_cut)
    
    # reshape emodes to match vertices of original surface
    emodes_reshaped = np.zeros([surface_orig.n_points,np.shape(emodes)[1]])
    for mode in range(np.shape(emodes)[1]):
        emodes_reshaped[indices,mode] = np.expand_dims(emodes[:,mode], axis=1);      

    return evals, emodes_reshaped

def cortical_mask(subject, subjects_dir, src, downsample=True):
    """ Create a mask of the cortical regions for a given subject using Freesurfer's annotations within MNE-Python.

    Parameters
    ----------
    subject : str
        The subject ID to use.
    subjects_dir : str
        The path to the subjects directory.
    src : MNE SourceSpaces
        The source space of the subject.
    downsample : bool
        If True, return a  sparse mask corresponding to the downsampled source space.  If False, return a dense mask corresponding to the full dense MRI surface.
    
    Returns
    -------
    mask_list:  list
        A list of two arrays, one for each hemisphere, with 1s in the cortical regions and 0s elsewhere.
    """

    # By Default DK Atlas shouldn't have non-cortical regions
    parcellation = 'aparc'

    mask_list = []

    for i, hemi in enumerate(['lh', 'rh']):
        selected_label = mne.read_labels_from_annot(subject, subjects_dir=subjects_dir, parc=parcellation, hemi=hemi)

        # Cortical_label = 1 starts as a dummy boolean
        # Obtain all the cortical labels
        cortical_label = 1
        for label in selected_label:
            if cortical_label == 1:
                if 'unknown' not in label.name and '?' not in label.name:
                    cortical_label = label
            elif 'unknown' not in label.name and '?' not in label.name:
                cortical_label += label

        # Turn the cortical labels into a mask of 0s and 1s

        # Create an array of zeros with the desired shape
        num_vertices = src[i]['np']
        array = np.zeros((num_vertices,))

        # Set the elements in cortical_label.vertices to 1
        array[cortical_label.vertices] = 1

        if downsample:
            # Downsample the mask to the downsampled source space
            array = array[src[i]['vertno']]

        mask_list.append(array)

    return mask_list

def both_hemi_calc_eig(src, downsample=True, meshlist=None, mask=None, num_modes=200, fixsign=True, normalise=True):
    """Calculate the eigenvalues and eigenmodes of a surface for both hemispheres.

    Parameters:
    -----------
    src: mne source space
        The MNE source space if provided to obtain the points and triangles.  If None, the code will expect that the points and triangles are provided in the meshlist parameter.
    downsample: bool
        If True, the points and triangles are extracted from the MNE source space (which should be downsampled).  If False, the points and triangles are extracted from the MRI dense surface.
    meshlist: list
        List of four arrays. The first two arrays are the points and triangles of the left hemisphere, and the last two arrays are the points and triangles of the right hemisphere.
    num_modes : int
        Number of eigenmodes to be calculated in total for both hemispheres.  Therefore the number of eigenmodes for each hemisphere will be num_modes//2.
    mask: array (n_vertices x 1)
        Mask of cortical vertices. 1 if vertex is in cortex, 0 otherwise.
    fixsign: bool
        Whether to fix the sign of the eigenvectors based on the first element.  Default is True.
    normalise: bool
        Whether to normalise the eigenmodes to unit norm.  The solver should be roughly normalised already, but may be slightly biased towards lower modes.
        
    Returns:
    --------
    evals : array (num_modes x 1)
        Eigenvalues
    emodes : array (number of surface points x num_modes)
        Eigenmodes
    """

    # If source space is provided, obtain the points and triangles from the source space
    if src:
        l_points, l_trias = get_tria_from_src(src, hemi=0, downsample=downsample)
        r_points, r_trias = get_tria_from_src(src, hemi=1, downsample=downsample)
    # Otherwise, assume the points and triangles are provided in the meshlist
    elif src is None and meshlist:
        l_points = meshlist[0]
        l_trias = meshlist[1]
        r_points = meshlist[2]
        r_trias = meshlist[3]
    else:
        raise ValueError('Either src or meshlist (a list of vertices and triangles) must be provided.')

    # If no mask provided:
    if mask is None:
        # Construct for left hemisphere:
        l_tria = lapy.TriaMesh(l_points, l_trias)
        # Construct eigenmodes
        l_evals, l_emodes = calc_eig(l_tria, num_modes//2) 
        
        # Construct for right hemisphere: 
        r_tria = lapy.TriaMesh(r_points, r_trias)
        # Construct eigenmodes
        r_evals, r_emodes = calc_eig(r_tria, num_modes//2) 

    else:
        # Setting the vertices and triangles based on the src
        l_evals, l_emodes = calc_surface_eigenmodes(l_points, l_trias, mask=mask[0], num_modes=num_modes//2, )
        r_evals, r_emodes = calc_surface_eigenmodes(r_points, r_trias, mask=mask[1], num_modes=num_modes//2, )

    if fixsign:
        # The eigenmodes sometimes have a sign ambiguity, so we need to make sure they are consistent
        # We can do this by checking the sign of the first element of the eigenvector
        for i in range(num_modes//2):
            if l_emodes[0, i] < 0:
                l_emodes[:, i] = -l_emodes[:, i]
            if r_emodes[0, i] < 0:
                r_emodes[:, i] = -r_emodes[:, i]

    # Concatenate the eigenvalues and eigenmodes for both hemispheres
    evals = np.concatenate((l_evals,r_evals), axis=0)
    emodes = np.zeros((l_emodes.shape[0]+r_emodes.shape[0],l_emodes.shape[1]+r_emodes.shape[1]))
    emodes[:l_emodes.shape[0],:l_emodes.shape[1]] = l_emodes
    emodes[l_emodes.shape[0]:,l_emodes.shape[1]:] = r_emodes
    # This is a 2 by 2 of modes, alternatively could've stacked them.
    # emodes = np.concatenate((l_emodes,r_emodes), axis=0)

    if normalise:
        # Normalise the eigenmodes to unit norm
        emodes = emodes / np.linalg.norm(emodes, axis=0)[np.newaxis, :]

    return evals, emodes

def subset_eigenmodes(emodes, src, downsample=True, method='average', normalise=True, **kwargs):
    """Subset the eigenmodes to only include specific ones in the regions of interest.  Confirmed to work for a set containing both hemispheres.
    The regions of interest are defined by the method, which then dictate extra arguments required.  

    Parameters
    ----------
    emodes : np.ndarray
        The eigenmodes to subset.
    src : list
        The original source space which the eigenmodes were generated on. 
    downsample : bool
        Whether the mesh used to calculate the original eigenmodes was the downsampled src (True) or the dense MRI mesh (False).
    method : str
        The method to use to subset the eigenmodes and assign the values. Can be 'average' or 'nearest'.
        'average' signifies averaging across all vertices in the src for the region of interest.  Depending on 'downsample', it will use the downsampled src or the dense MRI mesh.
        'nearest' signifies finding the nearest vertex in the src mesh for each vertex in the subset src.
    normalise : bool
        Whether to normalise the subsetted eigenmodes to unit norm.  Not doing so generally weights the lower modes more slightly. 
    kwargs : dict
        The extra arguments required for the method. 
        'average' requires 'parcellation', 'subjects_dir', 'region_labels', and 'subject'.
            'region_labels' as taken from con.region_labels. 
        'nearest' requires 'subsetsrc'.
            'subsetsrc' is the source space used to find the nearest vertex.

    Returns
    -------
    subset_emodes : np.ndarray
        The subset eigenmodes for the regions of interest. 
    """

    # Grab necessary parameters from kwargs
    if method == 'average':
        parcellation = kwargs['parcellation']
        subjects_dir = kwargs['subjects_dir']
        region_labels = kwargs['region_labels']
        subject = kwargs['subject']
    elif method == 'nearest':
        subsetsrc = kwargs['subsetsrc']

    if method == 'nearest':
        # Find the nearest vertex in the higher space mesh for each vertex in the subset src - Downsample signifies dense MRI vs inputted downsample source space
        l_points, _ = get_tria_from_src(src, hemi=0, downsample=downsample)
        r_points, _ = get_tria_from_src(src, hemi=1, downsample=downsample)
        indices_lh, _ = find_each_nearest_vertex(subsetsrc[0]['rr'][subsetsrc[0]['inuse']], l_points)
        indices_rh, _ = find_each_nearest_vertex(subsetsrc[1]['rr'][subsetsrc[1]['inuse']], r_points)
        indices = np.concatenate((indices_lh, len(l_points)+indices_rh), )        
        subset_emodes = emodes[indices, :]

    elif method == 'average':
        # Number of vertices in left hemisphere
        if downsample:
            # Downsampled source space
            l_points, _ = get_tria_from_src(src, hemi=0, downsample=downsample)
            nv_lh = len(l_points)
        else:
            # Dense MRI vertices
            nv_lh = src[0]['np']

        # Get labels from MNE for each region 
        labels = mne.read_labels_from_annot(subject, parcellation, subjects_dir=subjects_dir)
        # Remove the ??? regions
        labels = [label for label in labels if '???' not in label.name]
        # Get the label names for each region
        label_names = np.array([label.name[:-3] for label in labels])

        # Find the reordering to match region_labels with label_names
        # Ignore subcortical structures from the region_names in the connectivity file
        if parcellation == 'HCPMMP1':
            # Glasser
            region_labels =  region_labels[19:]
        elif parcellation == 'native.1000Parcels_Yeo2011_7Networks':
            # Yeo
            region_labels =  region_labels[:-19]
        reorder_indices = np.array([np.where(label_names == region)[0][0] for region in region_labels])
        # Reorder the labels to match the region_labels
        labels = [labels[i] for i in reorder_indices]
        label_names = label_names[reorder_indices]

        # Initialise the array of eigenmodes
        subset_emodes = np.zeros((len(labels), emodes.shape[1]))

        # Average the eigenmodes over the vertices in each label
        for count, label in enumerate(labels):
            # Identify hemisphere
            if label.hemi == 'lh':
                hemi = 0
            elif label.hemi == 'rh':
                hemi = 1

            # Subset based on if the original eigenmodes were computed on dense MRI mesh or downsampled source space
            if downsample:
                # Need to use the src
                label_vertices = np.where(np.isin(src[hemi]['vertno'], label.vertices))[0] + hemi*nv_lh
            else:
                # Use the vertices in the label which already correspond to the dense MRI mesh
                label_vertices = label.vertices + hemi*nv_lh

            # Assign the subset eigenmodes as required
            subset_emodes[count, :] = np.mean(emodes[label_vertices,:], axis=0)
        
    if normalise:
        # Normalise the subsetted eigenmodes to unit norm
        subset_emodes = subset_emodes / np.linalg.norm(subset_emodes, axis=0)[np.newaxis, :]

    return subset_emodes

def gem_solver(M, G, n_orient, emodes, rcond=None, N=None):
    """ Solve the inverse problem using the structural eigenmodes method.  

    Parameters
    ----------
    M : array, shape (n_channels, n_times)
        The whitened data.
    G : array, shape (n_channels, n_dipoles)
        The gain matrix a.k.a. the forward operator. The number of locations
        is n_dipoles / n_orient. n_orient will be 1 for a fixed orientation
        constraint or 3 when using a free orientation model.
    n_orient : int
        Can be 1 or 3 depending if one works with fixed or free orientations.
        If n_orient is 3, then ``G[:, 2::3]`` corresponds to the dipoles that
        are normal to the cortex.
    emodes: array, shape (n_sourcelocations, n_modes)
        The eigenmodes of the surface.
    rcond: float
        The cutoff ratio for small singular values when calculating the SVD for the pseudo-inverse.  Default is None.
    N : int
        The N number of strongest dipoles to select.  If None, all dipoles are used.

    Returns
    -------
    X : array, (n_active_dipoles, n_times)
        The time series of the dipoles in the active set.
    active_set : array (n_dipoles)
        Array of bool. Entry j is True if dipole j is in the active set.
        We have ``X_full[active_set] == X`` where X_full is the full X matrix
        such that ``M = G X_full``.
    """

    # Repeat the eigenmodes for each orientation
    emodes = np.repeat(emodes, n_orient, axis=0)    
    # Multiply gain matrix with eigenmodes
    mat = G @ emodes

    # Just grab the least squares solution - numpy returns the smallest 2-norm solution if it is underdetermined
    w, res, rank, s = np.linalg.lstsq(mat, M, rcond=rcond)
    # Log the residual
    var_exp = log_exp_var(M, np.sqrt(res))
    # Now return in source space
    X = emodes @ w

    # If N is not None, select the N strongest dipoles
    if N:
        indices = np.argsort(np.sum(X**2, axis=1))[-N:]
        active_set = np.zeros(G.shape[1], dtype=bool)
        for idx in indices:
            idx -= idx % n_orient
            active_set[idx : idx + n_orient] = True
        X = X[active_set]
    else:
        active_set = np.ones(G.shape[1], dtype=bool)

    return X, active_set, var_exp, w

def gem_precompute_svd(G, emodes, n_orient=1, rcond=None, trunc_emodes=False):
    """ Precompute the SVD decomposition of the gain matrix and eigenmodes.

    Parameters
    ----------
    G : array, shape (n_channels, n_dipoles)
        The gain matrix a.k.a. the forward operator. The number of locations
        is n_dipoles / n_orient. n_orient will be 1 for a fixed orientation
        constraint or 3 when using a free orientation model.
    emodes: array, shape (n_sourcelocations, n_modes)
        The eigenmodes of the surface.
    n_orient : int
        Can be 1 or 3 depending if one works with fixed or free orientations.
        If n_orient is 3, then ``G[:, 2::3]`` corresponds to the dipoles that
        are normal to the cortex.
    rcond: float
        The cutoff ratio for small singular values when calculating the SVD for the pseudo-inverse.  Default is None.
    trunc_emodes: bool
        If True, the number of eigenmodes are truncated to the number of singular values larger than rcond.
        This means that instead of truncating orthogonal components from SVD, we directly truncate the set of EEG modes.
        Default is False.  Probably needs some debugging if used in conjunction with solver as need to output the truncated emodes.

    Returns
    -------
    V : array, shape (n_modes, n_modes)
        The left singular vectors of the gain matrix and eigenmodes.
    S_inv : array, shape (n_modes)
        The inverse of the singular values of the gain matrix and eigenmodes.
    Uh : array, shape (n_modes, n_modes)
        The right singular vectors of the gain matrix and eigenmodes.
    """

    def closest_2n_squared(n_singular_values):
        # Calculate the closest integer n
        n = int(np.sqrt(n_singular_values / 2))
        # Compute the two closest values of the form 2 * n**2
        lower = 2 * n**2
        upper = 2 * (n + 1)**2
        # Return the closest value
        return lower if abs(lower - n_singular_values) <= abs(upper - n_singular_values) else upper

    # Check lower-left quadrant of emodes - to determine if eigenmodes sorted by hemisphere
    if np.sum(emodes[emodes.shape[0]//2:,:emodes.shape[1]//2]) == 0:
        hemi_sort = True
    else:  
        hemi_sort = False

    # If rcond == None, set to machine epsilon as default for np.linalg.lstsq
    if rcond == None:
        rcond = np.finfo(np.float64).eps

    # Repeat the eigenmodes for each orientation
    emodes = np.repeat(emodes, n_orient, axis=0)
    # Multiply gain matrix with eigenmodes
    mat = G @ emodes

    if trunc_emodes:
        # Perform SVD on the EEG modes
        _, S, _ = np.linalg.svd(mat, full_matrices=False)
        # Identify the number of singular values larger than rcond
        n_singular_values = np.sum(S/S.max() > rcond)
        if n_singular_values < 32:
            # If the number of singular values is less than 32, set it to 32
            n_singular_values = 32
        else:
            # Find the closest number that satisfied 2*n**2
            n_singular_values = closest_2n_squared(n_singular_values)

        # Truncate the eigenmodes to the number of singular values larger than rcond based on ordering of eigenmodes
        if hemi_sort:
            # If the eigenmodes are sorted by hemisphere, we need to truncate them accordingly
            emodes=np.concatenate((emodes[:, :n_singular_values//2], emodes[:, emodes.shape[1]//2:emodes.shape[1]//2+n_singular_values//2]), axis=1)
            mat =  G @emodes
        else:
            # If not sorted by hemisphere, it is sorted by eigenvalue and is simple. 
            mat = mat[:, :n_singular_values]
            emodes = emodes[:, :n_singular_values]

    # Perform SVD on the truncated EEG modes
    U, S, Vh = np.linalg.svd(mat, full_matrices=False)
    # Remove small singular values
    S_inv = np.zeros_like(S)
    S_inv[S/S.max() > rcond] = 1 / S[S/S.max() > rcond]
    V = Vh.T
    Uh = U.T

    return V, S_inv, Uh

def gem_solver_svd(M, G, n_orient, emodes, svd_decomp, N=None):
    """ Solve the inverse problem using the structural eigenmodes method.  
    This version relies on precomputing the SVD outside of the function and using the decomposition as inputs to solve the problem more efficiently.
    Needs some checking, as it appears slightly different to the original solver and slightly less stable.

    Parameters
    ----------
    M : array, shape (n_channels, n_times)
        The whitened data.
    G : array, shape (n_channels, n_dipoles)
        The gain matrix a.k.a. the forward operator. The number of locations
        is n_dipoles / n_orient. n_orient will be 1 for a fixed orientation
        constraint or 3 when using a free orientation model.
        Necessary input for MNE-Python's apply_solver function.
    n_orient : int
        Can be 1 or 3 depending if one works with fixed or free orientations.
        If n_orient is 3, then ``G[:, 2::3]`` corresponds to the dipoles that
        are normal to the cortex.
    svd_decomp: tuple
        The SVD decomposition of the gain matrix and eigenmodes.  This is the output of gem_precompute_svd.
    N : int
        The N number of strongest dipoles to select.  If None, all dipoles are used.

    Returns
    -------
    X : array, (n_active_dipoles, n_times)
        The time series of the dipoles in the active set.
    active_set : array (n_dipoles)
        Array of bool. Entry j is True if dipole j is in the active set.
        We have ``X_full[active_set] == X`` where X_full is the full X matrix
        such that ``M = G X_full``.
    """

    # Unpack the SVD decomposition
    V, S_inv, Uh = svd_decomp

    # Multiply gain matrix with eigenmodes
    mat = G @ emodes

    # The weight matrix is
    w = V @ np.diag(S_inv) @ Uh @ M
    # Now return in source space
    X = emodes @ w
    # Calculate the residual in sensor space
    res = np.linalg.norm(M - mat @ w, axis=0)
    var_exp = log_exp_var(M, res)

    # If N is not None, select the N strongest dipoles
    if N:
        indices = np.argsort(np.sum(X**2, axis=1))[-N:]
        active_set = np.zeros(G.shape[1], dtype=bool)
        for idx in indices:
            idx -= idx % n_orient
            active_set[idx : idx + n_orient] = True
        X = X[active_set]
    else:
        active_set = np.ones(G.shape[1], dtype=bool)

    return X, active_set, var_exp, w

def gem_solver_regularisation(M, G, n_orient, emodes, alpha, reg_mat, N=None):
    """ Solve the inverse problem using the structural eigenmodes method via regularisation.

    Parameters
    ----------
    M : array, shape (n_channels, n_times)
        The whitened data.
    G : array, shape (n_channels, n_dipoles)
        The gain matrix a.k.a. the forward operator. The number of locations
        is n_dipoles / n_orient. n_orient will be 1 for a fixed orientation
        constraint or 3 when using a free orientation model.
    n_orient : int
        Can be 1 or 3 depending if one works with fixed or free orientations.
        If n_orient is 3, then ``G[:, 2::3]`` corresponds to the dipoles that
        are normal to the cortex.
    emodes: array, shape (n_sourcelocations, n_modes)
        The eigenmodes of the surface.
    alpha: float
        The regularisation parameter.  This is also referred to as lambda in regularisation literature.
    reg_mat: array, shape (n_modes, n_modes)
        The regularisation matrix of choice.  An example is the identity matrix for Tikhonov regularisation.
    N : int
        The N number of strongest dipoles to select.  If None, all dipoles are used.

    Returns
    -------
    X : array, (n_active_dipoles, n_times)
        The time series of the dipoles in the active set.
    active_set : array (n_dipoles)
        Array of bool. Entry j is True if dipole j is in the active set.
        We have ``X_full[active_set] == X`` where X_full is the full X matrix
        such that ``M = G X_full``.
    """

    # Repeat the eigenmodes for each orientation
    emodes = np.repeat(emodes, n_orient, axis=0)
    # Multiply gain matrix with eigenmodes
    mat = G @ emodes
    # Define estimator
    w = np.linalg.inv((mat.T @ mat + alpha*reg_mat)) @ mat.T @ M
    # Calculate the residual in sensor space
    res = np.linalg.norm(M - mat @ w, axis=0)
    var_exp = log_exp_var(M, res)
    # Now return in source space
    X = emodes @ w

    # If N is not None, select the N strongest dipoles
    if N:
        indices = np.argsort(np.sum(X**2, axis=1))[-N:]
        active_set = np.zeros(G.shape[1], dtype=bool)
        for idx in indices:
            idx -= idx % n_orient
            active_set[idx : idx + n_orient] = True
        X = X[active_set]
    else:
        active_set = np.ones(G.shape[1], dtype=bool)

    return X, active_set, var_exp, w

def gem_sparse_solver(M, G, n_orient, emodes, alpha=1e-7, max_iter=1000, N=None):
    """ Solve the inverse problem using the structural eigenmodes method with L1 regularization for sparsity.  

    Parameters
    ----------
    M : array, shape (n_channels, n_times)
        The whitened data.
    G : array, shape (n_channels, n_dipoles)
        The gain matrix a.k.a. the forward operator. The number of locations
        is n_dipoles / n_orient. n_orient will be 1 for a fixed orientation
        constraint or 3 when using a free orientation model.
    n_orient : int
        Can be 1 or 3 depending if one works with fixed or free orientations.
        If n_orient is 3, then ``G[:, 2::3]`` corresponds to the dipoles that
        are normal to the cortex.
    emodes: array, shape (n_sourcelocations, n_modes)
        The eigenmodes of the surface.
    alpha: float
        The L1 regularization parameter that controls sparsity. Higher values = more sparsity.
    max_iter: int
        The maximum number of iterations for the Lasso solver.
    N : int
        The N number of strongest dipoles to select.  If None, all dipoles are used.

    Returns
    -------
    X : array, (n_active_dipoles, n_times)
        The time series of the dipoles in the active set.
    active_set : array (n_dipoles)
        Array of bool. Entry j is True if dipole j is in the active set.
        We have ``X_full[active_set] == X`` where X_full is the full X matrix
        such that ``M = G X_full``.
    """
    from sklearn.linear_model import Lasso

    # Repeat the eigenmodes for each orientation
    emodes = np.repeat(emodes, n_orient, axis=0)    
    # Multiply gain matrix with eigenmodes
    mat = G @ emodes

    # Use Lasso regression for L1 regularization to promote sparsity
    lasso = Lasso(alpha=alpha, fit_intercept=False, max_iter=max_iter)
    w = np.zeros((mat.shape[1], M.shape[1]))
    for t in range(M.shape[1]):
        w[:,t] = lasso.fit(mat, M[:,t]).coef_

    # Calculate residual and explained variance
    res = np.linalg.norm(M - mat @ w, axis=0)
    var_exp = log_exp_var(M, res)
    
    # Transform back to source space
    X = emodes @ w

    # If N is not None, select the N strongest dipoles
    if N:
        indices = np.argsort(np.sum(X**2, axis=1))[-N:]
        active_set = np.zeros(G.shape[1], dtype=bool)
        for idx in indices:
            idx -= idx % n_orient
            active_set[idx : idx + n_orient] = True
        X = X[active_set]
    else:
        active_set = np.ones(G.shape[1], dtype=bool)

    return X, active_set, var_exp, w

def rotate_matrix(M, method='indirect', seed=None):
    """
    Code implementation courtesy of: https://github.com/SNG-Newy/eigenstrapping/tree/main. 
    Rotate an (n/m)-by-n matrix of arbitrary length n by the two methods 
    as outlined in [1].
    
    Parameters
    ----------
    M : 2D np.ndarray
        Input matrix
    method : str, optional
        Which method to use. Refers to the nomenclature in [1], where
        'indirect' refers to the Householder QR decomposition method [2],
        while 'direct' refers to the method of selecting random points
        on the unit n-sphere directly as in [3]. The default is 'indirect'.

    Returns
    -------
    X_rotated : TYPE
        DESCRIPTION.
        
    References
    ----------
    [1] Blaser R, Fryzlewicz P. Random Rotation Ensembles. Journal of 
        Machine Learning Research. 2016;17(4):1-26.
    [2] Householder A. S. Unitary triangularization of a nonsymmetric 
        matrix. Journal of the ACM, 5:339–342, 1958.
    [3] Knuth D. E. Art of computer programming, volume 2: 
        seminumerical algorithms. Addison-Wesley Professional, Reading, 
        Mass, 3 edition edition, November 1997.

    """
    n = M.shape[1]
    if method == 'indirect':
        rot = indirect_method(n, seed=seed)
        M_rotated = np.dot(M, rot)
    
    elif method == 'direct':
        rot = direct_method(n)
        M_rotated = np.dot(M, rot)
    
    else:
        raise ValueError("Method must be one of 'indirect' or 'direct'")
        
    return M_rotated

def direct_method(n, seed=None):
    """ Code implementation courtesy of: https://github.com/SNG-Newy/eigenstrapping/tree/main. """
    from sklearn.utils.validation import check_random_state
    from numpy.linalg import qr

    rs = check_random_state(seed)
    # 1. Draw n independent random normal N(0, 1) variates
    v = rs.normal(size=n)
    
    # 2. Normalize
    x = v / np.linalg.norm(v)
    
    # 3. Treat the vector as a single column matrix
    X = x[:, np.newaxis]
    
    # 4. Apply QR decomposition
    Q, _ = qr(X)
    
    return Q

def indirect_method(n, seed=None):
    """ Code implementation courtesy of: https://github.com/SNG-Newy/eigenstrapping/tree/main. """
    from sklearn.utils.validation import check_random_state
    from scipy.stats import special_ortho_group

    rs = check_random_state(seed)
    
    # Compute the QR decomposition
    if n < 2:
        return rs.normal(size=(n, n))
    rotate = special_ortho_group.rvs(dim=n, random_state=rs)
    
    return rotate

def random_large_rotation(N, c=0.7, seed=None):
    """ Is the indirect_method of https://github.com/SNG-Newy/eigenstrapping/tree/main with a threshold for rejection."""
    from sklearn.utils.validation import check_random_state
    from scipy.stats import special_ortho_group

    rs = check_random_state(seed)
    threshold = c * np.sqrt(2 * N)

    while True:
        R = special_ortho_group.rvs(dim=N, random_state=rs)
        distance = np.linalg.norm(R - np.eye(N), ord='fro')
        if distance > threshold:
            return R

def gen_surrogate_eigenmodes(emodes, evals, thresh=0, normalize=True, seed=None):
    """
    Generate surrogate eigenmodes by rotating the original eigenmodes.
    Code implementation courtesy of: https://github.com/SNG-Newy/eigenstrapping/tree/main. 
    
    Parameters
    ----------
    emodes : np.ndarray
        The eigenmodes to be rotated.
    evals : np.ndarray
        The eigenvalues corresponding to the eigenmodes.
    thresh : float, optional
        The threshold constant for rejecting rotations too close to the identity. Default is 0 to accept all rotations.
    normalize : bool, optional
        Whether to normalize the eigenmodes. Default is True.
    seed : int, optional
        Random seed for reproducibility. Default is None.
            
    Returns
    -------
    new_modes : np.ndarray
        The rotated eigenmodes.
    """

    def _get_eigengroups(eigs, suppress_message=False):
        """
        Helper function to find eigengroups
        """
        if suppress_message is False:
            print("IMPORTANT: EIGENMODES MUST BE TRUNCATED AT FIRST NON-ZERO MODE FOR THIS FUNCTION TO WORK")
        lam = eigs.shape[1] # number of eigenmodes, accounting for discarded non-zero mode
        l = np.floor((lam-1)/2).astype(int)    
        # Handle cases where lam is from 4 to 7
        if lam == 3:
            return [np.arange(0, 3)]
        elif 4 <= lam < 8:
            return [np.arange(0, 3), np.arange(3, lam)]
        elif lam < 3:
            raise ValueError('Number of modes to resample cannot be less than 3')
        
        groups = []#[np.array([0])]
        ii = 0
        i = 0
        for g in range(1, l):
            ii += 2*g+1
            if ii >= lam:
                groups.append(np.arange(i,lam))
                return groups
            groups.append(np.arange(i,ii))
            i = ii

    def compute_axes_ellipsoid(eigenvalues):
        """
        Compute the axes of an ellipsoid given the eigenmodes.
        """    
        return np.sqrt(eigenvalues)
        

    def transform_to_spheroid(eigenvalues, eigenmodes):
        """
        Transform the eigenmodes to a spheroid space
        """
        ellipsoid_axes = compute_axes_ellipsoid(eigenvalues)
        spheroid_eigenmodes = np.divide(eigenmodes, np.sqrt(eigenvalues))
        
        return spheroid_eigenmodes
        
        
    def transform_to_ellipsoid(eigenvalues, eigenmodes):
        """
        Transform the eigenmodes in spheroid space back to ellipsoid by stretching
        """        
        ellipsoid_axes = compute_axes_ellipsoid(eigenvalues)        
        ellipsoid_eigenmodes = np.multiply(eigenmodes, np.sqrt(eigenvalues))
        
        return ellipsoid_eigenmodes

    groups = _get_eigengroups(emodes, suppress_message=False)

    # initialize the new modes
    new_modes = np.zeros_like(emodes)
    # resample the hypersphere (except for groups 1 and 2)
    for idx in range(len(groups)):
        group_modes = emodes[:, groups[idx]]
        group_evals = evals[groups[idx]]
        p = group_modes
        # else, transform to spheroid and index the angles properly
        if normalize:
            p = transform_to_spheroid(group_evals, group_modes)

        # p_rot = rotate_matrix(p, method='indirect', seed=seed)
        rot = random_large_rotation(p.shape[1], c=thresh, seed=seed)
        p_rot = np.dot(p, rot)
        
        # transform back to ellipsoid
        if normalize:
            p_rot = transform_to_ellipsoid(group_evals, p_rot)
        
        new_modes[:, groups[idx]] = p_rot

    # Add the 0th eigenmode
    new_modes = np.concatenate((np.ones((new_modes.shape[0], 1)), new_modes), axis=1)

    return new_modes

def both_hemi_surrogate_eig(emodes, evals, thresh=0, normalize=True, seed=None):
    """
    Generate surrogate eigenmodes for both hemispheres by splitting the original eigenmodes into left and right hemispheres, 
    generating surrogate eigenmodes for each hemisphere through the rotational eigenstrapping method, and then combining them.
    Uses functions from Koussis et. al 2024 (https://github.com/SNG-Newy/eigenstrapping/tree/main) with minor adjustments. 

    Parameters
    ----------
    emodes : np.ndarray
        The original eigenmodes to be split and rotated.  
        The 0th eigenmode needs to be truncated for a later function which this function handles for you automatically.
    evals : np.ndarray
        The eigenvalues corresponding to the original eigenmodes.
    normalize : bool, optional
        Whether to normalize the eigenmodes. Default is True.
    thresh : float, optional
        The threshold constant for rejecting rotations too close to the identity. Default is 0 to accept all rotations.
    seed : int, optional
        Random seed for reproducibility. Default is None.
        
    Returns
    -------
    surrogate_emodes : np.ndarray
        The combined surrogate eigenmodes for both hemispheres.    

    """

    emodes_lh = emodes[:emodes.shape[0]//2,1:emodes.shape[1]//2]    
    evals_lh = evals[1:len(evals)//2]
    new_lh = gen_surrogate_eigenmodes(emodes_lh, evals_lh, seed=seed, normalize=normalize, thresh=thresh)

    emodes_rh = emodes[emodes.shape[0]//2:,emodes.shape[1]//2+1:]
    evals_rh = evals[len(evals)//2+1:]
    new_rh = gen_surrogate_eigenmodes(emodes_rh, evals_rh, seed=seed, normalize=normalize, thresh=thresh)

    surrogate_emodes = np.zeros((new_lh .shape[0]+new_rh.shape[0],new_lh .shape[1]+new_rh.shape[1]))
    surrogate_emodes[:new_lh .shape[0],:new_lh .shape[1]] = new_lh 
    surrogate_emodes[new_lh .shape[0]:,new_lh .shape[1]:] = new_rh

    return surrogate_emodes

def calc_con_eigenmodes(con, num_modes, threshold=0, normed=True, binarise=True, fixsign=True): 
    """ Generate graph Laplacian Eigenmodes from a connectivity matrix.  
    
    Parameters
    ----------
    con : ndarray
        Connectivity matrix consisting of weights / streamlines between regions.        
    num_modes : int
        Number of eigenmodes to compute.
    threshold : float
        Threshold for the connectivity matrix.  Anything below absolute value of threshold is set to 0.  Default is 0.
    normed : bool
        Whether to normalise the graph Laplacian.  Default is True.
    binarise : bool
        Whether to binarise the connectivity matrix.  Default is True.
    fixsign : bool
        Whether to fix the sign of the eigenvectors based on the first element.  Default is True.

    Returns
    -------
    evals : ndarray
        Eigenvalues of the graph Laplacian.
    evecs : ndarray
        Eigenvectors of the graph Laplacian.
    """

    if binarise:
        con = (con > threshold).astype(float)
    else:
        con[con<=threshold] = 0 

    # Generate Graph Laplacian
    glap = scipy.sparse.csgraph.laplacian(con, normed=normed, form='array')
    # Generate Laplacian Eigenvectors
    evals, evecs = scipy.sparse.linalg.eigsh(glap, k=num_modes, which='SM', return_eigenvectors=True)

    if fixsign:
        # The eigenmodes sometimes have a sign ambiguity, so we need to make sure they are consistent
        # We can do this by checking the sign of the first element of the eigenvector
        for i in np.arange(evecs.shape[1]):
            if evecs[0, i] < 0:
                evecs[:,i] *= -1

    return evals, evecs