import numpy as np
from pathlib import Path
import csv
from functools import partial
import concurrent.futures

from scipy import stats
from scipy.signal import find_peaks
from scipy.optimize import fsolve

from tvb.simulator.lab import *
import mne

from useful_fns import *

def params_dict_writer(params_dict, file_quality, PATH):
    """ Function which takes in params_dict and strings and turns it into a csv for storing.  Consider switching to JSON or dif delimter.  Can't use commas. 
    
    Parameters:
    -----------
    params_dict: dict
        Dictionary of parameters to be saved.
    file_quality: str
        String which is appended to the file name as discriptors of the file. 
    PATH: Path
        Path object which tells the function where to save the file.

    Returns:
    --------
    None    
    """
    
    with open(PATH / Path('Params_' + file_quality + '.csv'), "w") as outfile:
        writer = csv.writer(outfile)
        for key, val in sorted(params_dict.items()):
            writer.writerow([key, val])   

def params_dict_reader(file_path):
    """ Function to read the CSV file created from params_dict_writer() and convert it back into a dictionary.
    
    Parameters:
    -----------
    file_path: str
        Path to the file to be read.    
    """

    params_dict = {}

    with open(file_path, "r") as infile:
        reader = csv.reader(infile)
        for row in reader:
            if len(row) == 2:
                key, val = row
                params_dict[key] = val

    return params_dict

def calc_SCFC(raw_x,con,snip=0):
    """
    Function which calculates the SCFC score.  This provides a sense of how well the simulation is doing.
    The inputs are the raw_x time series, the connectivity matrix, and the snip value which is the number of time points to snip off the beginning of the simulation.

    Parameters:
    -----------
    raw_x: 4D array
        Time series data from the TVB simulation.  Dimensions are (time, state variables , nodes, -)
    con: TVB Connectivity
        TVB Connectivity object
    snip: int
        Number which defines how many time points to snip off the beginning of the simulation.  Default is 0.

    Returns:
    --------
    Scorr: float
        Spearman correlation between the SCM and FCM.  This is the SCFC score.
    """

    # Snip off the first N time points
    TSeriesMatrix = raw_x[snip:,0,:,0].T
    FCM = np.corrcoef(TSeriesMatrix)

    FCM_Upper = FCM[np.triu_indices(FCM.shape[0], k = 1)]
    SCM_Upper = con.weights[np.triu_indices(con.weights.shape[0], k = 1)]

    # Comparing SC vs FC with Spearman Corr  (Known as SCFC)
    SCM = con.weights

    # Check if SCM is symmetric: 
    Sym_check = np.allclose(SCM, SCM.T,equal_nan=True)
    if Sym_check == True:
        #It is a symmetric SCM, so only use upper triangles
        # Grab Upper triangles
        FCM_Upper = FCM[np.triu_indices(FCM.shape[0], k = 1)]
        SCM_Upper = con.weights[np.triu_indices(con.weights.shape[0], k = 1)]

    # Spearman Correlation
    Scorr = stats.spearmanr(a=FCM_Upper,b=SCM_Upper)
    
    return Scorr

def con_loader(path, load_tracts=True, threshold=None, weights_only=False, load_centres=False, zerodiag=True, subject='fsaverage', subjects_dir=None):
    """
    This function is for creating a connectome for TVB. It intakes a folder path containing the relevant files: streamline_count_atlas.csv, region_names.txt, streamline_lengths_mean.csv.
    Outputs a TVB Connectivity() object.  Load_tracts can be optionally set to False to not load the streamline_lengths_mean.csv file, which sets all tract lengths to 0.
    If weights_only is set to True, then it takes the file path of the weights file only and assumes load_trats = False. This is useful for when you only have the weights file.

    Parameters:
    -----------
    path: Path
        Path to the folder containing the files.
    load_tracts: bool
        Whether to load the streamline_lengths_mean.csv file.  Default is True. If false, all tract lengths are set to 0.
    threshold: float
        Threshold to apply to the weights.  Default is None.
    weights_only: bool
        Whether to only load the weights file.  
    load_centres: bool
        Whether to load the region centres using centre_loader and fsaverage defaults.  Default is False to call centre_loader and assign centres externally. 
    zerodiag: bool
        Whether to set the diagonal of the connectivity matrix to 0.  Default is True to stop strong recursive effects in simulation.
    subject: str
        The subject to load the centres of mass for, only relevant if load_centres is True.  Default is 'fsaverage'.
    subjects_dir: str
        The path to the subjects directory, only relevant if load_centres is True.  Default is None.  
    
    Returns:
    --------
    con: TVB Connectivity
        TVB Connectivity object.
    """
    con = connectivity.Connectivity()

    if weights_only:
        # con.weights = sparse.load_npz(path).toarray()
        con.weights = np.loadtxt(path, delimiter=',')
        load_tracts = False
        con.region_labels = np.arange(con.weights.shape[0])
    
    else:    
        con.weights = np.loadtxt(path / Path('streamline_count_atlas.csv'), delimiter=',')
        con.region_labels = np.loadtxt(path / Path('region_names.txt'), dtype=str)

    # Do not log transform the weights
    if zerodiag:
        # Modify the connectome such that the diagonal is 0 (no self connections from connectome) - This is required to stop very strong recursive effects causing it to go to infinity.
        np.fill_diagonal(con.weights, 0)
    # Normalise the weights
    con.weights = con.scaled_weights()
    # Option to apply a threshold to the weights such that any below this threshold is set to 0 - Can do this to make code run a bit faster.  But I wouldn't think this is a good idea
    if threshold:
        con.weights[con.weights<=threshold] = 0

    # Specify some metadata
    con.number_of_regions = con.weights.shape[0]
    # Need to define centre 
    # Generate a spherical one first - Noting that this will be different every run
    # con.centres_spherical(con.number_of_regions)
    # Actually, just use 0s, so there is no confusion that there is information on subcortical structures.
    con.centres = np.zeros((con.number_of_regions,3))
    # Replace cortical regions based on centres from the freesurfer parcellation if requested
    if load_centres:
        # Set all centres to np.nan - So that the subcortical structures are nans
        con.centres[:] = np.nan
        if con.number_of_regions == 68:
            parcellation = 'aparc' # DK atlas
        elif con.number_of_regions == 379:
            parcellation = 'HCPMMP1'
        elif con.number_of_regions == 1019:
            parcellation = 'native.1000Parcels_Yeo2011_7Networks'
        # Need to define centres for the regions - based on freesurfer parcellation
        cort_centres = centre_loader(con.region_labels, parcellation, subject=subject, subjects_dir=subjects_dir)
        # Assign centres to cortical regions depending on parcellation
        if parcellation == 'HCPMMP1':
            # Glasser
            con.centres[19:] = cort_centres
        elif parcellation == 'native.1000Parcels_Yeo2011_7Networks':
            # Yeo
            con.centres[:-19] = cort_centres
        elif parcellation == 'aparc':
            con.centres = cort_centres
        else:
            raise ValueError('Parcellation not recognised.')

    # Load tract lengths if requested 
    if load_tracts:
        con.tract_lengths = np.loadtxt(path / Path('streamline_lengths_mean.csv'), delimiter=',')
        # Adjust all nans to 0
        con.tract_lengths[np.isnan(con.tract_lengths)]=0
    else:
    # Set tract_lengths to 0
        con.tract_lengths = np.zeros((con.weights.shape[0],con.weights.shape[0]))

    # For whatever else TVB needs
    con.configure()

    return con

def get_label_normal(label, atcom=False, coords='MRI_RAS', subject='fsaverage', subjects_dir=None, surf_type='white', ):
    """
    Obtain the normal for the specified label.  This can be the average normal across all triangles in the label or the normal at the centre of mass of the label.
    For obtaining the normal for multiple labels, use the get_normals_parallel() function instead.

    Parameters
    ----------
    label : mne.Label
        The label for which to obtain the normal.
    atcom : bool
        Whether to obtain the normal at the centre of mass of the label.  False means that instead the normal is obtained at the average of all triangles in the label.  
    coords : str
        The coordinate system to use.  Can be MRI_RAS (Freesurfer's default output) or MNI.
    subject : str
        The subject for which to obtain the normal.
    subjects_dir : str
        The subjects directory.
    surf_type : str
        The surface Freesurfer file to read the data from.  Default is 'white'.  'sphere' is also a sensible option. 
    
    Returns
    -------
    label_normal : np.ndarray
        The normal at the centre of mass of the label.
    """
    
    hemi = label.hemi
    surface = mne.surface.read_surface(subjects_dir / subject / "surf" / f"{hemi}.{surf_type}")
    hemi_i = 0 if hemi == 'lh' else 1

    if atcom:
        # Obtain centre of mass
        com_vert = label.center_of_mass(subject=subject, restrict_vertices=False, surf=surf_type, subjects_dir=subjects_dir,)
        # Obtain triangles which contain centre of mass
        com_tri = surface[1][np.any(surface[1] == com_vert, axis=1)]
    else:
        # Obtain triangles which contain label vertices
        com_tri = surface[1][np.any(np.isin(surface[1], label.vertices), axis=1)]

    if coords == 'MNI':
        # Obtain vertex positions of triangles in MNI space
        com_vert_pos = mne.vertex_to_mni(com_tri, hemis=hemi_i, subject=subject, subjects_dir=subjects_dir)
        # Obtain normal at region centre - https://surfer.nmr.mgh.harvard.edu/fswiki/FreeSurferWiki/SurfaceNormal
        v0 =  com_vert_pos[:, 0, :]
        v1 =  com_vert_pos[:, 1, :]
        v2 =  com_vert_pos[:, 2, :]

    elif coords == 'MRI_RAS':
        # By default Freesurfer is in MRI_RAS space, and MNE-Python seems to be doing calculations in this space for all the src code.
        # com_vert_pos = np.array([surface[0][tri] for tri in com_tri])  # List comprehension is a bit slower for individual, so use this version
        v0 =  surface[0][com_tri[:, 0], :]
        v1 =  surface[0][com_tri[:, 1], :]
        v2 =  surface[0][com_tri[:, 2], :]

    tri_normals = np.cross(v1 - v0, v2 - v0)
    com_normal = np.mean(tri_normals, axis=0)
    com_normal = com_normal / np.linalg.norm(com_normal)

    return com_normal

def get_normals_parallel(labels_points, atcom=False, coords='MRI_RAS', subject='fsaverage', subjects_dir=None, surf_type='white',):
    """
    Parallel version of get_label_normal() function.  Example call: all_com_norms = np.array(get_normals_parallel(selected_label, atcom=False, surf_type='sphere'))
    """ 

    with concurrent.futures.ThreadPoolExecutor() as executor:
        result = np.array(list(executor.map(partial(get_label_normal, atcom=atcom, coords=coords, subject=subject, subjects_dir=subjects_dir, surf_type=surf_type), labels_points)))
    return result

def centre_loader(region_labels, parcellation, coords='MRI_RAS', subject='fsaverage', subjects_dir=None, surf_type='white', restrict_vertices=False):
    """
    Loads the centres of mass for the cortical regions in the parcellation.  Currently does not work for subcortical regions.

    Parameters:
    -----------
    region_labels: list
        List of region labels to load the centres of mass for.  Can be obtained from con.region_labels.
    parcellation: str
        The parcellation to load the centres of mass for.  Options are: 'HCPMMP1', 'native.1000Parcels_Yeo2011_7Networks', 'aparc' (which is DK atlas).
    coords: str
        The coordinate system to load the centres of mass in.  Can be MRI_RAS (Freesurfer's default output) or MNI.
    subject: str
        The subject to load the centres of mass for.  Default is 'fsaverage'.
    subjects_dir: str 
        The path to the subjects directory.  Default is None.
    surf_type: str
        The surface Freesurfer file to read the data from.  Default is 'white'.  'sphere' is also a sensible option. 
    restrict_vertices: bool
        Whether to restrict the vertices to the label vertices, otherwise can be any vertex from the higher resolution surf.  Default is False.  

    Returns:
    --------
    com_reordered: np.ndarray
        The centres of mass for the regions in the parcellation.  The order of the centres of mass is the same as the order of the region_labels.

    """

    selected_label = mne.read_labels_from_annot(subject, subjects_dir=subjects_dir, parc=parcellation)
    # Generate array of labels with same naming convention as the regions, 
    label_names = np.array([label.name[:-3] for label in selected_label])

    # Find the normal for the labels

    # Find the vertex of centre of mass for each region using MNE
    com = np.array([label.center_of_mass(subject=subject, restrict_vertices=restrict_vertices, surf=surf_type, subjects_dir=subjects_dir, ) for label in selected_label])
    com_hemi = np.array([0 if label.hemi=='lh' else 1 for label in selected_label])

    if coords == 'MNI':
        # Transform vertex to MNI (mm) space
        com = np.array(mne.vertex_to_mni(com, hemis=com_hemi, subject=subject, subjects_dir=subjects_dir))
    elif coords == 'MRI_RAS':
        surface_l = mne.surface.read_surface(subjects_dir / subject / "surf" / f"lh.{surf_type}")
        surface_r = mne.surface.read_surface(subjects_dir / subject / "surf" / f"rh.{surf_type}")
        com = np.array([surface_l[0][com[i]] if com_hemi[i] == 0 else surface_r[0][com[i]] for i in np.arange(len(com_hemi))])

    # Ignore subcortical structures from the region_names in the connectivity file
    if parcellation == 'HCPMMP1':
        # Glasser
        region_labels =  region_labels[19:]
    elif parcellation == 'native.1000Parcels_Yeo2011_7Networks':
        # Yeo
        region_labels =  region_labels[:-19]
    
    # Find the reordering to match region_labels with label_names
    reorder_indices = np.array([np.where(label_names == region)[0][0] for region in region_labels])
    com_reordered = com[reorder_indices]

    return com_reordered

def multiregionselector(con, centre, radius, restrict_hemi=True):
    """
    Selects multiple regions based on the distance from the centre of mass of the region.  This is useful for selecting regions for a seizure onset zone.

    Parameters:
    -----------
    con: TVB Connectivity Object
        The connectivity object to select the regions from.
    centre: str or int
        The region name or index to select the initial region from.
    radius: float
        The radius (mm) from the centre of mass to select the regions from.
    restrict_hemi: bool
        Whether to restrict the selection to the same hemisphere.  Default is True.  If False, will select regions from both hemispheres.

    Returns:
    --------
    selected_regions: np.array
        Array of indices of the selected regions.
    """

    if type(centre) == str:
        # Find the index of the region
        centre_ind = np.where(con.region_labels == centre)[0][0]
    else:
        centre_ind = centre
    centre = con.centres[centre_ind]

    # Calculate the distance from the centre of mass of each region
    distance = np.linalg.norm(con.centres - centre, axis=1)

    # Restrict to within same hemisphere:
    if restrict_hemi:
        # Obtain hemisphere for each label:
        hemis = ['lh' if 'LH' in region else 'rh' if 'RH' in region else 'sc' for region in con.region_labels] 
        same_hemi = np.array([hemis[i] == hemis[centre_ind] for i in range(len(hemis))])
        selected_regions = np.where((distance < radius) & same_hemi)[0]

    # If not restricting to same hemisphere, just select the regions within the radius
    else:        
        selected_regions = np.where(distance < radius)[0]

    return selected_regions

def patch_simulator(con, centre, fwd, info, selected_label, radius=20, restrict_hemi=True, full_src=True):
    """
    Simulate a patch of data in the source space.

    Parameters
    ----------
    con : Connectivity object
        The connectivity object.
    centre: str or int
        The region name or index to select the initial region from.   
    fwd : mne Forward object
        The forward solution.
    info : mne Info object
        The info object.
    selected_label : list
        The list of labels covering the entire brain.
    radius: int
        The extent of the patch in mm.
    restrict_hemi : bool
        Whether to restrict the simulation to one hemisphere.
    full_src : bool
        Whether to output the stc in the full source space.

    Returns
    -------
    stc : mne SourceEstimate object
        The simulated source estimate.
    sim : mne Evoked object
        The simulated evoked object.
    """

    source_simulator = mne.simulation.SourceSimulator(fwd['src'], )
    region_inds = multiregionselector(con, centre, radius=radius, restrict_hemi=restrict_hemi)

    # Generate the list of label names
    label_names = np.array([label.name for label in selected_label])

    for i in region_inds:
        # Get the region name
        region = con.region_labels[i]
        # Get the label
        label = selected_label[np.where(region==label_names)[0][0]]

        n_events = 1
        events = np.zeros((n_events, 3), int)
        events[:, 0] = 0 # Events sample.
        events[:, 2] = 0  # All events have the sample id.

        # Add the data for this region
        source_simulator.add_data(label, np.array([10e-9]), events)

    stc = source_simulator.get_stc()
    if full_src:
        stc = map_data_to_full_src(stc, fwd['src'])

    shape = stc.data.shape
    print('The data has %s vertex locations with %s sample points each.' % shape)

    # Exclude the four overlapping channels with same positions in the MNE  and two electrodes which are present in 10-20 but not 10-05
    fwd = mne.pick_channels_forward(fwd, exclude = ['T7', 'P7', 'P8', 'T8', 'O9', 'O10'])
    # mne.apply_forward_raw might be better to create a RAW object, since these forward solutions might be averaged later
    sim = mne.apply_forward(fwd, stc, info)
    # To specify EEG or MEG use sim.pick('eeg') outside of the function
    return stc, sim

def label_seizures(raw_x, n=1, label_spike=True):
    """  
    This function is to label the time series data from TVB's Epileptor model. Returns a labelled array of the same shape as raw_x. 
    n affects the smoothing which is performed on the data.  Larger n means more smoothing. 
    If working with temporally averaged data already, can set n to 1, otherwise n=500 appears appropriate.
    The seizure-deteciton part, instead of just loooking at slope of z, may also need to consider the value where there's a bifurcation, which is ~2.7 it also does the DC shift.
    This isn't a complication if there is no noise for the z parameter, but
    Should also add a sanity check component which compares this output with just taking the volatility/energy.

    Seziure classes are labelled as follows.
    # 0 - No spike
    # 1 - Seizure-like activity
    # 2 - Spontaneous Spike (if requested)

    Parameters:
    -----------
    raw_x: 4D array
        Time series data from the TVB simulation.  Dimensions are (time, state variables , nodes, -)
    n: int
        Smoothing parameter.  Default is 1.
    label_spike: bool
        Whether to label interictal/preictal spikes.  Default is True.
    """

    # Set up the array
    labelled_x = np.zeros((raw_x.shape[0],raw_x.shape[2]))
    # Calculate this outside of the loop as this is a stistic across entire raw_x to save time
    z_peak_height = np.mean(raw_x[:,1,:,0])+1.5*np.std(raw_x[:,1,:,0], dtype=np.float64)

    # Iterate for each region
    for region in range(raw_x.shape[2]):

        # First, put the spikes/peaks in the raw_x if requested
        if label_spike == True:
            peaks, _ = find_peaks(raw_x[:,0,region,0], distance = 500, height = np.mean(raw_x[:,0,region,0])+1.5*np.std(raw_x[:,0,region,0]))
            labelled_x[peaks,region] = 2

        # Perform peak detection on z.  
        # By default, find_peaks ignores the edges.  We don't want a peak that occurs at 0, but do wish to recognise one that occurs at the end. 
        # So add a 0 to the end of the array
        interim_array = np.concatenate((raw_x[:,1,region,0],np.zeros(1)))
        z_peaks, _ = find_peaks(interim_array, height = z_peak_height)
        #z_troughs, _ = find_peaks(-raw_x[:,1,region,0], distance = len(raw_x)/3, )
        # Troughs aren't that useful

        if z_peaks.shape[0]!=0:
        # If there are peaks, then we need to label the seizure activity.  Otherwise, we can just leave it as 0.

            # So go from the peak backwards to when the derivative is no longer positive.
            # Use z_diff
            z_diff = np.diff(np.convolve(raw_x[:, 1, region, 0], np.ones(n)/n, mode='same'))
            z_diff = np.concatenate((np.zeros((1,)),z_diff),axis=0)
            logic = z_diff > 0 

            # For each peak in the z_peaks, find the first previous time point where the derivative is no longer positive
            seizure_start = []
            for peak in z_peaks:
                # Find the first previous time point where the derivative is no longer positive

                # We want to start from the peak and work backwards. 
                for i in range(1,peak+1):
                    # We want to start from the peak and work backwards.  I am not too sure why I need this n/2 term... Probably because of the convolution
                    if not logic[peak-i-int(n/2)]:
                        peak_start = peak-i-int(n/2)
                        seizure_start.append(peak_start)
                        break

            for i, peak in enumerate(z_peaks):
            # Now fill in the labelled array
                labelled_x[seizure_start[i]:peak,region] = 1  

    return labelled_x

def seizure_prop_time(labelled_x, EZ, raw_t):
    """ Calculate the time between the Epileptic Zone (EZ) start and the start of seizure for every other zone.  i.e. the propagation time of the seizure.

    Parameters:
    -----------
    labelled_x: 2D array
        Labelled time series data from the TVB simulation.  Dimensions are (time, nodes)
    EZ: int
        Index of the Epileptic Zone.
    raw_t: 1D array
        Time data from the simulation.
    """
    # First, we need to find the start of the EZ
    # This is simply the index where the first 1 appears    
    EZ_start = np.where(labelled_x[:,EZ]==1)[0][0]

    # Now, we need to find the start of all the other zones
    # This is simply the index where the first 1 appears
    other_start = np.array([np.where(labelled_x[:,i]==1)[0][0] if (labelled_x[:,i]==1).any() else np.nan for i in np.arange(labelled_x.shape[1])])
    propagation_time = np.array([raw_t[(other_start-EZ_start).astype(int)]-raw_t.min() if not np.isnan(other_start) else np.nan for other_start in other_start])
    # Noting it will be 0 for the EZ itself and np.nan for a region with no seizure
    
    return propagation_time

def get_equilibrium(model, init):
    """ For obtaining the equilibrium condition of a single node of a model for use with a TVB simulation.   
    Will need to accompany the code with:
    epileptor_equil = models.Epileptor()
    epileptor_equil.x0 = np.array([-3.0])   
    init_cond = get_equilibrium(epileptor_equil, np.array([0.0, 0.0, 3.0, -1.0, 1.0, 0.0]))
    init_cond_reshaped = np.repeat(init_cond, num_regions).reshape((1, len(init_cond), num_regions, 1))

    Parameters:
    -----------
    model: TVB Model
        TVB Model object.
    init: 1D array
        Initial conditions for the model.
    """
    nvars = len(model.state_variables)
    cvars = len(model.cvar)

    def func(x):
        fx = model.dfun(x.reshape((nvars, 1, 1)),
                        np.zeros((cvars, 1, 1)))
        return fx.flatten()

    x = fsolve(func, init)
    return x

def tvb_loadtomne(raw_x, subject, fwd, info, parcellation, selected_label=None, fs_path= Path("/Applications/freesurfer/7.3.2/"), ):
    """ This is a function to load TVB simulations (or any other similar time series simulation) into MNE objects. 
        The MNE outputs are a stc object containing the source data reflecting the TVB time series, and the EEG/MEG simulation after solving the MNE forward on this underlying source data. 
    
    Parameters:
    -----------
    raw_x: 2D array (time, regions)
        The time series simulation data where each row is a time point and each column is a region.
    subject: str
        The subject ID.
    fwd: mne Forward
        The MNE forward object for producing the EEG simulation.
    info: mne Info
        The MNE Info object for the EEG simulation.
    parcellation: str
        The parcellation to use.  This is used to select the appropriate labels.  Options are: 'HCPMMP1', 'native.1000Parcels_Yeo2011_7Networks', 'aparc' (which is DK atlas).
    selected_label: list
        Provided if a custom list of labels wants to be used for the function.  Default is None, which means labels from the Freesurfer annot files are used.
    fs_path: Path
        The path to the FreeSurfer directory.
    
    Returns:
    --------
    stc: MNE SourceEstimate
        The MNE SourceEstimate object containing the TVB simulation data.
    sim: 
        The EEG/MEG simulation after solving the MNE forward.
    """

    # Read Region Labels in correct order (TVB) from a pre-made file and Ignore subcortical structures for now.  
    if parcellation == 'HCPMMP1':
        # Glasser
        regions = np.genfromtxt('External_Resources/HCP_data/HCP100206_con379/region_names.txt', dtype='str')
        regions =  regions[19:]
        if raw_x.shape[1] == 379:
            # If raw_x is in the original format, dictated by having 379 regions, then we need to remove the 19 subcortical structures, otherwise no need
            raw_x = raw_x[:,19:]
    elif parcellation == 'native.1000Parcels_Yeo2011_7Networks':
        # Yeo
        regions = np.genfromtxt('External_Resources/HCP_data/HCP100206_con1019/region_names.txt', dtype='str')
        regions =  regions[:-19]
        if raw_x.shape[1] == 1019:
            # If raw_x is in the original format, dictated by having 1019 regions, then we need to remove the 19 subcortical structures, otherwise no need
            raw_x = raw_x[:,:-19]
    elif parcellation == 'aparc':
        # DK
        regions = np.genfromtxt('68_Region_Labels.csv', dtype='str')

    # Creation of activity
    subjects_dir = fs_path / "subjects"
    tstep = 1 / info['sfreq'] 
    source_simulator = mne.simulation.SourceSimulator(fwd['src'], tstep=tstep)

    # Obtain list containing all the annotation labels if required from Freesurfer annot files
    if selected_label == None:
        selected_label = mne.read_labels_from_annot(subject, subjects_dir=subjects_dir, parc=parcellation)
        # Need to ensure that the provided region list follows the same order as the labels from the subject
        # Generate array of labels with same naming convention as the regions, 
        label_names = np.array([label.name[:-3] for label in selected_label])
    else:
        # Generate the list of label names
        label_names = np.array([label.name for label in selected_label])

    if len(regions) > raw_x.shape[1]:
        raise ValueError \
        ('Warning: The number of regions in the connectivity file is greater than the number of regions in the stc data.  ' \
        'This will bug out.  Consider running a helper function such as map_data_to_full_src() first or setting full_src=True.')

    for count, region in enumerate(regions):
        # Obtain the appropriate label
        label = selected_label[np.where(region==label_names)[0][0]]
        # Define the time course of the activity
        source_time_series = raw_x[:,count] #raw_x[:,0,count,0]
        # Define when the activity occurs using events.
        n_events = 1
        events = np.zeros((n_events, 3), int)
        events[:, 0] = np.arange(n_events)  # Events sample. - Size of this matters as it affects the length of the source simulation.  Assigns an event number to each event. 
        events[:, 2] = 1  # Assign all events have the same id (i.e. type).

        # Add the data for this region
        source_simulator.add_data(label, source_time_series, events)

    stc = source_simulator.get_stc()

    shape = stc.data.shape
    print('The data has %s vertex locations with %s sample points each.' % shape)

    # Exclude the four overlapping channels with same positions in the MNE  and two electrodes which are present in 10-20 but not 10-05
    fwd = mne.pick_channels_forward(fwd, exclude = ['T7', 'P7', 'P8', 'T8', 'O9', 'O10'])
    # mne.apply_forward_raw might be better to create a RAW object, since these forward solutions might be averaged later
    sim = mne.apply_forward(fwd, stc, info)
    # To specify EEG or MEG use sim.pick('eeg') outside of the function
    return stc, sim

def generate_mne_info(sampling_freq=78.125, montage_name='standard_1005', add_meg=True):
    """
    Generate an MNE Info object for the EEG/MEG simulation.  

    Parameters:
    -----------
    sampling_freq: float
        The sampling frequency of the EEG/MEG simulation.  Default is 78.125Hz from  78.125Hz = 20kHz /256 =  from 1/dt*1000 / scaling of 256  - Or it is /128 by the temporal averaging = 0.61 Hz
    montage_name: str
        The name of the montage to use, accepts any of the built-in MNE montages.  Default is 'standard_1005'.
    add_meg: bool
        Whether to add MEG channels to the info object.  Default is True.
    
    Returns:
    --------
    info: MNE Info
        MNE Info object.
    """

    # Create info object starting with a specified MNE montage
    montage = mne.channels.make_standard_montage(montage_name)
    ch_names = montage.ch_names 
    ch_types = ["eeg" for i in montage.ch_names]
    
    if add_meg:
        from mne.datasets import sample
        # Use the MEG sensor locations from MNE's sample data which involves the Neuromag System - Load the info object from MNE's datasource
        info_meg = mne.io.read_info(sample.data_path()/'MEG/sample/sample_audvis_raw.fif')

        # Create channel name and channel types in sensible order - EEG then MEG
        meg_ch_names = [ch for ch in info_meg.ch_names if ch.startswith('MEG')]
        meg_ch_types = [chtype for chtype in info_meg.get_channel_types() if chtype in ['grad', 'mag', ]]
        ch_names = ch_names + meg_ch_names
        ch_types = ch_types + meg_ch_types

    info = mne.create_info(ch_names=ch_names, sfreq=sampling_freq, ch_types=ch_types, )
    info.set_montage(montage)

    if add_meg:
        # Copy over the dev_head_t from the MEG info - We don't necessarily need to stay in how that experiment was set up
        # For now, let's obtain the forward tilt component only - Scaled to make the y component 1 to retain identity matrix
        info['dev_head_t']['trans'][:,1] =  info_meg['dev_head_t']['trans'][:,1] / info_meg['dev_head_t']['trans'][:,1][1] 
        # And the z-component translation
        info['dev_head_t']['trans'][2,3] = info_meg['dev_head_t']['trans'][2,3] 

        # Add the bad MEG channel - Should not be necessary
        # info['bads'] = ['MEG 2443']

        # Obtain indices of MEG channels
        meg_indices = np.array([i for i, ch_name in enumerate(info_meg.ch_names) if ch_name.startswith('MEG')])
        # Obtian MEG channel information from the MEG info object
        meg_chs = [info_meg['chs'][i] for i in meg_indices]

        # Edit the MEG channels in the chs attribute to provide sensor locations
        info['chs'][-len(meg_chs):] = meg_chs

        # Sanity check - Check if the channel names are the same
        for i, ch in enumerate(info['chs']):
            if ch['ch_name'] != info['ch_names'][i]:
                print(f"Channel names are different at index {i}")

    # Exclude the four overlapping channels with same positions in the MNE  and two electrodes which are present in 10-20 but not 10-05
    ch_inds = mne.pick_channels(ch_names=info['ch_names'], include=[], exclude=['T7', 'P7', 'P8', 'T8', 'O9', 'O10'])
    info = mne.pick_info(info, ch_inds)
    
    return info

def generate_com_labels(region_centres, label_names, label_hemi, subject):
    """
    Generate a list of MNE Label objects with the Centre of Mass as the sole vertex per label.

    Parameters:
    -----------
    region_centres: np.ndarray
        The centres of mass for the cortical regions.
    label_names: np.ndarray
        The names of the labels.
    label_hemi: np.ndarray
        The hemisphere of the labels.
    subject: str
        The subject of the labels.

    Returns:
    --------
    com_labels: list
        List of MNE Label objects with the Centre of Mass as the sole vertex per label.
    """

    # # Count the number of 'lh' and 'rh'labels that came before
    counts = {'lh': -1, 'rh': -1}
    com_labels = []
    for i in np.arange(len(label_names)):
        hemi = label_hemi[i]
        counts[hemi] += 1
        com_labels.append(mne.label.Label(vertices=np.array([counts[hemi]]), pos=np.array([region_centres[i]]), hemi=hemi, name=label_names[i], subject=subject))

    return com_labels