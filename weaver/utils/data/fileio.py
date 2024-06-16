import math
import awkward as ak
import tqdm
import traceback
from .tools import _concat
from ..logger import _logger, warn_n_times
import numpy as np


#def _read_hdf5(filepath, branches, load_range=None):
    #import tables
    #tables.set_blosc_max_threads(4)
    #with tables.open_file(filepath) as f:
        #outputs = {k: getattr(f.root, k)[:] for k in branches}
    #if load_range is None:
        #load_range = (0, 1)
    #start = math.trunc(load_range[0] * len(outputs[branches[0]]))
    #stop = max(start + 1, math.trunc(load_range[1] * len(outputs[branches[0]])))
    #for k, v in outputs.items():
        #outputs[k] = v[start:stop]
    #return ak.Array(outputs)

def preprocessing(ufo_eta, ufo_phi):
    # Turn all 'NaN' values into 0s: locate NaNs, set to 0 for pt, eta, phi
    # Centre hardest constituent in eta/phi plane
    # Find eta and phi shifts that need to be applied
    eta_shift = ufo_eta[:, 0]
    phi_shift = ufo_phi[:, 0]

    # Apply them using np.newaxis
    eta_center = ufo_eta - eta_shift[:, np.newaxis]
    phi_center = ufo_phi - phi_shift[:, np.newaxis]

    # Fix discontinuity in phi at +/- pi using np.where
    phi_center = np.where(phi_center > np.pi, phi_center - 2*np.pi, phi_center)
    phi_center = np.where(phi_center < -np.pi, phi_center + 2*np.pi, phi_center)

    # 2. Rotate such that 2nd hardest constituent sits on negative phi axis
    second_eta = eta_center[:, 1]
    second_phi = phi_center[:, 1]
    alpha = np.arctan2(second_phi, second_eta) + np.pi/2
    eta_rot = (eta_center * np.cos(alpha[:, np.newaxis]) + phi_center * np.sin(alpha[:, np.newaxis]))
    phi_rot = (-eta_center * np.sin(alpha[:, np.newaxis]) + phi_center * np.cos(alpha[:, np.newaxis]))

    # 3. If needed, reflect so 3rd hardest constituent is in positive eta
    third_eta = eta_rot[:, 2]
    parity = np.where(third_eta < 0, -1, 1)
    eta_flip = (eta_rot * parity[:, np.newaxis]) #.astype(np.float32)
    # Cast to float32 needed to keep numpy from turning eta to double precision

    return eta_flip, phi_rot

def _read_hdf5(filepath, branches, load_range=None):
    import h5py
    with h5py.File(filepath, "r") as h5file:
        concat = True # define to be true when you want to concatenate the flow and tower inputs
        towers = False # define to be true when using towers only
        branches_flow = [branch for branch in branches if branch.startswith('flow') == True]
        branches_jets = [branch for branch in branches if branch.startswith('jets') == True]
        branches_towers = [branch for branch in branches if branch.startswith('towers') == True]
        if concat == True:
            branches_towers = [branch.replace('flow', 'towers') for branch in branches_flow]
        outputs_flow = {}
        outputs_jets = {}
        outputs_towers = {}
        if load_range is None:
            load_range = (0, 1)
        if len(branches_flow) != 0:
            branches2_flow = [branch.replace('flow_', '') for branch in branches_flow]
            data_flow = h5file['flow']
            outputs_flow = {'flow_' + k: data_flow.fields(k) for k in branches2_flow}
            columns_list_flow = list(branches2_flow) # dictionary keys as a list, so that it can be iterated over
            start_flow = math.trunc(load_range[0] * len(outputs_flow['flow_' + columns_list_flow[0]]))
            stop_flow = max(start_flow + 1, math.trunc(load_range[1] * len(outputs_flow['flow_' + columns_list_flow[0]])))
            for k, v in outputs_flow.items():
                outputs_flow[k] = v[start_flow:stop_flow]
            if 'flow_deta' and 'flow_dphi' in branches:
                flow_deta_dphi = preprocessing(outputs_flow['flow_deta'], outputs_flow['flow_dphi'])
                outputs_flow['flow_deta'] = flow_deta_dphi[0]
                outputs_flow['flow_dphi'] = flow_deta_dphi[1]
        if len(branches_jets) != 0:
            branches2_jets = [branch.replace('jets_', '') for branch in branches_jets]
            data_jets = h5file['jets']
            outputs_jets = {'jets_' + k: data_jets.fields(k) for k in branches2_jets}
            columns_list_jets = list(branches2_jets) # dictionary keys as a list, so that it can be iterated over
            start_jets = math.trunc(load_range[0] * len(outputs_jets['jets_' + columns_list_jets[0]]))
            stop_jets = max(start_jets + 1, math.trunc(load_range[1] * len(outputs_jets['jets_' + columns_list_jets[0]])))
            for k, v in outputs_jets.items():
                outputs_jets[k] = v[start_jets:stop_jets]
            if 'jets_PartonTruthLabelID' in branches:
                outputs_jets['jets_PartonTruthLabelID'] = np.asarray([np.int32(x<21) for x in outputs_jets['jets_PartonTruthLabelID']])
            if 'jets_eta' in branches:
                outputs_jets['jets_eta'] = np.asarray([abs(x) for x in outputs_jets['jets_eta']])
        if len(branches_towers) != 0:
            branches2_towers = [branch.replace('towers_', '') for branch in branches_towers]
            data_towers = h5file['towers']
            outputs_towers = {'towers_' + k: data_towers.fields(k) for k in branches2_towers}
            columns_list_towers = list(branches2_towers) # dictionary keys as a list, so that it can be iterated over
            start_towers = math.trunc(load_range[0] * len(outputs_towers['towers_' + columns_list_towers[0]]))
            stop_towers = max(start_towers + 1, math.trunc(load_range[1] * len(outputs_towers['towers_' + columns_list_towers[0]])))
            for k, v in outputs_towers.items():
                outputs_towers[k] = v[start_towers:stop_towers]
            if 'towers_deta' and 'towers_dphi' in branches:
                towers_deta_dphi = preprocessing(outputs_towers['towers_deta'], outputs_towers['towers_dphi'])
                outputs_towers['towers_deta'] = towers_deta_dphi[0]
                outputs_towers['towers_dphi'] = towers_deta_dphi[1]
        if len(outputs_towers) != 0 and concat == True:
            for key_flow in outputs_flow.keys(): # must use same tower variables as flow variables
                key_towers = key_flow.replace('flow', 'towers')
                NaNs = np.isnan(outputs_flow[key_flow])
                outputs_flow[key_flow][NaNs] = 0
                NaNs = np.isnan(outputs_towers[key_towers])
                outputs_towers[key_towers][NaNs] = 0
                outputs_flow[key_flow] = np.hstack([outputs_flow[key_flow], outputs_towers[key_towers]])
                #print(len(outputs_flow[key_flow][0]))
        #print(branches_flow, branches_jets)
        outputs_flow.update(outputs_jets)
        if towers == True:
            outputs_flow.update(outputs_towers)
        #outputs_flow.update(outputs_towers)
        return ak.Array(outputs_flow)

def _read_root(filepath, branches, load_range=None, treename=None, branch_magic=None):
    import uproot
    with uproot.open(filepath) as f:
        if treename is None:
            treenames = set([k.split(';')[0] for k, v in f.items() if getattr(v, 'classname', '') == 'TTree'])
            if len(treenames) == 1:
                treename = treenames.pop()
            else:
                raise RuntimeError(
                    'Need to specify `treename` as more than one trees are found in file %s: %s' %
                    (filepath, str(treenames)))
        tree = f[treename]
        if load_range is not None:
            start = math.trunc(load_range[0] * tree.num_entries)
            stop = max(start + 1, math.trunc(load_range[1] * tree.num_entries))
        else:
            start, stop = None, None
        if branch_magic is not None:
            branch_dict = {}
            for name in branches:
                decoded_name = name
                for src, tgt in branch_magic.items():
                    if src in decoded_name:
                        decoded_name = decoded_name.replace(src, tgt)
                branch_dict[name] = decoded_name
            outputs = tree.arrays(filter_name=list(branch_dict.values()), entry_start=start, entry_stop=stop)
            for name, decoded_name in branch_dict.items():
                if name != decoded_name:
                    outputs[name] = outputs[decoded_name]
        else:
            outputs = tree.arrays(filter_name=branches, entry_start=start, entry_stop=stop)
    return outputs


def _read_awkd(filepath, branches, load_range=None):
    import awkward0
    with awkward0.load(filepath) as f:
        outputs = {k: f[k] for k in branches}
    if load_range is None:
        load_range = (0, 1)
    start = math.trunc(load_range[0] * len(outputs[branches[0]]))
    stop = max(start + 1, math.trunc(load_range[1] * len(outputs[branches[0]])))
    for k, v in outputs.items():
        outputs[k] = ak.from_awkward0(v[start:stop])
    return ak.Array(outputs)


def _read_parquet(filepath, branches, load_range=None):
    outputs = ak.from_parquet(filepath, columns=branches)
    if load_range is not None:
        start = math.trunc(load_range[0] * len(outputs))
        stop = max(start + 1, math.trunc(load_range[1] * len(outputs)))
        outputs = outputs[start:stop]
    return outputs


def _read_files(filelist, branches, load_range=None, show_progressbar=False, file_magic=None, **kwargs):
    import os
    branches = list(branches)
    table = []
    if show_progressbar:
        filelist = tqdm.tqdm(filelist)
    for filepath in filelist:
        ext = os.path.splitext(filepath)[1]
        if ext not in ('.h5', '.root', '.awkd', '.parquet'):
            raise RuntimeError('File %s of type `%s` is not supported!' % (filepath, ext))
        try:
            if ext == '.h5':
                a = _read_hdf5(filepath, branches, load_range=load_range)
            elif ext == '.root':
                a = _read_root(filepath, branches, load_range=load_range,
                               treename=kwargs.get('treename', None),
                               branch_magic=kwargs.get('branch_magic', None))
            elif ext == '.awkd':
                a = _read_awkd(filepath, branches, load_range=load_range)
            elif ext == '.parquet':
                a = _read_parquet(filepath, branches, load_range=load_range)
        except Exception as e:
            a = None
            _logger.error('When reading file %s:', filepath)
            _logger.error(traceback.format_exc())
        if a is not None:
            if file_magic is not None:
                import re
                for var, value_dict in file_magic.items():
                    if var in a.fields:
                        warn_n_times(f'Var `{var}` already defined in the arrays '
                                     f'but will be OVERWRITTEN by file_magic {value_dict}.')
                    a[var] = 0
                    for fn_pattern, value in value_dict.items():
                        if re.search(fn_pattern, filepath):
                            a[var] = value
                            break
            table.append(a)
    table = _concat(table)  # ak.Array
    if len(table) == 0:
        raise RuntimeError(f'Zero entries loaded when reading files {filelist} with `load_range`={load_range}.')
    return table


def _write_root(file, table, treename='Events', compression=-1, step=1048576):
    import uproot
    if compression == -1:
        compression = uproot.LZ4(4)
    with uproot.recreate(file, compression=compression) as fout:
        for k,v in table.items():
            print("k", k, "v", v)
            print(type(v))
        if isinstance(v, np.ndarray): ## Added by Eesha
            dtype = v.dtype
        elif isinstance(v, ak.highlevel.Array):
            dtype = v.type
        else:
            raise TypeError("Unsupported type for v")
        tree = fout.mktree(treename, {k: dtype for k, v in table.items()})
        start = 0
        while start < len(list(table.values())[0]) - 1:
            tree.extend({k: v[start:start + step] for k, v in table.items()})
            start += step
