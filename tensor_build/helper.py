import numpy as np
#We choose 128 as # time bin. We have 44 DOMs. We have 1 feature which is hit count. And we have N events with non-zero hits. 
#Our input tensor X has shape (N, 1, 44, 128). Because energy variation is large, we put output in log scale. Our output y is log10(energy)


N_STRINGS = 4
N_MODULES = 11
N_DOM = N_STRINGS * N_MODULES  # 44

#based on the 99th percetile time we found earlier
T_MAX = 3500.0
N_T_BINS = 128
DT = T_MAX / N_T_BINS
MIN_HITS_PER_EVENT = 1 #cut zero hits events

def dom_index(string_id, module_id):
    #4 strings, 11 modules each. flatten to a single line
    return int(string_id) * N_MODULES + int(module_id)


def build_event_image(photons,
                      n_dom=N_DOM,
                      n_t_bins=N_T_BINS,
                      t_max=T_MAX,
                      dt=DT):


    t = np.array(photons["t"], dtype=np.float64)
    if t.size == 0:
        return None #no hits

    dom_ids    = np.array(photons["sensor_id"], dtype=np.int64)    # module_id
    dom_strings = np.array(photons["string_id"], dtype=np.int64)    # string_id

    #since every event arrives at different times, translate to absolute time with respect to time of first hit for each event. Name that t0
    t0 = t.min()
    t_abs = t - t0


    mask = (t_abs >= 0.0) & (t_abs < t_max) #we exclude the most extream cases. no need for them
    if not np.any(mask):
        return None

    t_abs   = t_abs[mask]
    dom_ids    = dom_ids[mask]
    dom_strings = dom_strings[mask]


    j = (t_abs / dt).astype(np.int64)         # put into timebins



    dom_idx = (dom_strings * N_MODULES + dom_ids).astype(np.int64)
    

    

    #put dom and 
    flat_idx = dom_idx * n_t_bins + j
    img_flat = np.bincount(flat_idx,
                           minlength=n_dom * n_t_bins).astype(np.float32)

    #back to doms and bins
    img = img_flat.reshape(n_dom, n_t_bins)

    if img.sum() < MIN_HITS_PER_EVENT:
        return None
    img = np.log1p(img) #var is too large use log scale

    return img



def get_muon_energy(mc_truth):

    types = np.array(mc_truth["final_state_type"])
    energies = np.array(mc_truth["final_state_energy"])
    mask = (types == 13) | (types == -13)# +13 is muon, -13 is anti-muon. but track is identical we treat it the same

    muon_E = energies[mask]
    if muon_E.size == 0:
        return None
    return float(muon_E[0])
