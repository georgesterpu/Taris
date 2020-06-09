import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

wb_labels = './avsr/misc/labels_boundaries_lrs2'
# wb_labels = './avsr/misc/labels_boundaries_libri_testclean'


def write_praat_intensity(indices, values, fname):
    import math
    from string import Template

    header = Template("""File type = "ooTextFile"
        Object class = "IntensityTier"
        xmin = 0 
        xmax = ${XMAX} 
        points: size = ${SIZE}
        """)

    pt = Template("""points [$INDEX]:
            number = $STAMP
            value = $VAL
        """)

    final = ''
    for idx, (pos, val), in enumerate(zip(indices, values)):
        current_pt = pt.substitute(INDEX=idx + 1, STAMP=float(pos / 1000), VAL=val)
        final += current_pt

    final = header.substitute(SIZE=idx + 1, XMAX=math.ceil(float(pos / 1000))) + final

    with open(fname, 'w') as f:
        f.write(final)


def get_segmentation_timestamps(halting_history):
    # halting_stamps = np.where(halting_history == 1)

    stamps = np.arange(halting_history.shape[0], dtype=np.float32) * 30

    return stamps, halting_history


def get_integer_crosses(cumsum):
    floored = np.floor(cumsum)
    diff = np.diff(floored, prepend=0)
    crosses = np.where(diff == 1)
    return crosses


def get_nearest_boundary_error(boundaries, cross):
    delay = np.abs(boundaries - cross).min()
    return delay


def compute_objective_score(ref_ts, hyp_ts):
    r"""
    TODO Not implemented yet
    :return:
    """
    # delays = [get_nearest_boundary_error(ref_ts, c) for c in hyp_ts]
    # return np.mean(delays)
    return 0.0


def load_timestamps_dict(dict_path):
    with open(dict_path, 'r') as f:
        contents = f.read().splitlines()
    label_dict = dict([line.split(' ', maxsplit=1) for line in contents])
    return label_dict


def gen_len_hists(file, ts, cumsum):
    ref_ts_dict = load_timestamps_dict(wb_labels)  # TODO refactor me
    ref_ts_str = ref_ts_dict[file].split()
    ref_ts = np.asarray([float(x)*1000 for x in ref_ts_str])

    crosses = get_integer_crosses(cumsum)
    hyp_ts = ts[crosses]

    score_mean = compute_objective_score(ref_ts, hyp_ts)

    hyp_lens = np.int32(np.diff(hyp_ts, prepend=0))
    ref_lens = np.int32(np.diff(ref_ts, prepend=0))

    return score_mean, hyp_lens, ref_lens


def average_dict_values(d):
    return np.mean(list(d.values()))


def plot_histograms(hist1, hist2, bins, fname):
    plt.bar(bins[:-1], hist1, alpha=0.5, width=25.0)
    plt.bar(bins[:-1], hist2, alpha=0.5, width=25.0)
    plt.legend(['hypothesis', 'reference'])
    plt.xlabel('Word length [ms]')
    plt.ylabel('Number of occurrences')
    plt.xlim(left=0)
    plt.savefig(fname, dpi=300, bbox_inches='tight', pad_inches=0.0)
