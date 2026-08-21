"""
gen_noise_covariance_fixture.py
=================================

Numerical fixture for ClosedLoop/test_noise_covariance_equivalence.cpp:
preprocesses the real test recording exactly like calibrate_all_units.py
does (causal highpass + CAR, float32, matching the GPU pipeline), picks a
real target unit + explicit interferers, computes its selected channels
(gf.select_channels) and noise covariance (gf.noise_covariance_vectorized),
and dumps everything the C++ port needs to reproduce the same R matrix:
the preprocessed scratch file itself (persisted at a fixed path -- Python's
own copy is normally deleted at exit), selected channel indices, target+
interferer spike times (train split only), and the expected R.

Isolates NoiseCovariance.cpp from channel selection (LcmvFit.cpp,
already validated separately) and from the offline GPU preprocessing driver
(OfflinePreprocessor.cpp) -- this fixture supplies known-good channels/data
directly, matching the "isolate one function per phase" pattern already
used for gen_lcmv_fixture.py.
"""
import argparse
import shutil
import struct
import sys

import numpy as np

sys.path.insert(0, "C:/Users/kesha/OneDrive/Desktop/LiveSorter/FilterGen")
import generate_filter as gf
import threshold_sweep_real as tsr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ks-dir", required=True)
    ap.add_argument("--bin-path", required=True)
    ap.add_argument("--channel-map-json", required=True)
    ap.add_argument("--target", type=int, required=True)
    ap.add_argument("--interferers", type=int, nargs="+", required=True)
    ap.add_argument("--n-channels", type=int, default=5)
    ap.add_argument("--template-length", type=int, default=61)
    ap.add_argument("--template-offset", type=int, default=20)
    ap.add_argument("--train-frac", type=float, default=0.5)
    ap.add_argument("--max-spikes", type=int, default=2000)
    ap.add_argument("--fc", type=float, default=300.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-meta", required=True)
    ap.add_argument("--out-scratch", required=True)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    class _Args:
        pass
    load_args = _Args()
    load_args.ks_dir = args.ks_dir
    load_args.bin_path = args.bin_path
    load_args.meta_path = None
    load_args.channel_map_json = args.channel_map_json
    load_args.filter = True
    load_args.car = True
    load_args.fc = args.fc
    load_args.causal_highpass = True

    spike_t, spike_cl, data, np_ch, fs = tsr.load_and_prepare(load_args, rng, dtype=np.float32)

    split_t = int(args.train_frac * data.shape[0])
    data_train = data[:split_t]
    spike_t_train = spike_t[spike_t < split_t]
    spike_cl_train = spike_cl[spike_t < split_t]

    target_spikes = spike_t_train[spike_cl_train == args.target]
    interferer_times = [spike_t_train[spike_cl_train == cid] for cid in args.interferers]

    target_wf, _ = gf.mean_waveform(data_train, target_spikes, args.template_length,
                                     args.template_offset, args.max_spikes, rng)
    interferer_wfs = [gf.mean_waveform(data_train, t, args.template_length, args.template_offset,
                                        args.max_spikes, rng)[0] for t in interferer_times]

    sel = gf.select_channels(target_wf, interferer_wfs, args.n_channels)
    sel_channels = np_ch[sel]

    data_sel = data_train[:, sel]
    local_spike_times = np.sort(np.concatenate([target_spikes] + interferer_times))

    R = gf.noise_covariance_vectorized(data_sel, local_spike_times, args.template_length,
                                        args.template_offset, data_sel.shape[0])

    # Persist the scratch file the C++ side will load -- data.filename is
    # load_and_prepare's tempfile, normally deleted at Python exit.
    shutil.copyfile(data.filename, args.out_scratch)

    # Binary fixture layout (matching this session's established pattern --
    # no general JSON parser on the C++ side):
    #   int64 nSamplesTrain, int32 nChannelsGroup, int32 N,
    #   int32 templateLength, int32 templateOffset
    #   int32 nSpikes
    #   int64[nSpikes]  localSpikeTimes
    #   int32[N]        selChannels (indices into the CAR-group columns)
    #   int32 dim
    #   float64[dim*dim] R
    sel_i32 = np.asarray(sel, dtype="<i4")
    spikes_i64 = np.asarray(local_spike_times, dtype="<i8")
    R_f64 = R.astype("<f8").ravel()

    with open(args.out_meta, "wb") as fh:
        fh.write(struct.pack("<qiiii", int(split_t), int(data.shape[1]), int(len(sel)),
                              args.template_length, args.template_offset))
        fh.write(struct.pack("<i", len(spikes_i64)))
        fh.write(spikes_i64.tobytes())
        fh.write(sel_i32.tobytes())
        fh.write(struct.pack("<i", int(R.shape[0])))
        fh.write(R_f64.tobytes())

    print(f"Wrote scratch file to {args.out_scratch} ({data.nbytes/1e9:.1f} GB)")
    print(f"Wrote fixture metadata to {args.out_meta} "
          f"(dim={R.shape[0]}, nSpikes={len(local_spike_times)}, sel={list(sel)})")


if __name__ == "__main__":
    main()
