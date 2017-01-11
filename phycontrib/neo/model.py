
import logging
import os
import os.path as op
import shutil
import neo
import quantities as pq
import numpy as np
import copy
import shutil
import exdir

from phy.io.array import (_concatenate_virtual_arrays,
                          _index_of,
                          _spikes_in_clusters,
                          )

logger = logging.getLogger(__name__)

try:
    from klusta.traces import PCA as klusta_pca
except ImportError:  # pragma: no cover
    logger.warn("Package klusta not installed: the KwikGUI will not work.")


def copy_file_or_folder(fname, fname_copy):
    if op.isfile(fname):
        shutil.copy(fname, fname_copy)
    if op.isdir(fname):
        shutil.copytree(fname, fname_copy)


def delete_file_or_folder(fname):
    if op.isfile(fname):
        os.remove(fname)
    if op.isdir(fname):
        shutil.rmtree(fname)

# TODO test if saving preserves cluster id
# TODO save group metadata
# TODO save metadata
# TODO make klustaexdir script that takes neo or rawdata files and saves to exdir
# TODO save masks
# TODO check masks
# TODO load features and masks if exdir and exists
# TODO save exdir without neo?
# TODO save probe info in exdir


class NeoModel(object):
    n_pcs = 3

    def __init__(self, data_path=None, channel_group=None, segment_num=None,
                 save_path=None, save_ext='.exdir', **kwargs):
        data_path = data_path or ''
        dir_path = (op.dirname(op.realpath(op.expanduser(data_path)))
                    if data_path else os.getcwd())
        save_path = save_path or dir_path
        fname = op.splitext(op.split(data_path)[1])[0]
        self.save_ext = save_ext
        if self.save_ext[0] != '.':
            self.save_ext = '.' + self.save_ext
        self.save_path = op.join(save_path, fname + self.save_ext)
        backup = self.save_path+'.bak'
        if op.exists(backup):
            delete_file_or_folder(backup)
        if op.exists(self.save_path):
            copy_file_or_folder(self.save_path, backup)
        self.data_path = data_path

        self.segment_num = segment_num
        self.channel_group = channel_group
        self.dir_path = dir_path
        self.__dict__.update(kwargs)
        self._load_data()

    def describe(self):
        def _print(name, value):
            print("{0: <26}{1}".format(name, value))

        _print('Data file', self.data_path)
        _print('Number of channels', len(self.chx.index))
        _print('Duration', '{}'.format(self.duration))
        _print('Number of spikes', self.n_spikes)
        _print('Available channel groups', self.avail_groups)

    def save(self, spike_clusters):
        assert spike_clusters.shape == self.spike_clusters.shape
        assert spike_clusters.dtype == self.spike_clusters.dtype
        self.spike_clusters = spike_clusters
        blk = neo.Block()
        seg = neo.Segment(duration=self.duration)
        blk.segments.append(seg)
        chx = neo.ChannelIndex(index=self.chx.index,
                               name=self.chx.name,
                               **self.chx.annotations)
        blk.channel_indexes.append(chx)
        wf_units = self.sptrs[0].waveforms.units
        for sc in np.unique(spike_clusters):
            mask = self.spike_clusters == sc
            waveforms = np.swapaxes(self.waveforms[mask], 1, 2) * wf_units
            sptr = neo.SpikeTrain(times=self.spike_times[mask] * pq.s,
                                  waveforms=waveforms,
                                  sampling_rate=self.sample_rate * pq.Hz,
                                  name='cluster #%i' % sc,
                                  t_stop=self.duration,
                                  **{'cluster_id': sc})
            sptr.channel_index = chx
            unt = neo.Unit(name='Unit #{}'.format(sc), **{'cluster_id': sc})
            unt.spiketrains.append(sptr)
            chx.units.append(unt)
            seg.spiketrains.append(sptr)
        if self.save_ext == '.exdir':
            io = neo.ExdirIO(self.save_path, 'w')
        io.save(blk)
        try:
            io.close()
        except:
            pass  # TODO except proper error
        if self.save_ext == '.exdir': # TODO blir cluster identitet bevart av neo.io?
            # save features and masks
            self._exdir_folder = exdir.File(folder=self.save_path)
            self._processing = self._exdir_folder["processing"]
            # self.save_spike_clusters(spike_clusters)
            self.save_spike_features(spike_clusters)
            # self.save_event_waveform(spike_clusters)
            # self.save_spike_features(spike_clusters)

    # def save_spike_clusters(self, spike_clusters):
    #     # for saving phy data directly to disc
    #     grp = self.channel_group
    #     ch_group = self._processing['channel_group_{}'.format(grp)]
    #     clust = ch_group.create_group('Clustering')
    #     clust.create_dataset('electrode_idx', self.chx.index)
    #     clust.create_dataset('cluster_nums', np.unique(spike_clusters))
    #     clust.create_dataset('nums', spike_clusters)
    #     clust.create_dataset('times', self.spike_times)

    def save_spike_features(self, spike_clusters):
        # for saving phy data directly to disc
        grp = 'channel_group_{}'.format(self.channel_group)
        seg = 'Segment_{}'.format(self.segment_num)
        ch_group = ch_group = self._processing[seg][grp]
        feat = ch_group.create_group('FeatureExtraction')
        feat.create_dataset('electrode_idx', self.chx.index)
        feat.create_dataset('features', self.features)
        feat.create_dataset('masks', self.masks)
        feat.create_dataset('times', self.spike_times)

    # def save_event_waveform(self, spike_times, waveforms, channel_indexes,
    #                          sampling_rate, channel_group, t_start, t_stop):
    #     event_wf_group = channel_group.create_group('EventWaveform')
    #     wf_group = event_wf_group.create_group('waveform_timeseries')
    #     wf_group.attrs['start_time'] = t_start
    #     wf_group.attrs['stop_time'] = t_stop
    #     wf_group.attrs['electrode_idx'] = channel_indexes
    #     ts_data = wf_group.create_dataset("timestamps", spike_times)
    #     wf = wf_group.create_dataset("waveforms", waveforms)
    #     wf.attrs['sample_rate'] = sampling_rate

    # def save_unit_times(self, sptrs, channel_group, t_start, t_stop):
    #     unit_times_group = channel_group.create_group('UnitTimes')
    #     unit_times_group.attrs['start_time'] = t_start
    #     unit_times_group.attrs['stop_time'] = t_stop
    #     for sptr_id, sptr in enumerate(sptrs):
    #         times_group = unit_times_group.create_group('{}'.format(sptr_id))
    #         ts_data = times_group.create_dataset('times', sptr.times)

    def _load_data(self):
        io = neo.get_io(self.data_path)
        blk = io.read_block()  # TODO params to select what to read
        try:
            io.close()
        except:
            pass
        if self.segment_num is None:
            self.segment_num = 0  # TODO find the right seg num
        self.seg = blk.segments[self.segment_num]
        self.duration = self.seg.duration

        if not all(['group_id' in st.channel_index.annotations
                    for st in self.seg.spiketrains]):
            raise ValueError('"group_id" must be in' +
                             ' channel_index.annotations')
        grps = {st.channel_index.annotations['group_id']:
                st.channel_index.name for st in self.seg.spiketrains}
        grps_ids = np.array(list(grps.keys()), dtype=int)
        self.avail_groups = np.sort(np.unique(grps_ids))
        if self.channel_group is None:
            self.channel_group = self.avail_groups[0]
        if self.channel_group not in self.avail_groups:
            raise ValueError('channel group not available,' +
                             ' see available channel groups in neo-describe')
        self.chx, = (chx for chx in blk.channel_indexes
                     if chx.name == grps[self.channel_group])
        self.n_chans = len(self.chx.index)

        self.sptrs = [st for st in self.seg.spiketrains
                      if st.channel_index.name == grps[self.channel_group]]
        self.sample_rate = self.sptrs[0].sampling_rate.rescale('Hz').magnitude
        # self.sorted_idxs = np.argsort(times)
        self.spike_times = self._load_spike_times()  # [self.sorted_idxs]
        ns, = self.n_spikes, = self.spike_times.shape

        self.spike_clusters = self._load_spike_clusters()  # [self.sorted_idxs]
        assert self.spike_clusters.shape == (ns,)

        self.cluster_groups = self._load_cluster_groups()

        self.waveforms = self._load_waveforms()  # [self.sorted_idxs, :, :]
        assert self.waveforms.shape[::2] == (ns, self.n_chans)
        
        if op.split(self.data_path)[-1] == '.exdir':
            pass
            # TODO try to load features from file
        self.features, self.masks = self._load_features_masks()

        self.amplitudes = self._load_amplitudes()  # [self.sorted_idxs]
        assert self.amplitudes.shape == (ns, self.n_chans)

        # TODO load positino from params
        ch_pos = np.zeros((self.n_chans, 2))
        ch_pos[:, 1] = np.arange(self.n_chans)
        self.channel_positions = ch_pos

    def get_metadata(self, name):
        return None

    def get_waveforms(self, spike_ids, channel_ids):
        wf = self.waveforms
        return wf[spike_ids, :, :][:, :, channel_ids]  # TODO wierd fix

    def get_features_masks(self, spike_ids, channel_ids):
        # we select the primary principal component
        features = self.features[:, :, 0]
        features = features[spike_ids, :][:, channel_ids]
        features = np.reshape(features, (len(spike_ids), len(channel_ids)))
        masks = np.ones((len(spike_ids), len(channel_ids)), dtype=bool)
        return features, masks

    def _load_features_masks(self):
        logger.debug("Loading features.")
        masks = np.ones((self.n_spikes, self.n_chans), dtype=bool)
        pca = klusta_pca(self.n_pcs)
        pca.fit(self.waveforms, masks)
        features = pca.transform(self.waveforms)
        return features, masks

    def _load_cluster_groups(self):
        logger.debug("Loading cluster groups.")
        out = {i: 'unsorted' for i in np.unique(self.spike_clusters)}
        return out

    def _load_spike_times(self):
        logger.debug("Loading spike times.")
        out = np.array([t for sptr in self.sptrs
                        for t in sptr.times.rescale('s').magnitude])
        # HACK sometimes out is shape (n_spikes, 1)
        return np.reshape(out, len(out))

    def _load_spike_clusters(self):
        logger.debug("Loading spike clusters.")
        out = np.array([i for j, sptr in enumerate(self.sptrs)
                        for i in [j]*len(sptr)])
        # HACK sometimes out is shape (n_spikes, 1)
        return np.reshape(out, len(out))

    def _load_waveforms(self):  # TODO this should be masks for memory saving
        logger.debug("Loading spike waveforms.")
        wfs = np.vstack([sptr.waveforms for sptr in self.sptrs])
        wfs = np.array(wfs, dtype=np.float64)
        assert wfs.shape[1:] == self.sptrs[0].waveforms.shape[1:]
        # neo: num_spikes, num_chans, samples_per_spike = wfs.shape
        return wfs.swapaxes(1, 2)

    def _load_amplitudes(self):
        logger.debug("Loading spike amplitudes.")
        left_sweep = 0.2 * pq.ms  # TODO select left sweep
        # TODO multiple sampling rates is not allowed
        mask = int(self.sample_rate * left_sweep.rescale('s').magnitude)
        out = self.waveforms[:, mask, :]
        return out
