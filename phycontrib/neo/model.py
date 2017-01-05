
import logging
import os
import os.path as op
import shutil
import neo
import quantities as pq
import numpy as np

from phy.io.array import (_concatenate_virtual_arrays,
                          _index_of,
                          _spikes_in_clusters,
                          )

logger = logging.getLogger(__name__)



class NeoModel(object):

    def __init__(self, data_path=None, **kwargs):
        data_path = data_path or ''
        dir_path = (op.dirname(op.realpath(op.expanduser(data_path)))
                    if data_path else os.getcwd())
        self.data_path = data_path
        self.dir_path = dir_path
        self.__dict__.update(kwargs)
        self._load_data()

    def describe(self):
        def _print(name, value):
            print("{0: <24}{1}".format(name, value))

        _print('Data file', self.data_path)
        _print('Number of channels', len(self.chx.channel_ids))
        _print('Duration', '{}'.format(self.seg.duration))
        _print('Number of spikes', self.n_spikes)
        _print('Available channel groups', self.avail_groups)

    def _load_data(self):
        IO = neo.get_io(self.data_path)
        blk = IO.read_block() # TODO params to select what to read
        if self.segment_num is None:
            self.segment_num = 0 # TODO find the right seg num
        self.seg = blk.segments[self.segment_num]

        self.avail_groups = np.sort(np.unique([st.channel_index.index
                                 for st in self.seg.spiketrains]))
        if self.channel_group is None:
            self.channel_group = self.avail_groups[0]
        if not self.channel_group in self.avail_groups:
            raise ValueError('channel group not available,'+
                             ' see available channel groups in neo-describe')
        self.chx, = (chx for chx in blk.channel_indexes
                     if chx.index==self.channel_group)

        self.sptrs = [st for st in self.seg.spiketrains
                      if st.channel_index.index==self.channel_group]

        #self.sorted_idxs = np.argsort(times)
        self.spike_times = self._load_spike_times()#[self.sorted_idxs]
        ns, = self.n_spikes, = self.spike_times.shape

        self.spike_clusters = self._load_spike_clusters()#[self.sorted_idxs]
        assert self.spike_clusters.shape == (ns,)

        self.cluster_groups = self._load_cluster_groups()

        self.waveforms = self._load_waveforms()#[self.sorted_idxs, :, :]
        assert self.waveforms.shape[0] == ns

        self.amplitudes = self._load_amplitudes()#[self.sorted_idxs]
        assert self.amplitudes.shape == (ns,), '{} {}'.format(self.amplitudes.shape, (ns,))

        # TODO load positino from params
        n_chan = len(self.chx.channel_ids)
        ch_pos = np.zeros((n_chan, 2))
        ch_pos[:,1] = np.arange(n_chan)
        self.channel_positions =  ch_pos

    def get_metadata(self, name):
        return None

    def get_waveforms(self, spike_ids, channel_ids):
        wf = self.waveforms
        return wf[spike_ids, :, :][:, :, channel_ids] # TODO wierd fix

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

    def _load_waveforms(self): # TODO this should be masks for memory saving
        logger.debug("Loading spike waveforms.")
        wfs = np.vstack([sptr.waveforms for sptr in self.sptrs])
        num_spikes, num_chans, samples_per_spike = wfs.shape
        assert wfs.shape[1:] == self.sptrs[0].waveforms.shape[1:]
        wfs = wfs.reshape(num_spikes, samples_per_spike, num_chans)
        return wfs

    def _load_amplitudes(self):
        logger.debug("Loading spike amplitudes.")
        left_sweep = 0.2 * pq.ms # TODO select left sweep
        # TODO multiple sampling rates is not allowed
        mask = int(self.sptrs[0].sampling_rate.rescale('Hz') *
                   left_sweep.rescale('s'))
        logger.debug(mask)
        # TODO select channel
        out = self.waveforms[:, mask, 0]
        # HACK sometimes out is shape (n_spikes, 1)
        return np.reshape(out, len(out))
