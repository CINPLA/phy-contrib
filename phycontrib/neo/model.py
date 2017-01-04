
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

    def __init__(self, data_path=None, channel_group=1, segment_n=0, **kwargs):
        data_path = data_path or ''
        dir_path = (op.dirname(op.realpath(op.expanduser(data_path)))
                    if data_path else os.getcwd())
        self.data_path = data_path
        self.dir_path = dir_path
        self.__dict__.update(kwargs)
        self.channel_group = channel_group
        self.segment_n = segment_n
        self._load_data()

    def describe(self):
        def _print(name, value):
            print("{0: <24}{1}".format(name, value))

        _print('Data file', self.data_path)
        _print('Number of channels', len(self.chx.channel_ids))
        _print('Duration', '{}'.format(self.seg.duration))
        _print('Number of spikes', len(self.sptr))

    def _load_data(self):
        f = neo.AxonaIO(self.data_path)
        blk = f.read_block() # TODO params to select what to read
        self.chx = blk.channel_indexes[self.channel_group]
        self.seg = blk.segments[self.segment_n]
        self.sptr = self.chx.units[0].spiketrains[0] # assumes 0 is the mua cluster

        self.spike_times = self.sptr.times.rescale('s').magnitude
        ns, = self.n_spikes, = self.spike_times.shape

        self.amplitudes = self._load_amplitudes()
        assert self.amplitudes.shape == (ns,), "{}, {}".format(ns, self.amplitudes.shape)

        self.spike_clusters = self._load_spike_clusters()
        assert self.spike_clusters.shape == (ns,)

        n_chan = len(self.chx.channel_ids)
        ch_pos = np.zeros((n_chan, 2))
        ch_pos[:,1] = np.arange(n_chan)
        self.channel_positions =  ch_pos # TODO load positino from params

    def get_metadata(self, name):
        return None

    def get_waveforms(self, spike_ids, channel_ids):
        sptr = self.sptr
        wf = sptr.waveforms
        num_spikes, num_chans, samples_per_spike = wf.shape
        wf = wf.reshape(num_spikes, samples_per_spike, num_chans)
        out = wf[spike_ids, :, :][:, :, channel_ids] # TODO wierd fix
        return out

    def _load_amplitudes(self):
        logger.debug("Loading spike amplitudes.")
        sptr = self.sptr
        left_sweep = 0.2 * pq.ms # TODO select left sweep
        mask = int(sptr.sampling_rate.rescale('Hz') *
                   left_sweep.rescale('s'))
        out = sptr.waveforms[:, 0, mask] # TODO select channel
        return out

    def _load_spike_clusters(self):
        logger.debug("Loading spike clusters.")
        # NOTE: we make a zero cluster which is updated
        # during manual clustering.
        out = np.zeros(self.n_spikes, dtype=np.int64)
        return out
