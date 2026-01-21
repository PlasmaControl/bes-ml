# bes_ml2/elm_prediction_datamodule.py

import dataclasses
import os
import typing

import numpy as np
import h5py
import torch

from lightning.pytorch import LightningDataModule

# Reuse your existing datasets + utilities from the velocimetry module
from bes_ml2.velocimetry_datamodule import (
    Velocimetry_Datamodule,
    TrainValTest_Dataset,
    PredictDataset,
)

@dataclasses.dataclass(eq=False)
class ELM_Prediction_Datamodule(Velocimetry_Datamodule):
    """
    Datamodule for ELM probability prediction using BES windows.

    HDF5 structure expected:
      elms/<event_id>/
          attrs: shot, t_start, t_stop (=t_ELM), ...
          bes_signals: (64, T)
          bes_time:    (T,)   [ms]
    """

    # ---- where the events live in the file ----
    elms_group: str = "elms"

    # ---- label windows relative to t_ELM (ms) ----
    # P_ELM = 1.0 if  t_ELM-2.56 < t < t_ELM-0.5
    pos_window_ms: typing.Tuple[float, float] = (-2.56, -0.5)
    # P_ELM = 0.5 if  t_ELM-7.0  < t < t_ELM-5.0
    mid_window_ms: typing.Tuple[float, float] = (-7.0, -5.0)
    # P_ELM = 0.0 if  t_ELM+1.2  < t < t_ELM+5.5
    neg_window_ms: typing.Tuple[float, float] = (1.2, 5.5)

    # If True: only keep windows where ALL W samples are inside a labeled region.
    # If False: only require the *window end time* to be labeled.
    label_requires_full_window: bool = True

    # If your ELM file doesn't have per-shot inboard_column_channel_order attrs,
    # we fall back to row-major channel->(8x8) mapping.
    channel_layout_fallback: str = "row_major"  # "row_major" | "error"

    def setup(self, stage=None):
        print(f"Running {self.__class__.__name__}.setup(stage={stage})")
        return super().setup(stage=stage)

    # -------------------------------------------------------------------------
    # Event discovery / timing
    # -------------------------------------------------------------------------
    def _load_events(self):
        """
        NEW FORMAT:
        <event_id>/
            @shot
            labels (T,)
            signals (64,T)
            time   (T,)

        Return list of (shot, event_id) tuples.
        """
        import h5py

        events = []
        with h5py.File(self.data_file, "r") as h5:
            for event_id in h5.keys():
                obj = h5.get(event_id)
                if not isinstance(obj, h5py.Group):
                    continue

                # require the new datasets to exist
                if not all(k in obj for k in ("signals", "time", "labels")):
                    continue

                shot = obj.attrs.get("shot", -1)
                events.append((int(shot), str(event_id)))

        if len(events) == 0:
            raise RuntimeError(
                f"No valid event groups found in {self.data_file}. "
                "Expected top-level groups containing signals/time/labels."
            )
        return events
    
    def get_time_for_index(self, shot_event_tuple):
        """
        Used for balanced chunking. Returns number of time samples in this event.
        """
        import h5py
        _shot, event = shot_event_tuple
        with h5py.File(self.data_file, "r") as h5:
            g = h5[str(event)]
            return int(g["signals"].shape[1])  # (64, T)



    # def _load_events(self):
    #     """
    #     Return list of (shot, event_id) tuples. shot is read from attrs.
    #     """
    #     events = []
    #     with h5py.File(self.data_file, "r") as h5:
    #         if self.elms_group not in h5:
    #             raise KeyError(f"Missing group '{self.elms_group}' in {self.data_file}")

    #         for event_id in h5[self.elms_group].keys():
    #             grp = h5[self.elms_group][event_id]
    #             shot = grp.attrs.get("shot", None)
    #             if shot is None:
    #                 # If shot is missing, still allow but keep a sentinel
    #                 shot = -1
    #             events.append((int(shot), str(event_id)))

    #     if len(events) == 0:
    #         raise RuntimeError(f"No events found under '{self.elms_group}' in {self.data_file}")
    #     return events

    # def get_time_for_index(self, shot_event_tuple):
    #     """
    #     Used for balanced chunking. Returns the number of time samples in this event.
    #     """
    #     _shot, event = shot_event_tuple
    #     with h5py.File(self.data_file, "r") as h5:
    #         g = h5[f"{self.elms_group}/{event}"]
    #         return int(g["bes_signals"].shape[1])

    # -------------------------------------------------------------------------
    # Core: label construction
    # -------------------------------------------------------------------------
    def _make_p_elm_labels(self, t_ms: np.ndarray, t_elm_ms: float) -> np.ndarray:
        """
        Build label vector aligned to sample times t_ms (ms).
        Returns float32 array with NaN where unlabeled.
        """
        dt = t_ms.astype(np.float32) - np.float32(t_elm_ms)  # ms relative to ELM
        out = np.full_like(dt, np.nan, dtype=np.float32)

        a, b = self.neg_window_ms
        m = (dt > a) & (dt < b)
        out[m] = np.float32(0.0)

        a, b = self.mid_window_ms
        m = (dt > a) & (dt < b)
        out[m] = np.float32(0.5)

        a, b = self.pos_window_ms
        m = (dt > a) & (dt < b)
        out[m] = np.float32(1.0)

        return out

    def _load_and_preprocess_data(self, event_ids, dataset_stage: str):
        """
        New-format loader for ELM probability regression.

        HDF5 layout:
        <event_id>/
            @shot: int
            labels: int8 (T,)
            signals: float64 (64,T)
            time: float64 (T,)

        We build 16-channel input by taking even channels 34-64 (1-indexed):
        ch_idx = np.arange(33, 65, 2)  # 0-index
        sig16  = signals[ch_idx, :]

        We infer t_on (≈ old t_stop) from the second occurrence of labels==1:
        idx_on = where(labels==1)[1] - 1
        t_on   = time_ms[idx_on]

        Then generate piecewise labels in three time ranges relative to t_on:
        (t_on-2.5, t_on-0.5) -> 1.0
        (t_on+1.2, t_on+4.5) -> 0.0
        (t_on-7.0, t_on-5.0) -> 0.2
        Outside these ranges label = NaN (so we can ignore).

        Returns:
        - if train and split_train_data_per_gpu: dataset
        - else stores in self.datasets[dataset_stage]
        """
        import numpy as np, h5py, time, torch
        from scipy.signal import decimate, resample_poly
        from fractions import Fraction

        t_start_wall = time.time()
        print(f"[{dataset_stage}] load+preprocess (NEW ELM format, target_fs={self.target_sampling_hz} Hz)")

        # ---------------- helpers ----------------
        def _guess_times_ms(times: np.ndarray) -> np.ndarray:
            """Convert to ms if needed; otherwise pass through."""
            times = np.asarray(times, dtype=np.float64)
            if times.size < 2:
                return times.astype(np.float32)
            dt = np.median(np.diff(times))

            # Heuristic 1: already ms (dt ~ 1e-3 for 1 MHz stored in ms)
            # Heuristic 2: seconds at 1 MHz (dt ~ 1e-6), convert -> ms
            if 5e-7 <= dt <= 5e-6:
                return (times * 1000.0).astype(np.float32)  # s -> ms

            # Sometimes stored in microseconds
            if 0.5 <= dt <= 2.0 and times.max() > 5e3:
                return (times / 1000.0).astype(np.float32)  # us -> ms

            return times.astype(np.float32)

        def _infer_fs_hz(t_ms: np.ndarray, fallback: float) -> float:
            if t_ms.size < 2:
                return float(fallback)
            dt_ms = float(np.median(np.diff(t_ms)))
            if dt_ms <= 0:
                return float(fallback)
            return float(1000.0 / dt_ms)

        def _downsample_1d(arr, orig_fs, target_fs, axis=0):
            if orig_fs == target_fs:
                return arr
            ratio = orig_fs / target_fs
            q = int(round(ratio))
            if abs(ratio - q) < 1e-6 and q >= 2:
                return decimate(arr, q, ftype="fir", axis=axis, zero_phase=True)
            frac = Fraction(target_fs / orig_fs).limit_denominator(64)
            return resample_poly(arr, up=frac.numerator, down=frac.denominator, axis=axis)

        def _downsample_stack(sig_trc, t_ms, orig_fs, target_fs):
            # sig_trc: (T,R,C)
            x = _downsample_1d(sig_trc, orig_fs, target_fs, axis=0)
            new_len = x.shape[0]
            t = np.linspace(t_ms[0], t_ms[-1], new_len, dtype=np.float32)
            return x.astype(np.float32), t

        def _labels_from_ranges(t_ms: np.ndarray, t_on_ms: float):
            # Your requested ranges/values
            ranges = [
                (t_on_ms - 2.5, t_on_ms - 0.5, 1.0),  # pre-ELM
                (t_on_ms + 1.2, t_on_ms + 4.5, 0.0),  # post-ELM
                (t_on_ms - 7.0, t_on_ms - 5.0, 0.2),  # early pre-ELM
            ]
            y = np.full_like(t_ms, np.nan, dtype=np.float32)
            for t0, t1, v in ranges:
                m = (t_ms >= np.float32(t0)) & (t_ms <= np.float32(t1))
                y[m] = np.float32(v)
            return y

        def _window_indices(valid_mask: np.ndarray, W: int, hop: int = 1) -> np.ndarray:
            T = valid_mask.size
            cs = np.cumsum(valid_mask.astype(np.int32))

            def wsum(t0):
                start = t0 - W + 1
                if start <= 0:
                    return cs[t0]
                return cs[t0] - cs[start - 1]

            t0s = []
            for t0 in range(W - 1, T, hop):
                if wsum(t0) == W:
                    t0s.append(t0)
            return np.array(t0s, dtype=np.int64)

        def _get_stats_over_windows(sig_trc: np.ndarray, sample_idx: np.ndarray, W: int):
            T, R, C = sig_trc.shape
            idx = sample_idx
            if idx.size > 20000:
                idx = idx[np.linspace(0, idx.size - 1, 20000, dtype=int)]
            acc_sum = 0.0
            acc_sq = 0.0
            count = 0
            for i_t0 in idx:
                s = int(i_t0 - W + 1)
                e = int(i_t0 + 1)
                if s < 0 or e > T:
                    continue
                x = sig_trc[s:e]
                acc_sum += x.sum()
                acc_sq += np.square(x).sum()
                count += x.size
            mean = acc_sum / max(count, 1)
            var = acc_sq / max(count, 1) - mean * mean
            return float(mean), float(max(var, 0.0) ** 0.5)

        # ---------------- selections ----------------
        W      = int(self.signal_window_size)
        hop    = int(getattr(self, "window_hop", 1))
        tgt_fs = float(self.target_sampling_hz)
        fallback_orig_fs = float(getattr(self, "sampling_frequency_hz", 1_000_000.0))

        # fixed 16ch -> reshape to 4x4
        R_sel, C_sel = 4, 4
        ch_idx = np.arange(33, 65, 2)  # 0-indexed selection

        all_sig, all_lbl, all_t = [], [], []
        all_shot, all_ev = [], []
        all_sample_idx = []
        base = 0
        used = 0

        with h5py.File(self.data_file, "r") as h5:
            print(f"[{dataset_stage}] got {len(event_ids)} events; head={event_ids[:3]}")

            for item in event_ids:
                # item can be either (shot,event) or just event
                if isinstance(item, (tuple, list)) and len(item) == 2:
                    shot_hint, ev = item
                else:
                    shot_hint, ev = None, item

                k = str(ev)  # <-- top-level group name like "10116"
                if k not in h5:
                    print(f"  missing group '{k}' (from item={item})")
                    continue

                g = h5[k]
                if "signals" not in g or "time" not in g or "labels" not in g:
                    print(f"  {k} missing one of: signals/time/labels")
                    continue

                # choose shot id: prefer attr, fall back to tuple hint
                shot_id = int(g.attrs.get("shot", shot_hint if shot_hint is not None else -1))

                sig64 = np.array(g["signals"], dtype=np.float32)  # (64,T)
                tt    = np.array(g["time"],    dtype=np.float64)  # (T,)
                lab0  = np.array(g["labels"],  dtype=np.int8)     # (T,)

                if sig64.ndim != 2 or sig64.shape[0] != 64:
                    print(f"  {k} bad signals shape {sig64.shape} (expected (64,T))")
                    continue
                if tt.size < W or sig64.shape[1] < W:
                    continue

                # Convert times to ms early (used for t_on and cropping)
                t_ms = _guess_times_ms(tt)

                # --- t_on from labels (your exact rule, with safety) ---
                ones = np.where(lab0 == 1)[0]
                if ones.size == 0:
                    print(f"  {k} has no labels==1; skipping")
                    continue
                idx_on = int(ones[1] - 1) if ones.size >= 2 else int(ones[0] - 1)
                idx_on = max(idx_on, 0)
                if idx_on >= t_ms.size:
                    print(f"  {k} idx_on out of bounds; skipping")
                    continue
                t_on = float(t_ms[idx_on])

                # --- select 16 channels ---
                sig16 = sig64[ch_idx, :]  # (16,T)

                # reshape to (T,4,4)
                T = sig16.shape[1]
                sig_trc = np.transpose(sig16.reshape(4, 4, T), (2, 0, 1)).astype(np.float32)

                # Optional absolute time cropping (pre-downsample)
                if getattr(self, "start_time_ms", None) is not None or getattr(self, "end_time_ms", None) is not None:
                    m = np.ones_like(t_ms, dtype=bool)
                    if getattr(self, "start_time_ms", None) is not None:
                        m &= (t_ms >= float(self.start_time_ms))
                    if getattr(self, "end_time_ms", None) is not None:
                        m &= (t_ms <= float(self.end_time_ms))
                    sig_trc = sig_trc[m]
                    t_ms = t_ms[m]
                    if sig_trc.shape[0] < W:
                        continue

                # infer orig fs and downsample
                orig_fs = _infer_fs_hz(t_ms, fallback=fallback_orig_fs)
                sig_ds, t_ds = _downsample_stack(sig_trc, t_ms, orig_fs, tgt_fs)

                # labels from ranges at downsampled times
                lbl = _labels_from_ranges(t_ds, t_on_ms=t_on)

                # window sampling: require label at end (and optionally whole window labeled)
                label_requires_full_window = bool(getattr(self, "label_requires_full_window", True))
                if label_requires_full_window:
                    valid = ~np.isnan(lbl)
                    sample_idx_ev = _window_indices(valid, W, hop=hop)
                else:
                    t0s = np.arange(W - 1, t_ds.size, hop, dtype=np.int64)
                    sample_idx_ev = t0s[~np.isnan(lbl[t0s])]

                if sample_idx_ev.size == 0:
                    continue

                # shot/event id per time point
                shot_id = int(g.attrs.get("shot", -1))
                shot_vec = np.full(t_ds.shape[0], shot_id, dtype=np.int32)
                ev_vec   = np.full(t_ds.shape[0], k, dtype=object)

                all_sig.append(sig_ds)
                all_lbl.append(lbl)
                all_t.append(t_ds)
                all_shot.append(shot_vec)
                all_ev.append(ev_vec)

                all_sample_idx.append(sample_idx_ev + base)
                base += t_ds.size
                used += 1

        if used == 0:
            raise RuntimeError("No usable events found in new-format file.")

        signals_trc = np.concatenate(all_sig, axis=0).astype(np.float32)  # (T_total,4,4)
        labels      = np.concatenate(all_lbl, axis=0).astype(np.float32)  # (T_total,)
        times_ms    = np.concatenate(all_t,   axis=0).astype(np.float32)  # (T_total,)
        shot_ids_t  = np.concatenate(all_shot, axis=0).astype(np.int32)   # (T_total,)
        event_ids_t = np.concatenate(all_ev,   axis=0).astype(object)     # (T_total,)
        sample_idx  = np.concatenate(all_sample_idx, axis=0).astype(np.int64)

        if sample_idx.size == 0:
            raise RuntimeError("No valid sample windows after filtering.")

        # ---------------- standardize (train stats) ----------------
        if getattr(self, "signal_mean", None) is None or getattr(self, "signal_stdev", None) is None:
            if dataset_stage != "train":
                print("[warn] signal_mean/stdev missing but stage != train; computing anyway.")
            m, s = _get_stats_over_windows(signals_trc, sample_idx, W)
            self.signal_mean, self.signal_stdev = float(m), float(s)
            # if you store hparams somewhere:
            if hasattr(self, "save_hyperparameters"):
                try:
                    self.save_hyperparameters({"signal_mean": float(m), "signal_stdev": float(s)})
                except Exception:
                    pass

        if dataset_stage in ["train", "validation", "test"] and bool(getattr(self, "standardize_signals", False)):
            m = np.float32(self.signal_mean)
            s = np.float32(self.signal_stdev)
            eps = np.float32(1e-12)
            signals_trc = (signals_trc - m) / np.maximum(s, eps)

        # ---------------- optional flip aug (label unchanged) ----------------
        if bool(getattr(self, "do_flip_augmentation", False)):
            T0 = signals_trc.shape[0]
            flipped = signals_trc[:, ::-1, :]  # flip R axis
            signals_trc = np.concatenate([signals_trc, flipped], axis=0)
            labels      = np.concatenate([labels, labels], axis=0)
            times_ms    = np.concatenate([times_ms, times_ms], axis=0)
            shot_ids_t  = np.concatenate([shot_ids_t, shot_ids_t], axis=0)
            event_ids_t = np.concatenate([event_ids_t, event_ids_t], axis=0)
            sample_idx  = np.concatenate([sample_idx, sample_idx + T0], axis=0)

        # ---------------- Train/Val/Test dataset (3-tuple) ----------------
        ds = TrainValTest_Dataset(
            signals_trc=signals_trc,      # (T_total, 4, 4)
            labels_scalar=labels,         # (T_total,)
            sample_indices=sample_idx,    # (N,)
            signal_window_size=W,         # window length
            times_ms=times_ms,            # (T_total,)
            n_rows_sel=R_sel,             # 4
            n_cols_sel=C_sel,             # 4
        )

        if dataset_stage == "train" and bool(getattr(self, "split_train_data_per_gpu", False)):
            print(f"[{dataset_stage}] built dataset in {time.time() - t_start_wall:.2f}s: "
                f"T={signals_trc.shape[0]} windows={sample_idx.size}")
            return ds
        else:
            self.datasets[dataset_stage] = ds
            print(f"[{dataset_stage}] built dataset in {time.time() - t_start_wall:.2f}s: "
                f"T={signals_trc.shape[0]} windows={sample_idx.size}")
            return None


    # # -------------------------------------------------------------------------
    # # Core: data loading for train/val/test
    # # -------------------------------------------------------------------------
    # def _load_and_preprocess_data(self, shot_event_indices, dataset_stage: str):
    #     """
    #     Builds block inputs (rows x cols) and scalar P_ELM labels.
    #     Windows are (1, W, R_sel, C_sel); label is P_ELM at the window end time.

    #     Keeps the same preprocessing pipeline as Velocimetry_Datamodule:
    #     - reshape_signals -> pad_to_full_grid -> sub-grid select
    #     - optional bandpass
    #     - time cropping (start_time_ms/end_time_ms)
    #     - downsample to target_sampling_hz
    #     - standardize with train mean/std
    #     - windowing via sample_indices
    #     """
    #     from scipy.signal import decimate, resample_poly
    #     from fractions import Fraction
    #     import time

    #     t_start_wall = time.time()
    #     print(f"[{dataset_stage}] load+preprocess (ELM labels, target_fs={self.target_sampling_hz} Hz)")

    #     # -------- helpers (scoped, copied style from your velo DM) --------
    #     def _guess_times_ms(times: np.ndarray) -> np.ndarray:
    #         # Your ELM file looks like ms already (dt ~ 0.001 ms). Keep the same robust logic.
    #         if times.size < 2:
    #             return times.astype(np.float32)
    #         dt = np.median(np.diff(times))
    #         # If stored in microseconds (dt≈1) and large absolute values, convert -> ms.
    #         if 0.5 <= dt <= 2.0 and times.max() > 5e3:
    #             return (times / 1000.0).astype(np.float32)
    #         return times.astype(np.float32)

    #     def _downsample_1d(arr, orig_fs, target_fs, axis=0):
    #         if orig_fs == target_fs:
    #             return arr
    #         ratio = orig_fs / target_fs
    #         q = int(round(ratio))
    #         if abs(ratio - q) < 1e-6 and q >= 2:
    #             return decimate(arr, q, ftype="fir", axis=axis, zero_phase=True)
    #         frac = Fraction(target_fs / orig_fs).limit_denominator(64)
    #         return resample_poly(arr, up=frac.numerator, down=frac.denominator, axis=axis)

    #     def _downsample_stack(sig_trc, t_ms, orig_fs, target_fs):
    #         x = _downsample_1d(sig_trc, orig_fs, target_fs, axis=0)
    #         new_len = x.shape[0]
    #         t = np.linspace(t_ms[0], t_ms[-1], new_len, dtype=np.float32)
    #         return x.astype(np.float32), t

    #     def _window_indices(valid_mask: np.ndarray, W: int, hop: int = 1) -> np.ndarray:
    #         T = valid_mask.size
    #         cs = np.cumsum(valid_mask.astype(np.int32))

    #         def wsum(t0):
    #             start = t0 - W + 1
    #             if start <= 0:
    #                 return cs[t0]
    #             return cs[t0] - cs[start - 1]

    #         t0s = []
    #         for t0 in range(W - 1, T, hop):
    #             if wsum(t0) == W:
    #                 t0s.append(t0)
    #         return np.array(t0s, dtype=np.int64)

    #     def _mask_from_windows_dict(shot: str, t_ms: np.ndarray, windows_dict):
    #         if not windows_dict:
    #             return np.ones_like(t_ms, dtype=bool)

    #         wins = windows_dict.get(shot, None)
    #         if wins is None:
    #             wins = windows_dict.get("all", windows_dict.get("*", None))
    #         if wins is None or len(wins) == 0:
    #             return np.ones_like(t_ms, dtype=bool)

    #         if len(wins) == 1 and (wins[0] == () or wins[0] == (None, None)):
    #             return np.ones_like(t_ms, dtype=bool)

    #         mask = np.zeros_like(t_ms, dtype=bool)
    #         for w in wins:
    #             if w == () or w is None:
    #                 return np.ones_like(t_ms, dtype=bool)
    #             t0, t1 = w
    #             if t0 is None:
    #                 t0 = -np.inf
    #             if t1 is None:
    #                 t1 = np.inf
    #             mask |= (t_ms >= t0) & (t_ms <= t1)
    #         return mask

    #     def _get_stats_over_windows(sig_trc: np.ndarray, sample_idx: np.ndarray, W: int):
    #         T, R, C = sig_trc.shape
    #         idx = sample_idx
    #         if idx.size > 20000:
    #             idx = idx[np.linspace(0, idx.size - 1, 20000, dtype=int)]
    #         acc_sum = 0.0
    #         acc_sq = 0.0
    #         count = 0
    #         for i_t0 in idx:
    #             s = int(i_t0 - W + 1)
    #             e = int(i_t0 + 1)
    #             if s < 0 or e > T:
    #                 continue
    #             x = sig_trc[s:e]
    #             acc_sum += x.sum()
    #             acc_sq += np.square(x).sum()
    #             count += x.size
    #         mean = acc_sum / max(count, 1)
    #         var = acc_sq / max(count, 1) - mean * mean
    #         return float(mean), float(max(var, 0.0) ** 0.5)

    #     # -------- selections --------
    #     W = int(self.signal_window_size)
    #     hop = int(self.window_hop)
    #     orig_fs = float(self.sampling_frequency_hz)  # typically 1e6
    #     tgt_fs = float(self.target_sampling_hz)

    #     row_idx = np.arange(8)[self.row_offset :: self.row_stride]
    #     col_idx = self._cols_from_spec(self.block_cols, n=8)
    #     R_sel, C_sel = row_idx.size, col_idx.size

    #     all_sig, all_lbl, all_t = [], [], []
    #     all_sample_idx = []
    #     base = 0
    #     used_events = 0

    #     with h5py.File(self.data_file, "r") as h5:
    #         if len(shot_event_indices) >= 5:
    #             print("  head events:", shot_event_indices[:5])

    #         for shot, event in shot_event_indices:
    #             gkey = f"{self.elms_group}/{event}"
    #             if gkey not in h5:
    #                 print(f"  missing {gkey}")
    #                 continue

    #             grp = h5[gkey]
    #             if "bes_signals" not in grp or "bes_time" not in grp:
    #                 print(f"  {gkey} missing 'bes_signals' or 'bes_time'")
    #                 continue

    #             # t_ELM from attrs
    #             t_elm = grp.attrs.get("t_stop", None)
    #             if t_elm is None:
    #                 print(f"  {gkey} missing attr 't_stop' (t_ELM)")
    #                 continue
    #             t_elm = float(t_elm)

    #             sig_64t = np.array(grp["bes_signals"], dtype=np.float32)  # (X, T) where X is 64 or 16
    #             times  = np.array(grp["bes_time"], dtype=np.float64)      # (T,)

    #             if sig_64t.shape[1] < W or times.size < W:
    #                 continue

    #             X, T = sig_64t.shape
    #             t_ms = _guess_times_ms(times)

    #             # ------------------------------------------------------------
    #             # Build sig_trc as (T, R, C) and handle 64-ch vs 16-ch cleanly
    #             # ------------------------------------------------------------
    #             if X == 64:
    #                 # Try to get inboard order like velocimetry; else fall back row-major.
    #                 order_attr = None
    #                 if "inboard_column_channel_order" in grp.attrs:
    #                     order_attr = grp.attrs["inboard_column_channel_order"]
    #                 elif "inboard_column_channel_order" in h5.attrs:
    #                     order_attr = h5.attrs["inboard_column_channel_order"]

    #                 if order_attr is not None:
    #                     sig_trc = self.reshape_signals(sig_64t, order_attr)  # (T, R_full, C_full)
    #                 else:
    #                     if self.channel_layout_fallback != "row_major":
    #                         raise ValueError(
    #                             f"{gkey}: missing inboard_column_channel_order and fallback={self.channel_layout_fallback}"
    #                         )
    #                     sig_rct = sig_64t.reshape(8, 8, T)          # (8, 8, T)
    #                     sig_trc = np.transpose(sig_rct, (2, 0, 1))  # (T, 8, 8)

    #                 # Pad to 8x8 (in case reshape_signals returned 7x8, etc.)
    #                 sig_trc = self._pad_to_full_grid(sig_trc, target_R=8, target_C=8)

    #                 # Sub-grid select into (T, R_sel, C_sel)
    #                 sig_trc = sig_trc[:, row_idx[:, None], col_idx[None, :]]

    #             elif X == 16:
    #                 # Already a 4x4 selected block stored directly.
    #                 # IMPORTANT: do NOT pad to 8x8 and re-select bottom rows/odd cols.
    #                 sig_rct = sig_64t.reshape(4, 4, T)          # (4, 4, T)
    #                 sig_trc = np.transpose(sig_rct, (2, 0, 1))  # (T, 4, 4)

    #                 # Sanity check it matches the model’s expected (R_sel, C_sel)
    #                 if sig_trc.shape[1] != R_sel or sig_trc.shape[2] != C_sel:
    #                     raise ValueError(
    #                         f"{gkey}: X==16 implies a (T,{R_sel},{C_sel}) block, but got {sig_trc.shape}. "
    #                         f"Check row_offset/row_stride/block_cols vs how the 16 channels were saved."
    #                     )

    #             else:
    #                 raise ValueError(f"{gkey}: expected 64 or 16 channels, got {X}")

    #             # Optional bandpass
    #             if self.lower_cutoff_frequency_hz is not None and self.upper_cutoff_frequency_hz is not None:
    #                 sig_trc = self.apply_bandpass_filter(sig_trc)

    #             # Absolute time cropping (pre-downsample)
    #             time_mask = np.ones_like(t_ms, dtype=bool)
    #             if self.start_time_ms is not None:
    #                 time_mask &= (t_ms >= self.start_time_ms)
    #             if self.end_time_ms is not None:
    #                 time_mask &= (t_ms <= self.end_time_ms)
    #             sig_trc = sig_trc[time_mask]
    #             t_ms = t_ms[time_mask]
    #             if sig_trc.shape[0] < W:
    #                 continue

    #             # Downsample to target rate
    #             sig_ds, t_ds = _downsample_stack(sig_trc, t_ms, orig_fs, tgt_fs)

    #             # Label vector at downsampled times
    #             lbl = self._make_p_elm_labels(t_ds, t_elm_ms=t_elm)

    #             # Shot window masks (kept compatible with your existing knobs)
    #             shot_mask = np.ones_like(t_ds, dtype=bool)
    #             if getattr(self, "shot_time_windows", None) and str(shot) in self.shot_time_windows:
    #                 shot_mask = np.zeros_like(t_ds, dtype=bool)
    #                 for t0, t1 in self.shot_time_windows[str(shot)]:
    #                     shot_mask |= (t_ds >= t0) & (t_ds <= t1)

    #             # Train-only extra restriction
    #             if dataset_stage == "train":
    #                 shot_mask &= _mask_from_windows_dict(str(shot), t_ds, getattr(self, "train_time_windows", None))

    #             # Decide which window end-times are allowed
    #             if self.label_requires_full_window:
    #                 valid = shot_mask & ~np.isnan(lbl)
    #                 sample_idx_ev = _window_indices(valid, W, hop=hop)
    #             else:
    #                 # only require window end time to be labeled; still respect shot_mask at t0
    #                 t0s = np.arange(W - 1, t_ds.shape[0], hop, dtype=np.int64)
    #                 ok = shot_mask[t0s] & ~np.isnan(lbl[t0s])
    #                 sample_idx_ev = t0s[ok]

    #             if sample_idx_ev.size == 0:
    #                 continue

    #             all_sig.append(sig_ds)
    #             all_lbl.append(lbl)
    #             all_t.append(t_ds)

    #             all_sample_idx.append(sample_idx_ev + base)
    #             base += t_ds.shape[0]
    #             used_events += 1

    #     if used_events == 0:
    #         raise RuntimeError("No usable ELM events for the requested configuration.")

        # signals_tc_rc = np.concatenate(all_sig, axis=0).astype(np.float32)  # (T_total, R_sel, C_sel)
        # labels_scalar = np.concatenate(all_lbl, axis=0).astype(np.float32)  # (T_total,)
        # times_ms = np.concatenate(all_t, axis=0).astype(np.float32)         # (T_total,)
        # sample_idx = np.concatenate(all_sample_idx, axis=0).astype(np.int64)

        # if sample_idx.size == 0:
        #     raise RuntimeError("No valid sample windows after filtering.")

        # # Stats & standardize
        # if self.signal_mean is None or self.signal_stdev is None:
        #     assert dataset_stage == "train" or not getattr(self, "train_events", True), f"Dataset_stage: {dataset_stage}"
        #     m, s = _get_stats_over_windows(signals_tc_rc, sample_idx, W)
        #     self.signal_mean, self.signal_stdev = float(m), float(s)
        #     self.save_hyperparameters({"signal_mean": float(m), "signal_stdev": float(s)})

        # if dataset_stage in ["train", "validation", "test"] and self.standardize_signals:
        #     m = np.float32(self.signal_mean)
        #     s = np.float32(self.signal_stdev)
        #     eps = np.float32(1e-12)
        #     print(f"  standardizing: mean={float(m):.4f}, std={float(s):.4f}")
        #     signals_tc_rc = (signals_tc_rc - m) / np.maximum(s, eps)

        # # Flip augmentation: for ELM probability, label should NOT change.
        # if self.do_flip_augmentation:
        #     T0 = signals_tc_rc.shape[0]
        #     flipped = signals_tc_rc[:, ::-1, :]
        #     signals_tc_rc = np.concatenate([signals_tc_rc, flipped], axis=0)
        #     labels_scalar = np.concatenate([labels_scalar, labels_scalar], axis=0)
        #     times_ms = np.concatenate([times_ms, times_ms], axis=0)
        #     sample_idx = np.concatenate([sample_idx, sample_idx + T0], axis=0)

        # ds = TrainValTest_Dataset(
        #     signals_trc=signals_tc_rc,
        #     labels_scalar=labels_scalar,
        #     sample_indices=sample_idx,
        #     signal_window_size=W,
        #     times_ms=times_ms,
        #     n_rows_sel=R_sel,
        #     n_cols_sel=C_sel,
        # )

        # if dataset_stage == "train" and self.split_train_data_per_gpu:
        #     return ds
        # else:
        #     self.datasets[dataset_stage] = ds
        #     print(f"[{dataset_stage}] built dataset in {time.time() - t_start_wall:.2f}s: "
        #           f"T={signals_tc_rc.shape[0]} windows={sample_idx.size}")
        #     return None

    # # -------------------------------------------------------------------------
    # # Predict dataset
    # # -------------------------------------------------------------------------
    # def _load_and_preprocess_predict_data(self, shot_event_indices):
    #     """
    #     Prepare prediction samples as sliding windows from (R_sel x C_sel) block.
    #     Each item: (window[1,W,R_sel,C_sel], label_scalar, time_ms, shot_id, event_id)

    #     Here label_scalar is computed from t_stop if present; otherwise NaN.
    #     """
    #     from scipy.signal import decimate, resample_poly
    #     from fractions import Fraction
    #     import time

    #     t_start_wall = time.time()

    #     def _guess_times_ms(times: np.ndarray) -> np.ndarray:
    #         if times.size < 2:
    #             return times.astype(np.float32)
    #         dt = np.median(np.diff(times))
    #         if 0.5 <= dt <= 2.0 and times.max() > 5e3:
    #             return (times / 1000.0).astype(np.float32)
    #         return times.astype(np.float32)

    #     def _downsample_1d(arr, orig_fs, target_fs, axis=0):
    #         if orig_fs == target_fs:
    #             return arr
    #         ratio = orig_fs / target_fs
    #         q = int(round(ratio))
    #         if abs(ratio - q) < 1e-6 and q >= 2:
    #             return decimate(arr, q, ftype="fir", axis=axis, zero_phase=True)
    #         frac = Fraction(target_fs / orig_fs).limit_denominator(64)
    #         return resample_poly(arr, up=frac.numerator, down=frac.denominator, axis=axis)

    #     def _downsample_stack(sig_trc, t_ms, orig_fs, target_fs):
    #         x = _downsample_1d(sig_trc, orig_fs, target_fs, axis=0)
    #         new_len = x.shape[0]
    #         t = np.linspace(t_ms[0], t_ms[-1], new_len, dtype=np.float32)
    #         return x.astype(np.float32), t

    #     W = int(self.signal_window_size)
    #     hop = int(getattr(self, "predict_window_stride", 1))

    #     row_idx = np.arange(8)[self.row_offset :: self.row_stride]
    #     col_idx = self._cols_from_spec(self.block_cols, n=8)
    #     R_sel, C_sel = row_idx.size, col_idx.size

    #     orig_fs = float(self.sampling_frequency_hz)
    #     tgt_fs = float(self.target_sampling_hz)

    #     win_list, lbl_list, t0_list, shot_list, event_list = [], [], [], [], []

    #     with h5py.File(self.data_file, "r") as h5:
    #         valid_events = []
    #         for shot, event in shot_event_indices:
    #             gkey = f"{self.elms_group}/{event}"
    #             if gkey in h5 and "bes_signals" in h5[gkey] and "bes_time" in h5[gkey]:
    #                 valid_events.append((shot, event))
    #             else:
    #                 print(f"  [predict] skipping {gkey}: missing datasets")

    #         for i, (shot, event) in enumerate(valid_events):
    #             if i % 100 == 0:
    #                 print(f"  [predict] {i:04d}/{len(valid_events):04d}  shot={shot}  event={event}")

    #             grp = h5[f"{self.elms_group}/{event}"]

    #             sig_64t = np.array(grp["bes_signals"], dtype=np.float32)  # (X, T) where X is 64 or 16
    #             times  = np.array(grp["bes_time"], dtype=np.float64)      # (T,)

    #             if sig_64t.shape[1] < W or times.size < W:
    #                 continue

    #             t_elm = grp.attrs.get("t_stop", None)
    #             t_elm = float(t_elm) if t_elm is not None else None

    #             X, T = sig_64t.shape
    #             t_ms = _guess_times_ms(times)

    #             # ------------------------------------------------------------
    #             # Build sig_trc as (T, R, C) and handle 64-ch vs 16-ch cleanly
    #             # ------------------------------------------------------------
    #             if X == 64:
    #                 order_attr = None
    #                 if "inboard_column_channel_order" in grp.attrs:
    #                     order_attr = grp.attrs["inboard_column_channel_order"]
    #                 elif "inboard_column_channel_order" in h5.attrs:
    #                     order_attr = h5.attrs["inboard_column_channel_order"]

    #                 if order_attr is not None:
    #                     sig_trc = self.reshape_signals(sig_64t, order_attr)  # (T, R_full, C_full)
    #                 else:
    #                     if self.channel_layout_fallback != "row_major":
    #                         raise ValueError(
    #                             f"elms/{event}: missing inboard_column_channel_order and fallback={self.channel_layout_fallback}"
    #                         )
    #                     sig_rct = sig_64t.reshape(8, 8, T)
    #                     sig_trc = np.transpose(sig_rct, (2, 0, 1))  # (T, 8, 8)

    #                 # Pad and subselect for 8x8
    #                 sig_trc = self._pad_to_full_grid(sig_trc, target_R=8, target_C=8)
    #             elif X == 16:
    #                 # Already a 4x4 selected block stored directly.
    #                 sig_rct = sig_64t.reshape(4, 4, T)
    #                 sig_trc = np.transpose(sig_rct, (2, 0, 1))  # (T, 4, 4)

    #                 if sig_trc.shape[1] != R_sel or sig_trc.shape[2] != C_sel:
    #                     raise ValueError(
    #                         f"elms/{event}: X==16 implies a (T,{R_sel},{C_sel}) block, but got {sig_trc.shape}. "
    #                         f"Check row_offset/row_stride/block_cols vs how the 16 channels were saved."
    #                     )

    #             else:
    #                 raise ValueError(f"{gkey}: expected 64 or 16 channels, got {X}")


    #             # Crop time before downsample
    #             time_mask = np.ones_like(t_ms, dtype=bool)
    #             if self.start_time_ms is not None:
    #                 time_mask &= (t_ms >= self.start_time_ms)
    #             if self.end_time_ms is not None:
    #                 time_mask &= (t_ms <= self.end_time_ms)

    #             sig_trc = sig_trc[time_mask]
    #             t_ms = t_ms[time_mask]
    #             if sig_trc.shape[0] < W:
    #                 continue

    #             # IMPORTANT: only pad/select here if we started from 64 channels.
    #             # If X==16, sig_trc is already (T, R_sel, C_sel).
    #             if X == 64:
    #                 sig_trc = sig_trc[:, row_idx[:, None], col_idx[None, :]]

    #             if self.lower_cutoff_frequency_hz is not None and self.upper_cutoff_frequency_hz is not None:
    #                 sig_trc = self.apply_bandpass_filter(sig_trc)


    #             sig_ds, t_ds = _downsample_stack(sig_trc, t_ms, orig_fs, tgt_fs)

    #             # Build labels from t_stop if available
    #             if t_elm is not None:
    #                 label_vec = self._make_p_elm_labels(t_ds, t_elm_ms=t_elm)
    #             else:
    #                 label_vec = np.full_like(t_ds, np.nan, dtype=np.float32)

    #             # Respect any shot_time_windows if provided
    #             shot_mask = np.ones_like(t_ds, dtype=bool)
    #             if getattr(self, "shot_time_windows", None) and str(shot) in self.shot_time_windows:
    #                 shot_mask = np.zeros_like(t_ds, dtype=bool)
    #                 for t0, t1 in self.shot_time_windows[str(shot)]:
    #                     shot_mask |= (t_ds >= t0) & (t_ds <= t1)

    #             # Candidate window end indices within this event
    #             t0_candidates = np.arange(W - 1, sig_ds.shape[0], hop, dtype=int)
    #             t0_candidates = t0_candidates[shot_mask[t0_candidates]]
    #             if t0_candidates.size == 0:
    #                 continue

    #             # Standardize using train stats (same as your velocimetry predict path)
    #             if self.standardize_signals:
    #                 assert self.signal_mean is not None and self.signal_stdev is not None, \
    #                     "Predict requires training signal_mean/signal_stdev to standardize inputs."
    #                 sig_ds = (sig_ds - self.signal_mean) / max(self.signal_stdev, 1e-12)

    #             for t0 in t0_candidates:
    #                 win = sig_ds[t0 - W + 1 : t0 + 1, :, :]  # (W, R_sel, C_sel)
    #                 lbl = label_vec[t0]
    #                 t0_ms = np.float32(t_ds[t0])

    #                 win_list.append(win[None, ...])  # (1, W, R_sel, C_sel)
    #                 lbl_list.append(np.float32(lbl))
    #                 t0_list.append(t0_ms)
    #                 shot_list.append(int(shot))
    #                 event_list.append(str(event))

    #     windows = np.asarray(win_list, dtype=np.float32)   # (N, 1, W, R_sel, C_sel)
    #     labels = np.asarray(lbl_list, dtype=np.float32)    # (N,)
    #     times_ms = np.asarray(t0_list, dtype=np.float32)   # (N,)
    #     shots = np.asarray(shot_list)
    #     events = np.asarray(event_list)

    #     print(f"[predict] prepared {windows.shape[0]} windows in {time.time() - t_start_wall:.2f}s. "
    #           f"per-window shape={windows.shape[1:]}")
    #     return PredictDataset(
    #         windows=windows,
    #         labels=labels,
    #         times_ms=times_ms,
    #         shots=shots,
    #         events=events,
    #     )
