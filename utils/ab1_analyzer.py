import json
from typing import Dict   

import numpy as np 
from scipy.signal import find_peaks

from Bio import SeqIO
from Bio.Seq import Seq
from Bio.Align import PairwiseAligner
import utils.referece_loader as rl
 

try:
    from Bio.Align import PairwiseAligner
    from Bio.Seq import Seq
    BIOPYTHON_AVAILABLE = True
except ImportError:
    BIOPYTHON_AVAILABLE = False
    PairwiseAligner = None
    print("BioPython not available. Install with: pip install biopython")

class AB1Analyzer:
    def __init__(self):
        self.ab1_file = None
        self.sequence = ""
        self.quality_scores = []
        self.trace_data = {}
        self.global_alignment_params = {
            'match_score': 2,
            'mismatch_score': -0.5,
            'gap_open_penalty': -2,
            'gap_extend_penalty': -0.5,
            'one_alignment_only': True
        }
        self.abo_reference = rl.ReferenceLoader().load_abo_reference()




    def check_orientation(self, ab1_seq):
        """Return 'forward' or 'reverse' depending on which orientation aligns better."""
        
        aligner = PairwiseAligner() # type: ignore
        aligner.mode = 'global'
        aligner.match_score = self.global_alignment_params['match_score']
        aligner.mismatch_score = self.global_alignment_params['mismatch_score']
        aligner.open_gap_score = self.global_alignment_params['gap_open_penalty']
        aligner.extend_gap_score = self.global_alignment_params['gap_extend_penalty']

        score_fwd = aligner.align(self.abo_reference['full_sequence'], ab1_seq)[0]
        score_rev = aligner.align(self.abo_reference['full_sequence'], str(Seq(ab1_seq).reverse_complement()))[0]

        orientation = "forward" if score_fwd >= score_rev else "reverse"
        best_score = max(score_fwd, score_rev)
        return orientation
    

    def read_ab1_trace(self, ab1_file):
        reverse = True
        rec = SeqIO.read(ab1_file, "abi")
        raw = rec.annotations["abif_raw"]
        seq = str(rec.seq)
        orientation = self.check_orientation(seq)
        if orientation == "reverse":
            reverse = False


        # Channel order (e.g., b'GATC'). Fallback to 'ACGT' if missing.
        fwo = raw.get("FWO_", b"ACGT").decode("ascii")
        # DATA9..12 correspond to the four dye channels in the FWO_ order
        data_keys = [9, 10, 11, 12]

        # Map channels to A/C/G/T using FWO_
        traces = {b: None for b in "ACGT"}
        for base, k in zip(fwo, data_keys):
            arr = np.array(raw[f"DATA{k}"], dtype=np.int32)
            traces[base] = arr # type: ignore

        # Some files may not include all bases in FWO_, ensure all present
        for b in "ACGT":
            if traces[b] is None:
                traces[b] = np.zeros_like(traces[fwo[0]])  # type: ignore safe filler

        pos = np.array(raw["PLOC2"], dtype=np.int32)  # peak indices (0..n-1) in trace arrays
        n = len(traces["A"]) # type: ignore
        

        if reverse:
            # Reverse the arrays in-place
            for b in "ACGT":
                traces[b] = traces[b][::-1] # type: ignore
            # Properly transform PLOC2 for the reversed arrays
            # (not just reverse order: map p -> (n-1-p) and then reverse the list)
            pos = (n - 1 - pos)[::-1]
            # Reverse-complement the basecalls
            seq = str(rec.seq.reverse_complement())
        trace_data = traces
        trace_data["pos"] = pos # type: ignore
        trace_data["seq"] = seq    # type: ignore

        return {
            "A": traces["A"],
            "C": traces["C"],
            "G": traces["G"],
            "T": traces["T"],
            "pos": pos,   # valid indices into A/C/G/T arrays
            "seq": seq
        }

    def reverse_chromatogram(self, trace_data):
        """Reverse-complement a merged chromatogram trace."""
        if not trace_data or trace_data["A"] is None:
            raise ValueError("Trace data not loaded. Call read_ab1_trace() first.")
        n = len(trace_data["A"])
        rev = {
            "A": trace_data["T"][::-1],# type: ignore
            "C": trace_data["G"][::-1],# type: ignore
            "G": trace_data["C"][::-1],# type: ignore
            "T": trace_data["A"][::-1],# type: ignore
            "pos": (n - 1 - trace_data["pos"])[::-1],  # remap valid indices! # type: ignore
            "seq": str(Seq(trace_data["seq"]).reverse_complement()),
            "overlap_bases": trace_data.get("overlap_bases", None),
            "offset": trace_data.get("offset", None)
        }
        return rev

    def normalize_trace(self):
        norm = {}
        for base in "ACGT":
            sig = np.array(self.trace_data[base], dtype=float)
            denom = np.percentile(sig, 99) + 1e-6
            norm[base] = sig / denom
        norm["pos"] = self.trace_data["pos"]
        norm["seq"] = self.trace_data["seq"]
        return norm

    import numpy as np

    def merge_overlap(self, trace1, trace2, min_overlap=20):
        """
        Merge two overlapping AB1 chromatograms into one (same-direction reads).
        - Detect base overlap via called sequences.
        - Convert base overlap to *sample* offset using PLOC2.
        - Merge signals with padding if needed.
        - Merge true peak positions (PLOC2) into merged sample space.
        """
        seq1, seq2 = trace1["seq"], trace2["seq"]
        
        if seq1 is None or seq2 is None:
            raise ValueError("Cannot merge traces: sequence data is missing")

        # 1) Longest suffix-prefix overlap (in BASES)
        max_olap = 0
        for k in range(min_overlap, min(len(seq1), len(seq2)) + 1):
            if seq1[-k:] == seq2[:k]:
                max_olap = k
        # If no overlap, you can still concatenate, but hetero detection is less meaningful.
        # We'll proceed with max_olap ∈ [0..]

        # 2) Convert base overlap -> SAMPLE offset using PLOC2 (peak indices)
        pos1 = np.array(trace1["pos"], dtype=int)
        pos2 = np.array(trace2["pos"], dtype=int)
        if max_olap > 0 and len(pos1) >= max_olap and len(pos2) > 0:
            # Align first base of seq2 to the base in seq1 where overlap starts:
            # seq1[-max_olap] ↔ seq2[0]
            # sample offset so that pos2[0] lands on pos1[-max_olap]
            offset_samples = int(pos1[-max_olap] - pos2[0])
        else:
            # No base overlap: place seq2 right after seq1 in sample space
            offset_samples = int(pos1[-1] - pos1[0] + (pos2[-1] - pos2[0]) // 10)  # heuristic
            # A simpler choice is offset_samples = len(trace1["A"]), but PLOC2-based is better.

        # 3) Handle negative offset by pre-padding trace1
        pad_left = max(0, -offset_samples)
        offset_samples += pad_left  # shift right so it's non-negative now

        # 4) Determine merged length
        if trace1["A"] is None or trace2["A"] is None:
            raise ValueError("Cannot merge traces: trace data arrays are missing")
        len1, len2 = len(trace1["A"]), len(trace2["A"])
        total_len = max(pad_left + len1, offset_samples + len2)

        # 5) Merge signals (take max in overlap)
        merged = {}
        for base in "ACGT":
            out = np.zeros(total_len, dtype=np.int32)
            # Use zeros if trace data for this base is None
            trace1_base = trace1[base] if trace1[base] is not None else np.zeros(len1, dtype=np.int32)
            trace2_base = trace2[base] if trace2[base] is not None else np.zeros(len2, dtype=np.int32)
            # place trace1 (with left padding if needed)
            out[pad_left:pad_left + len1] = trace1_base[:len1] # type: ignore
            # place trace2 at sample offset
            start = offset_samples
            end = min(total_len, start + len2)
            seg_len = end - start
            if seg_len > 0:
                # max merge in overlap
                out[start:end] = np.maximum(out[start:end], trace2_base[:seg_len])
            merged[base] = out

        # 6) Merge peak positions in sample space and sort/unique
        pos1_merged = pad_left + pos1
        pos2_merged = offset_samples + pos2
        pos_all = np.concatenate([pos1_merged, pos2_merged])
        pos_all = pos_all[(pos_all >= 0) & (pos_all < total_len)]
        pos_all = np.unique(np.sort(pos_all))

        # 7) Consensus sequence (simple: seq1 + non-overlapping tail of seq2)
        consensus_seq = seq1 + seq2[max(0, max_olap):]

        merged["pos"] = pos_all
        merged["seq"] = consensus_seq
        merged["overlap_bases"] = int(max_olap)
        merged["offset_samples"] = int(offset_samples)
        merged["pad_left"] = int(pad_left)
        merged["total_len"] = int(total_len)
        return merged

    def normalize_trace_per_channel(self,trace_data, q=99):
        out = {}
        for b in "ACGT":
            sig = trace_data[b].astype(float) # type: ignore
            scale = np.percentile(sig, q) + 1e-6
            out[b] = sig / scale
        out["pos"] = trace_data["pos"]
        out["seq"] = trace_data["seq"]
        return out

    def detect_hetero(self,trace_data, ratio=0.10, half_window=3, min_sum=0.8, min_major=0.5):
        """
        Heterozygosity on normalized signals.
        - ratio: minor/major >= ratio
        - half_window: +/- samples around peak index (PLOC2)
        - min_sum: skip low-signal regions
        - min_major: require strong major peak
        """
        A, C, G, T = trace_data["A"], trace_data["C"], trace_data["G"], trace_data["T"]
        n = len(A)
        hetero = []

        for p in trace_data["pos"].astype(int): # type: ignore
            if p < 0 or p >= n:
                continue
            lo, hi = max(0, p - half_window), min(n, p + half_window + 1)

            vals = {
                "A": float(A[lo:hi].max()), # type: ignore
                "C": float(C[lo:hi].max()), # type: ignore
                "G": float(G[lo:hi].max()), # type: ignore
                "T": float(T[lo:hi].max()), # type: ignore
            }
            total = sum(vals.values())
            if total < min_sum:
                continue

            top = sorted(vals.items(), key=lambda x: x[1], reverse=True)
            major, minor = top[0][1], top[1][1]
            if major < min_major:
                continue
            if minor / (major + 1e-6) >= ratio:
                hetero.append((int(p), top))
        return hetero

    def extract_exon_traces(self, ab1_trace, exons, ref_start_pos=0):
        """
        Extract exon-specific regions from a chromatogram trace.

        ab1_trace: dict with A, C, G, T, pos, seq
        exons: list of dicts with exon, ref_start, ref_end
        ref_start_pos: the reference coordinate corresponding to ab1_trace["seq"][0]
        """
        seq_len = len(ab1_trace["seq"])
        ref_positions = np.arange(ref_start_pos, ref_start_pos + seq_len)

        exon_traces = []

        for exon in exons:
            start, end = exon["ref_start"], exon["ref_end"]
            mask = (ref_positions >= start) & (ref_positions <= end)
            if not np.any(mask):
                continue

            idx = np.where(mask)[0]
            start_idx, end_idx = idx[0], idx[-1]

            exon_trace = {
                "exon": exon["exon"],
                "A": ab1_trace["A"][start_idx:end_idx + 1],
                "C": ab1_trace["C"][start_idx:end_idx + 1],
                "G": ab1_trace["G"][start_idx:end_idx + 1],
                "T": ab1_trace["T"][start_idx:end_idx + 1],
                "pos": ab1_trace["pos"][start_idx:end_idx + 1] - ab1_trace["pos"][start_idx],
                "seq": ab1_trace["seq"][start_idx:end_idx + 1],
                "ref_start": start,
                "ref_end": end,
            }

            exon_traces.append(exon_trace)

        return exon_traces