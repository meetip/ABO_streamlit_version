"""
Proper FASTA Alignment Service
Uses BioPython's pairwise2 module for gap-aware sequence alignment
to eliminate coordinate shift issues and false SNP cascades.
"""

import json
from typing import Dict, List 
import utils.referece_loader as rl

 
GLOBAL = 1000
DEBUG = True

try:
    from Bio.Align import PairwiseAligner
    from Bio.Seq import Seq
    from Bio import SeqIO
    BIOPYTHON_AVAILABLE = True
except ImportError:
    BIOPYTHON_AVAILABLE = False
    PairwiseAligner = None
    print("BioPython not available. Install with: pip install biopython")


class FASTAAlignmentService:
    """
    Service for proper sequence alignment using BioPython's gap-aware algorithms.
    Replaces simple position-by-position comparison to eliminate false SNPs from indels.
    """

    def __init__(self, gene="ABO"):
        """Initialize the alignment service"""
        if not BIOPYTHON_AVAILABLE:
            raise ImportError(
                "BioPython is required. Install with: pip install biopython")

        # Load NCBI ABO reference data
        if gene == "ABO":
            self.abo_reference = rl.ReferenceLoader().load_abo_reference()

        # Alignment parameters (optimized for DNA sequences)
        self.local_alignment_params = {
            'match_score': 2,        # Positive score for matches
            'mismatch_score': -1,    # Penalty for mismatches
            'gap_open_penalty': -2,  # Higher penalty for opening a gap
            'gap_extend_penalty': -0.5,  # Higher penalty for extending a gap
            'one_alignment_only': True  # Return only the best alignment
        }

        self.global_alignment_params = {
            'match_score': 2,
            'mismatch_score': -0.5,
            'gap_open_penalty': -2,
            'gap_extend_penalty': -0.5,
            'one_alignment_only': True
        }

        

    def _extract_aligned_sequences(self, alignment):
        """Extract aligned sequences with gaps from BioPython alignment object"""
        alignment_str = str(alignment)
        lines = alignment_str.strip().split('\n')

        # if len(lines) >= 3:
        #     # Parse the alignment format:
        #     # target            0 ACGT--ACGT  8
        #     # query             0 ACGTTAACGT 10
        #     target_parts = lines[0].split()
        #     query_parts = lines[2].split()

        #     if len(target_parts) >= 3 and len(query_parts) >= 3:
        #         aligned_target = target_parts[2]  # The sequence with gaps
        #         aligned_query = query_parts[2]    # The sequence with gaps
        #         return aligned_query, aligned_target
        aligned_target = ""
        aligned_query = ""

        for i in range(0, len(lines), 4):

            target_parts = lines[i].split()
            query_parts = lines[i+2].split()

            if len(target_parts) >= 3 and len(query_parts) >= 3:
                aligned_target += target_parts[2]
                aligned_query += query_parts[2]

        return aligned_query, aligned_target

        # Fallback if parsing fails
       # return "", ""
 
    def align_sequence_to_exon(self, query_sequence: str, exon_number: int) -> Dict:
        """
        Perform proper local alignment between query sequence and specific exon.

        Args:
            query_sequence: The target sequence to align
            exon_number: Which ABO exon to align against (1-7)

        Returns:
            Dictionary with alignment results and variant information
        """
        if not self.abo_reference:
            return {'error': 'ABO reference not loaded'}

        # Get exon sequence from reference
        exon_data = None
        for exon in self.abo_reference['exons']:
            if exon['exon_number'] == exon_number:
                exon_data = exon
                break

        if not exon_data:
            return {'error': f'Exon {exon_number} not found'}

        exon_sequence = exon_data['sequence']
        

        # Create PairwiseAligner with Smith-Waterman (local) alignment settings
        if not BIOPYTHON_AVAILABLE or PairwiseAligner is None:
            return {'error': 'BioPython not available'}

        aligner = PairwiseAligner()  # type: ignore
        if len(exon_sequence) > GLOBAL:
            aligner.mode = 'global'
            aligner.match_score = self.global_alignment_params['match_score']
            aligner.mismatch_score = self.global_alignment_params['mismatch_score']
            aligner.open_gap_score = self.global_alignment_params['gap_open_penalty']
            aligner.extend_gap_score = self.global_alignment_params['gap_extend_penalty']
        else:
            aligner.mode = 'local'
            aligner.match_score = self.local_alignment_params['match_score']
            aligner.mismatch_score = self.local_alignment_params['mismatch_score']
            aligner.open_gap_score = self.local_alignment_params['gap_open_penalty']
            aligner.extend_gap_score = self.local_alignment_params['gap_extend_penalty']

        # Perform local alignment
        alignments = aligner.align(query_sequence, exon_sequence)

        # Check if there are too many alignments (indicating poor match)
        try:
            alignment_count = len(alignments)
            if alignment_count == 0:
                return {'error': 'No significant alignment found'}
        except OverflowError:
            # Too many optimal alignments indicates sequences are very different
            # Get the best alignment
            return {'error': 'Sequences too divergent - too many possible alignments'}
        best_alignment = alignments[0]
        score = best_alignment.score  # type: ignore

        # Extract aligned sequences with gaps from the alignment object
        aligned_ref, aligned_query = self._extract_aligned_sequences(
            best_alignment)

        # Calculate alignment statistics
        alignment_length = len(aligned_query)
        matches = sum(1 for q, r in zip(aligned_query, aligned_ref)
                      if q == r and q != '-' and r != '-')
        similarity = matches / alignment_length if alignment_length > 0 else 0

        # Get alignment coordinates
        query_start = best_alignment.coordinates[1][0]  # type: ignore
        query_end = best_alignment.coordinates[1][-1]  # type: ignore
        ref_start = best_alignment.coordinates[0][0]  # type: ignore
        ref_end = best_alignment.coordinates[0][-1]  # type: ignore

        # Parse variants from the alignment
        variants = self._extract_variants_from_alignment(
            aligned_query, aligned_ref, query_start, ref_start, exon_data
        )

        return {
            'exon_number': exon_number,
            'alignment_score': score,
            'similarity': similarity,
            'alignment_length': alignment_length,
            'matches': matches,
            'query_start': query_start,
            'query_end': query_end,
            'ref_start': ref_start,
            'ref_end': ref_end,
            'aligned_query': aligned_query,
            'aligned_reference': aligned_ref,
            'variants': variants,
            'exon_info': {
                'genomic_start': exon_data['start'],
                'genomic_end': exon_data['end'],
                'length': len(exon_sequence)
            }
        }

    def _extract_variants_from_alignment(self, aligned_query: str, aligned_ref: str,
                                         query_start: int, ref_start: int, exon_data: Dict) -> List[Dict]:
        """
        Extract variants from a proper alignment, handling gaps correctly.

        Args:
            aligned_query: Query sequence with gaps
            aligned_ref: Reference sequence with gaps
            query_start: Start position in query
            exon_data: Exon information with genomic coordinates

        Returns:
            List of variants with proper coordinates
        """
        variants = []
        query_pos = query_start
        ref_pos = 0

        i = 0
        while i < len(aligned_query):
            query_base = aligned_query[i]
            ref_base = aligned_ref[i]

            if query_base == '-':
                # Deletion in query (insertion in reference)
                # Count consecutive deletions
                del_length = 0
                del_sequence = ""
                j = i
                while j < len(aligned_query) and aligned_query[j] == '-':
                    del_sequence += aligned_ref[j]
                    del_length += 1
                    j += 1

                genomic_pos = exon_data['start'] + ref_pos
                isbt_pos = exon_data['cds_start'] + ref_pos

                variants.append({
                    'type': 'deletion',
                    'query_pos': query_pos,
                    'ref_pos': ref_pos,
                    'genomic_pos': genomic_pos,
                    'isbt_pos': isbt_pos,
                    'length': del_length,
                    'deleted_sequence': del_sequence,
                    'change': f"del({del_sequence})",
                    'exon': exon_data['exon_number']
                })

                ref_pos += del_length
                i += del_length

            elif ref_base == '-':
                # Insertion in query (deletion in reference)
                # Count consecutive insertions
                ins_length = 0
                ins_sequence = ""
                j = i
                while j < len(aligned_query) and aligned_ref[j] == '-':
                    ins_sequence += aligned_query[j]
                    ins_length += 1
                    j += 1

                genomic_pos = exon_data['start'] + ref_pos
                isbt_pos = exon_data['cds_start'] + ref_pos

                variants.append({
                    'type': 'insertion',
                    'query_pos': query_pos,
                    'ref_pos': ref_pos,
                    'genomic_pos': genomic_pos,
                    'isbt_pos': isbt_pos,
                    'length': ins_length,
                    'inserted_sequence': ins_sequence,
                    'change': f"ins({ins_sequence})",
                    'exon': exon_data['exon_number']
                })

                query_pos += ins_length
                i += ins_length

            else:
                # Match or mismatch
                if query_base != ref_base:
                    # SNP
                    genomic_pos = exon_data['start'] + ref_pos
                    isbt_pos = exon_data['cds_start'] + ref_pos

                    variants.append({
                        'type': 'SNP',
                        'query_pos': query_pos,
                        'ref_pos': ref_pos,
                        'genomic_pos': genomic_pos,
                        'isbt_pos': isbt_pos,
                        'ref_base': ref_base,
                        'alt_base': query_base,
                        'change': f"{ref_base}>{query_base}",
                        'exon': exon_data['exon_number']
                    })

                query_pos += 1
                ref_pos += 1
                i += 1

        return variants

    def analyze_multi_exon_sequence(self, query_sequence: str,
                                    exon_combination: List[int]) -> Dict:
        """
        Analyze a sequence that may contain multiple exons using proper alignment.

        Args:
            query_sequence: The sequence to analyze
            exon_combination: List of exon numbers to try (e.g., [5, 6, 7])

        Returns:
            Analysis results with proper alignment-based variant calling
        """
        results = {
            'query_length': len(query_sequence),
            'exon_combination': exon_combination,
            'exon_alignments': [],
            'total_variants': 0,
            'variant_summary': {'SNPs': 0, 'insertions': 0, 'deletions': 0},
            'overall_similarity': 0.0
        }

        total_score = 0
        total_length = 0
        all_variants = []

        if DEBUG: print(f"\n=== PROPER ALIGNMENT ANALYSIS ===")
        if DEBUG: print(f"Query length: {len(query_sequence)} bp")
        if DEBUG: print(f"Analyzing exons: {exon_combination}")

        # Align to each exon separately
        for exon_num in exon_combination:
            if DEBUG: print(f"\nAligning to exon {exon_num}...")

            alignment_result = self.align_sequence_to_exon(
                query_sequence, exon_num)

            if 'error' in alignment_result:
                print(f"  Error: {alignment_result['error']}")
                continue

            results['exon_alignments'].append(alignment_result)

            # Accumulate statistics
            total_score += alignment_result['alignment_score']
            total_length += alignment_result['alignment_length']

            variants = alignment_result['variants']
            all_variants.extend(variants)

            # Count variant types
            for variant in variants:
                variant_type = variant['type']
                if variant_type == 'SNP':
                    results['variant_summary']['SNPs'] += 1
                elif variant_type == 'insertion':
                    results['variant_summary']['insertions'] += 1
                elif variant_type == 'deletion':
                    results['variant_summary']['deletions'] += 1

            if DEBUG: print(f"  Score: {alignment_result['alignment_score']:.1f}, "
                  f"Similarity: {alignment_result['similarity']:.3f}, "
                  f"Variants: {len(variants)}")

        # Calculate overall statistics
        results['total_variants'] = len(all_variants)
        results['all_variants'] = all_variants
        results['overall_similarity'] = total_score / \
            total_length if total_length > 0 else 0
        if DEBUG:
            
            print(f"\n=== ALIGNMENT SUMMARY ===")
            print(f"Total variants: {results['total_variants']}")
            print(f"  SNPs: {results['variant_summary']['SNPs']}")
            print(f"  Insertions: {results['variant_summary']['insertions']}")
            print(f"  Deletions: {results['variant_summary']['deletions']}")
            print(f"Overall similarity: {results['overall_similarity']:.3f}")

        return results

    def create_coordinate_mapping_from_alignment(self, alignment_results: Dict) -> Dict:
        """
        Create coordinate mapping system from proper alignment results.
        Maps reporting coordinates to genomic coordinates while handling gaps.

        Args:
            alignment_results: Results from analyze_multi_exon_sequence

        Returns:
            Coordinate mapping compatible with existing system
        """
        coordinate_mapping = {
            'exon_boundaries': [],
            'total_positions': 0
        }

        current_reporting_pos = 1

        for alignment in alignment_results['exon_alignments']:
            exon_info = alignment['exon_info']
            exon_length = exon_info['length']

            boundary = {
                'exon_number': alignment['exon_number'],
                # Remove gaps
                'sequence': alignment['aligned_reference'].replace('-', ''),
                'reporting_start': current_reporting_pos,
                'reporting_end': current_reporting_pos + exon_length - 1,
                'genomic_start': exon_info['genomic_start'],
                'genomic_end': exon_info['genomic_end'],
                'cds_start': exon_info['cds_start'],
                'cds_end': exon_info['cds_end']
            }

            coordinate_mapping['exon_boundaries'].append(boundary)
            current_reporting_pos += exon_length

        coordinate_mapping['total_positions'] = current_reporting_pos - 1

        return coordinate_mapping

    def format_variants_for_reporting(self, variants: List[Dict],
                                      coordinate_mapping: Dict) -> List[Dict]:
        """
        Format variants for consistent reporting with coordinate mapping.

        Args:
            variants: Raw variants from alignment
            coordinate_mapping: Coordinate mapping system

        Returns:
            Formatted variants compatible with existing reporting system
        """
        formatted_variants = []

        for variant in variants:
            # Find the exon boundary for this variant
            exon_boundary = None
            for boundary in coordinate_mapping['exon_boundaries']:
                if boundary['exon_number'] == variant['exon']:
                    exon_boundary = boundary
                    break

            if not exon_boundary:
                continue

            # Calculate reporting coordinate
            offset_in_exon = variant['ref_pos']
            reporting_coord = exon_boundary['reporting_start'] + offset_in_exon

            formatted_variant = {
                'type': variant['type'],
                'change': variant['change'],
                'exon': variant['exon'],
                'coordinates': {
                    'reporting': reporting_coord,
                    'exon_relative': offset_in_exon + 1,  # 1-based
                    'genomic': variant['genomic_pos']
                }
            }

            # Add base information for SNPs
            if variant['type'] == 'SNP':
                formatted_variant['ref_base'] = variant['ref_base']
                formatted_variant['alt_base'] = variant['alt_base']

            formatted_variants.append(formatted_variant)

        # Sort by reporting coordinate
        formatted_variants.sort(key=lambda v: v['coordinates']['reporting'])

        return formatted_variants

# Example usage and testing functions
 

# if __name__ == "__main__":
#     if BIOPYTHON_AVAILABLE:
#         test_proper_alignment_service()
#     else:
#         print("BioPython not available. Please install with: pip install biopython")
