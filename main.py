import streamlit as st
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.Align import PairwiseAligner
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import io

import utils.FASTA_analyzer as fasta_utils
import utils.ab1_analyzer as ab1_utils
import utils.abo_identifier as abo_utils

import plotly.graph_objects as go

IUPAC_CODES = {
    'A':	'A',
    'C':	'C',
    'G':	'G',
    'T':	'T',
    'R':	'A or G',
    'Y':	'C or T',
    'S':	'G or C',
    'W':	'A or T',
    'K':	'G or T',
    'M':	'A or C',
    'B':	'C or G or T',
    'D':	'A or G or T',
    'H':	'A or C or T',
    'V':	'A or C or G',
    'N':	'A or C or G or T'}


def plot_chromatogram_plotly_old(trace, base_width=2):
    """
    Create an interactive chromatogram plot with Plotly.

    Parameters
    ----------
    trace : dict
        Dictionary containing "A", "C", "G", "T", "pos", and "seq".
    base_width : float
        Controls horizontal scaling (pixels per base).
    """
    seq_len = len(trace["seq"])
    width_px = max(800, int(seq_len * base_width))

    fig = go.Figure()
    x = list(range(seq_len))

    # Plot each base channel
    fig.add_trace(go.Scatter(
        y=trace["A"], x=x, mode="lines", name="A", line=dict(color="green", width=1)))
    fig.add_trace(go.Scatter(
        y=trace["C"], x=x, mode="lines", name="C", line=dict(color="blue", width=1)))
    fig.add_trace(go.Scatter(
        y=trace["G"], x=x, mode="lines", name="G", line=dict(color="black", width=1)))
    fig.add_trace(go.Scatter(
        y=trace["T"], x=x, mode="lines", name="T", line=dict(color="red", width=1)))

    # Optional: base labels (every Nth base)
    step = max(1, seq_len // 100)
    labels = [trace["seq"][i] if i % step == 0 else "" for i in range(seq_len)]
    fig.add_trace(go.Scatter(
        x=x,
        y=[0]*seq_len,
        text=labels,
        mode="text",
        textposition="top center",
        hoverinfo="skip",
        showlegend=False
    ))

    fig.update_layout(
        title=f"ABO Exon: {trace['exon']} (length={seq_len})",
        width=width_px,
        height=400,
        xaxis_title="Base Index",
        yaxis_title="Signal Intensity",
        hovermode="x unified",
        template="plotly_white",
        legend=dict(orientation="h", y=-0.2),
        margin=dict(l=40, r=20, t=40, b=40)
    )

    return fig


def plot_chromatogram_plotly(trace, base_width=2, hetero_sites=None,
                             cds_start=None, cds_end=None):
    """
    Interactive chromatogram plot with exon coordinate support and heterozygous tooltips.

    Parameters
    ----------
    trace : dict
        Must contain "A", "C", "G", "T", "pos", and "seq".
    base_width : float
        Horizontal scaling (pixels per base).
    hetero_sites : list[tuple[int, dict]]
        List of (base_index, base_intensity_dict) for heterozygous peaks.
    cds_start, cds_end : int, optional
        Genomic coordinates for the exon region (for x-axis labels).
    """
    seq_len = len(trace["seq"])
    width_px = max(800, int(seq_len * base_width))

    # Define coordinate offset
    start_offset = cds_start if cds_start is not None else 0
    x = [start_offset + i for i in range(seq_len)]

    fig = go.Figure()

    # Base channels
    fig.add_trace(go.Scatter(
        x=x, y=trace["A"], mode="lines", name="A", line=dict(color="green", width=1)))
    fig.add_trace(go.Scatter(
        x=x, y=trace["C"], mode="lines", name="C", line=dict(color="blue", width=1)))
    fig.add_trace(go.Scatter(
        x=x, y=trace["G"], mode="lines", name="G", line=dict(color="black", width=1)))
    fig.add_trace(go.Scatter(
        x=x, y=trace["T"], mode="lines", name="T", line=dict(color="red", width=1)))

    # Optional base labels
    step = max(1, seq_len // 100)
    labels = [trace["seq"][i] if i % step == 0 else "" for i in range(seq_len)]
    fig.add_trace(go.Scatter(
        x=x,
        y=[0]*seq_len,
        text=labels,
        mode="text",
        textposition="top center",
        hoverinfo="skip",
        showlegend=False
    ))

    # Heterozygous markers with tooltip
    if hetero_sites:
        ymax = max(max(trace["A"]), max(trace["C"]),
                   max(trace["G"]), max(trace["T"]))
        for base_idx, intensities in hetero_sites:
            pos = start_offset + base_idx
            bases_sorted = sorted(intensities.items(),
                                  key=lambda kv: kv[1], reverse=True)
            top = bases_sorted[0]
            second = bases_sorted[1] if len(bases_sorted) > 1 else ("", 0)
            ratio = f"{second[0]}:{second[1]} / {top[0]}:{top[1]}  (ratio={second[1]/(top[1]+1e-6):.2f})"
            hover_text = "<br>".join(
                [f"{b}: {v}" for b, v in intensities.items()]) + f"<br>{ratio}"
            fig.add_trace(go.Scatter(
                x=[pos, pos],
                y=[0, ymax],
                mode="lines",
                line=dict(color="orange", width=1, dash="dot"),
                name=f"Hetero {pos}",
                hovertemplate=f"Pos {pos}<br>{hover_text}<extra></extra>"
            ))

    exon_label = f"{cds_start}-{cds_end}" if cds_start is not None and cds_end is not None else "?"
    fig.update_layout(
        title=f"Chromatogram –  ABO Exon: {trace['exon']} (len={seq_len})",
        width=width_px,
        height=400,
        xaxis_title="Genomic Position" if cds_start else "Base Index",
        yaxis_title="Signal Intensity",
        hovermode="x unified",
        template="plotly_white",
        legend=dict(orientation="h", y=-0.2),
        margin=dict(l=40, r=20, t=40, b=40)
    )

    return fig


def display_alignment_with_snps(aligned_query, aligned_reference, cds_start=None, cds_end=None, variants=None, exon_number=None):
    """
    Display aligned sequences in code boxes with copy functionality and red highlighting for variants.
    """

    if exon_number:
        st.write(f"#### 🧬 Exon {exon_number} Alignment")
    else:
        st.write("#### 🧬 Sequence Alignment")

    # Build sequences with HTML highlighting for differences
    ref_html = "REF: "
    query_html = "QRY: "

    for ref_base, query_base in zip(aligned_reference, aligned_query):
        if ref_base != query_base:
            # Red highlighting for differences
            ref_html += f'<span style="color: red; font-weight: bold; background-color: #ffeeee;">{ref_base}</span>'
            query_html += f'<span style="color: red; font-weight: bold; background-color: #ffeeee;">{query_base}</span>'
        else:
            ref_html += ref_base
            query_html += query_base

    # Display sequences with highlighting
    st.write("**Reference Sequence:**")
    st.markdown(f'<div style="background-color: #f8f9fa; padding: 10px; border-radius: 5px; border: 1px solid #e9ecef; font-family: monospace; font-size: 14px; position: relative;">{ref_html}</div>',
                unsafe_allow_html=True)

    # Also provide plain text version for easy copying
    ref_plain = f"REF: {aligned_reference}"
    with st.expander("📋 Copy plain text reference"):
        st.code(ref_plain, language=None)

    st.write("**Query Sequence:**")
    st.markdown(f'<div style="background-color: #f8f9fa; padding: 10px; border-radius: 5px; border: 1px solid #e9ecef; font-family: monospace; font-size: 14px; position: relative;">{query_html}</div>',
                unsafe_allow_html=True)

    # Also provide plain text version for easy copying
    query_plain = f"QRY: {aligned_query}"
    with st.expander("📋 Copy plain text query"):
        st.code(query_plain, language=None)

    # Analyze differences and show summary
    differences = []
    snps = 0
    insertions = 0
    deletions = 0

    for i, (ref_base, query_base) in enumerate(zip(aligned_reference, aligned_query)):
        if ref_base != query_base:
            pos = i + 1
            if ref_base == '-':
                insertions += 1
                differences.append(f"Pos {pos}: Insertion ({query_base})")
            elif query_base == '-':
                deletions += 1
                differences.append(f"Pos {pos}: Deletion ({ref_base})")
            else:
                snps += 1
                differences.append(f"Pos {pos}: SNP ({ref_base}→{query_base})")

    # Display summary statistics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Differences", len(differences))
    with col2:
        st.metric("SNPs", snps)
    with col3:
        st.metric("Insertions", insertions)
    with col4:
        st.metric("Deletions", deletions)


def display_detailed_alignment_table(aligned_query, aligned_reference, variants=None, cds_start=None, cds_end=None):
    """
    Display detailed alignment in table format with position-by-position comparison.
    """

    st.write("#### 📊 Detailed Position-by-Position Analysis")

    # Prepare data for table
    positions = []
    ref_bases = []
    query_bases = []
    match_status = []
    in_cds = []
    variant_info = []

    # Create variant lookup by position
    variant_lookup = {}
    if variants:
        for var in variants:
            pos = var.get('alignment_pos', var.get(
                'ref_pos', var.get('position')))
            if pos is not None:
                variant_lookup[pos] = var

    for i, (ref_base, query_base) in enumerate(zip(aligned_reference, aligned_query)):
        pos = i + 1
        positions.append(pos)
        ref_bases.append(ref_base)
        query_bases.append(query_base)

        # Determine match status
        if ref_base == query_base:
            match_status.append("✅ Match")
        elif ref_base == '-':
            match_status.append("🔹 Insertion")
        elif query_base == '-':
            match_status.append("🔸 Deletion")
        else:
            match_status.append("❌ SNP")

        # Check if in CDS
        if cds_start and cds_end and cds_start <= pos <= cds_end:
            in_cds.append("Yes")
        else:
            in_cds.append("No")

        # Add variant information
        if pos in variant_lookup:
            var = variant_lookup[pos]
            variant_info.append(
                f"ISBT: {var.get('isbt_pos', 'N/A')}, Type: {var.get('type', 'N/A')}")
        else:
            variant_info.append("-")

    # Show only positions with differences (SNPs, insertions, deletions)
    show_all = st.checkbox("Show all positions", value=False)

    if not show_all:
        # Filter to show only differences
        filtered_data = []
        for i in range(len(positions)):
            if match_status[i] != "✅ Match":
                filtered_data.append({
                    "Position": positions[i],
                    "Reference": ref_bases[i],
                    "Query": query_bases[i],
                    "Status": match_status[i],
                    "In CDS": in_cds[i],
                    "Variant Info": variant_info[i]
                })

        if filtered_data:
            df = pd.DataFrame(filtered_data)
            st.write(
                f"**Showing {len(filtered_data)} positions with differences:**")
            st.dataframe(df, width='stretch', height=400)
        else:
            st.success("No differences found - sequences are identical!")
    else:
        # Show all positions
        all_data = {
            "Position": positions,
            "Reference": ref_bases,
            "Query": query_bases,
            "Status": match_status,
            "In CDS": in_cds,
            "Variant Info": variant_info
        }
        df = pd.DataFrame(all_data)
        st.write(f"**Showing all {len(positions)} positions:**")
        st.dataframe(df, width='stretch', height=400)


def process_ab1_files(fwd_ab1_files, exons_ref, threshold_ratio=0.3):

    mapping_service = fasta_utils.FASTAAlignmentService()
    traces = None
    if len(fwd_ab1_files) > 1:
        for i in fwd_ab1_files:
            ab1_service = ab1_utils.AB1Analyzer()
            trace = ab1_service.read_ab1_trace(i)
            trace = ab1_service.merge_overlap(
                traces, trace) if traces else trace
            traces = trace

    else:
        ab1_service = ab1_utils.AB1Analyzer()
        traces = ab1_service.read_ab1_trace(fwd_ab1_files[0])

    if traces:
        merged_reverse = ab1_service.reverse_chromatogram(traces)
        merged_norm = ab1_service.normalize_trace_per_channel(merged_reverse)
        hets = ab1_service.detect_hetero(merged_reverse, ratio=threshold_ratio)

        results = ab1_service.extract_exon_traces(merged_reverse, exons_ref)
        return results, hets

    return None, None


def process_fasta_file(fasta_file, exon_start=0, exon_end=0):
    # Convert Streamlit uploaded file to text mode for BioPython

    # Read the file content and convert to string
    fasta_content = fasta_file.read()
    if isinstance(fasta_content, bytes):
        fasta_content = fasta_content.decode('utf-8')

    # Create a text StringIO object for BioPython
    fasta_text = io.StringIO(fasta_content)

    # Parse the FASTA file
    fasta_records = list(SeqIO.parse(fasta_text, "fasta"))
    if not fasta_records:
        raise ValueError("No sequences found in FASTA file")

    # Use the first sequence
    first_record = fasta_records[0]
    service = fasta_utils.FASTAAlignmentService()
    fwd_seq = first_record.seq
    rev_seq = fwd_seq.reverse_complement()

    if exon_end > exon_start > 0:
        fwd_fasta_analysis = service.analyze_multi_exon_sequence(
            fwd_seq, list(range(exon_start, exon_end+1)))
        rev_fasta_analysis = service.analyze_multi_exon_sequence(
            rev_seq, list(range(exon_start, exon_end+1)))
    elif exon_start == 0:
        fwd_fasta_analysis = service.analyze_multi_exon_sequence(
            fwd_seq, list(range(1, 8)))
        rev_fasta_analysis = service.analyze_multi_exon_sequence(
            rev_seq, list(range(1, 8)))
    else:
        fwd_fasta_analysis = {}
        rev_fasta_analysis = {}
    strand = {"forward": fwd_fasta_analysis,
              "reverse": rev_fasta_analysis, "none": "none"}
    fwd_similarities = {}
    rev_similarities = {}
    if 'error' in fwd_fasta_analysis or 'error' in rev_fasta_analysis:
        return {}, []
    for i in fwd_fasta_analysis['exon_alignments']:
        fwd_similarities[i['exon_number']] = i['similarity']

    for i in rev_fasta_analysis['exon_alignments']:
        rev_similarities[i['exon_number']] = i['similarity']

    exon_comparison = {}
    for exon_num in fwd_similarities.keys():
        fwd_sim = fwd_similarities.get(exon_num, 0)
        rev_sim = rev_similarities.get(exon_num, 0)
        if fwd_sim > rev_sim and fwd_sim > 0.:
            exon_comparison[exon_num] = {
                "winner": "forward", "similarity": fwd_sim}
        elif rev_sim > fwd_sim and rev_sim > 0.9:
            exon_comparison[exon_num] = {
                "winner": "reverse", "similarity": rev_sim}
        else:
            exon_comparison[exon_num] = {
                "winner": "tie", "similarity": fwd_sim}

    count_forward_wins = sum(
        1 for v in exon_comparison.values() if v['winner'] == 'forward')
    count_reverse_wins = sum(
        1 for v in exon_comparison.values() if v['winner'] == 'reverse')
    count_ties = sum(1 for v in exon_comparison.values()
                     if v['winner'] == 'tie')
    summary = {
        "forward_wins": count_forward_wins,
        "reverse_wins": count_reverse_wins,
        "ties": count_ties
    }
    if summary["forward_wins"] > summary["reverse_wins"]:

        best_match = "forward"

    elif summary["reverse_wins"] > summary["forward_wins"]:
        best_match = "reverse"
    else:
        best_match = "forward"

    selected_strand = strand[best_match]
    aboRef = service.getABO_ref("exons")

    exons_ref = []
    exon_combination = []
    for i in selected_strand['exon_alignments']:
        if i['similarity'] > 0.9:
            x = i['exon_number']-1
            exon = {}
            exon['exon'] = i['exon_number']

            exon['ref_start'] = i['ref_start']
            exon['ref_end'] = i['ref_end']

            exon['cds_start'] = aboRef[x]['cds_start']
            exon['cds_end'] = aboRef[x]['cds_end']
            exons_ref.append(exon)
            exon_combination.append(i['exon_number'])

    filtered_exons = [exon for exon in selected_strand['exon_alignments']
                      if exon['exon_number'] in [e['exon'] for e in exons_ref]]
    selected_strand['exon_alignments'] = filtered_exons
    selected_strand['exon_combination'] = exon_combination

    total_variants = 0
    SNPs = 0
    insertions = 0
    deletions = 0
    for exon in selected_strand['exon_alignments']:
        total_variants += len(exon['variants'])
        for var in exon['variants']:
            if var['type'] == 'SNP':
                SNPs += 1
            elif var['type'] == 'insertion':
                insertions += 1
            elif var['type'] == 'deletion':
                deletions += 1
    selected_strand['total_variants'] = total_variants
    selected_strand['variant_summary'] = {
        "SNPs": SNPs,
        "insertions": insertions,
        "deletions": deletions
    }

    return selected_strand, exons_ref


def handle_IUPAC_codes(abo_identifier, i, types):
    print(i)
    types_list = {'SNP': 'alt_base', 'insertion': 'inserted_sequence',
                  'deletion': 'deleted_sequence'}
    het_variants = []
    var_nodes = []
    unknown = []
    variant_base = i[types_list[types]]
    possible_bases = IUPAC_CODES.get(variant_base, "").split(" or ")
    print(i['isbt_pos'], possible_bases)
    for base in possible_bases:
        field = types_list[i['type']]
        var_node = None
        if i['type'] == 'deletion':
            var_node = abo_identifier.get_variant_node(
                i['isbt_pos'], base, "", variant_base)
        elif i['type'] == 'insertion':
            var_node = abo_identifier.get_variant_node(
                i['isbt_pos'], "", base, variant_base)
        else:
            var_node = abo_identifier.get_variant_node(
                i['isbt_pos'], i['ref_base'], base, variant_base)

        if var_node is not None:
            print(var_node)
            var_nodes.append(var_node)
            het_variants.append(i[field])
        else:
            if base != i['ref_base']:

                unknown.append(i)
    return var_nodes, het_variants, unknown


def get_display_base(base):
    """Convert IUPAC code to display string."""
    if base in IUPAC_CODES:
        base_display = IUPAC_CODES[base]
        if base == base_display:
            return base
        else:
            return f"{base} ({base_display})"
    return base


def identify_abo_alleles(FASTA_variant_list):
    abo_identifier = abo_utils.ABOIdentifier("ABO")
    var_nodes = []
    het_variants = []
    unknown = []
    for exon in FASTA_variant_list['exon_alignments']:
        variants = exon['variants']
        for i in variants:
            if i['type'] == 'insertion':
                var_node, het_var, unk = handle_IUPAC_codes(
                    abo_identifier, i, 'insertion')
            elif i['type'] == 'deletion':
                var_node, het_var, unk = handle_IUPAC_codes(
                    abo_identifier, i, 'deletion')
            else:
                var_node, het_var, unk = handle_IUPAC_codes(
                    abo_identifier, i, 'SNP')
            var_nodes.extend(var_node)
            het_variants.extend(het_var)
            unknown.extend(unk)
    alleles = []
    node_iupac_map = {node[0]: node[2] for node in var_nodes}
    print(node_iupac_map)
    for node_name, node_data, iupac_code in var_nodes:
        # The identify_alleles method expects a list of tuples (node_name, node_data)
        # So we pass the current node as a list containing one tuple
        allele = abo_identifier.identify_alleles([(node_name, node_data)])
        # Use extend to add elements from the list
        alleles.append({node_name: allele})

    # Extract the lists of alleles from the 'alleles' list of dictionaries
    allele_lists = [list(d.values())[0] for d in alleles]

    # Find the intersection of all allele lists
    if allele_lists:
        # Start with the first list
        common_alleles = set(allele_lists[0])
        # Intersect with the remaining lists
        for allele_list in allele_lists[1:]:
            common_alleles.intersection_update(allele_list)
    else:
        common_alleles = set()
    allele_variants_list = []
    for i in common_alleles:
        v = abo_identifier.get_variants_for_allele(i)
        av_list = []
        for j in v:
            gene, location, change = j[0].split("_")
            exon = abo_identifier.get_exon(location)
            av_list.append({"name": j[0], "exon": exon, "location": int(
                location), "change": change})  # Convert location to int for sorting

        av_list.sort(key=lambda x: x['location'])

        allele_variants_list.append({i: av_list})
    allele_variants_list.sort(key=lambda x: list(x.keys())[
                              0])  # Sort by allele name

    variants_name = [x[0] for x in var_nodes]

    unknown_alleles_to_display = []
    for u in unknown:
        item = {}
        exon = abo_identifier.get_exon(u.get('isbt_pos'))
        item['isbt_pos'] = u.get('isbt_pos')
        if exon is None:
            item['exon'] = 'N/A'
        item['type'] = u.get('type')
        if u.get('type') == 'deletion':
            item['ref_base'] = get_display_base(
                u.get('deleted_sequence', 'N/A')),
            item['alt_base'] = '-',
        elif u.get('type') == 'insertion':
            item['ref_base'] = '-',
            item['alt_base'] = get_display_base(
                u.get('inserted_sequence', 'N/A')),

        else:
            item['ref_base'] = get_display_base(u.get('ref_base', 'N/A')),
            item['alt_base'] = get_display_base(u.get('alt_base', 'N/A')),

        unknown_alleles_to_display.append(item)
    return allele_variants_list, unknown_alleles_to_display, variants_name, node_iupac_map


def get_display_iupac_change(change, iupac_code):
    change_list = change.split(">")
    if len(change_list) > 1:
        ref_base = change_list[0]
        return f"{ref_base}>{iupac_code}"
    else:
        change = change[:-1] + f"({iupac_code})"
        return change


st.title("🧬 ABO blood group analysis")

# Upload section
# with st.sidebar:
fwd_ab1 = st.sidebar.file_uploader("Upload  AB1 file", type=[
    "ab1"], accept_multiple_files=True, help="You can upload multiple files for batch processing.")

fasta_files = st.sidebar.file_uploader(
    "Upload exon-specific FASTA", type=["fasta", "fa", "fas"], accept_multiple_files=True)
exon_start = st.sidebar.number_input(
    "Exon start (optional)", min_value=0, value=0)
exon_end = st.sidebar.number_input(
    "Exon end (optional, 0 = full length)", min_value=0, value=0)

threshold_ratio = st.sidebar.slider(
    "Heterozygosity threshold ratio", 0.1, 0.9, 0.3, 0.05)

analyze_button = st.sidebar.button("Analyze")


def get_cds(exon_number):
    for exon in exons_ref:
        if exon['exon'] == exon_number:
            return exon['cds_start'], exon['cds_end']
    return None, None


# --- Main Panel ---
st.title("Genetic Analysis Dashboard")
st.set_page_config(layout="wide")

# Tabs for displaying results
tab1, tab2, tab3, tab4 = st.tabs([
    "Chromatogram Check for Heterozygotes",
    "Exon-based SNP",
    "Allele Prediction",
    "Teams & References"
])

with tab4:
    st.markdown("### 📚 References")
    st.markdown("""
            Sirikul C, Wita R, Anukul N.,
                "Assessment of a New ABO Blood Group Genotyping Online Platform."
                , J Hemato Transfus Med. Vol. 35  Supplement 2025. ISSN 2985-2404 (online).")
    """)
    st.markdown("""
    S. Prananpaeng, T. Thaiyanto, R. Wita and N. Anukul, 
    "Whole-Exome Sequencing (WES) Analysis for ABO Subgroups Identification," 
    *2023 20th International Joint Conference on Computer Science and Software Engineering (JCSSE)*, 
    2023, pp. 264-269, url: [https://ieeexplore.ieee.org/document/10202117/](https://ieeexplore.ieee.org/document/10202117/).
    """)
    st.markdown("""
    R. Wita, S. Somhom, J. Chawachat, A. Thongratsameethong, N. Anukul and C. Sirikul, 
    "DNA Sequencing Analysis Framework for ABO Genotyping and ABO Discrepancy Resolution," 
    *2021 18th International Conference on Electrical Engineering/Electronics, Computer, 
    Telecommunications and Information Technology (ECTI-CON)*, Chiang Mai, Thailand, 2021, 
    pp. 913-916, doi: [10.1109/ECTI-CON51831.2021.9454861](https://ieeexplore.ieee.org/document/9454861).
    """)
    st.markdown("""
            Anukul N, Wita R, Leetrakool N, Sirikul C, Veeraphan N, Wongchai S. 
             "Two novel alleles on Fucosyltransferase 2 from northern Thai paraBombay 
             family and computational prediction on mutation effect."
              Transfusion. 2021;1–11. [https://doi.org/10.1111/trf.16646](https://doi.org/10.1111/trf.16646)
             """)

    st.subheader("👥 Teams")
    st.markdown("#### InnoGeHLA Lab")
    st.markdown("""
The InnoGeHLA (**Inno**vation **Ge**nomics **HLA**) Lab at Chiang Mai University is a collaborative research group 
                that brings together expertise in genomics, bioinformatics, and transfusion science. 
                The lab’s mission is to blend molecular biology with computational innovation to enhance 
                precision medicine and genetic diagnostics, with a special focus on blood group and HLA 
                genotyping.

The team is led by **Asst. Prof. Nampeung Anukul**, whose work centers on blood group genetics and transfusion 
                science, and **Miss Chonthicha Sirikul**, who specializes in molecular diagnostics and hematology.
                 Joining them is **Asst. Prof. Ratsameetip Wita**, a computer scientist from the Faculty of 
                Science who explores the use of artificial intelligence and bioinformatics to interpret 
                complex genetic data.

Together, they form a dynamic and interdisciplinary team that bridges the gap between biomedical research 
                and computational science—working to turn genomic insights into practical tools that 
                benefit both laboratories and patients.
                """)
    st.markdown("##### Lab members")

    # CSS for circular images
    st.markdown("""
        <style>
        .member-card {
            text-align: center;
            padding: 20px;
        }
        
        .member-name {
            font-weight: bold;
            font-size: 18px;
            margin-top: 15px;
            color: #333;
            text-align: center;
        }
        
        .member-position {
            font-size: 14px;
            color: #666;
            margin-top: 5px;
            line-height: 1.6;
            text-align: center;
        }
        
        /* Center the image within its container */
        .element-container:has(div[data-testid="stImage"]) {
            display: flex;
            justify-content: center;
            align-items: center;
        }
        
        div[data-testid="stImage"] {
            text-align: center;
        }
        
        /* Remove width fit-content from Streamlit's emotion cache class */
        .st-emotion-cache-p75nl5 {
            width: 100% !important;
        }
        
        /* Make Streamlit images circular */
        div[data-testid="stImage"] img {
            border-radius: 50%;
            width: 200px;
            height: 200px;
            object-fit: cover;
            object-position: center;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
            display: inline-block;
        }
        </style>
    """, unsafe_allow_html=True)

    

    # Create three columns for team members
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown('<div class="member-card">', unsafe_allow_html=True)
        try:
            from PIL import Image
            img = Image.open("utils/img/NP.jpg")
            st.image(img, width=200)
        except Exception as e:
            st.image("https://via.placeholder.com/200", width=200)
        st.markdown('<div class="member-name">Nampeung Anukul</div>',
                    unsafe_allow_html=True)
        st.markdown('''<div class="member-position">
            Division of
Blood Transfusion Science,<br>Faculty of Associated Medical Sciences<br>
            Chiang Mai University<br>
            nampeung.a@cmu.ac.th
        </div>''', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="member-card">', unsafe_allow_html=True)
        try:
            from PIL import Image
            img = Image.open("utils/img/CS.jpg")
            st.image(img, width=200)
        except Exception as e:
            st.image("https://via.placeholder.com/200", width=200)
        st.markdown(
            '<div class="member-name">Chonthicha Sirikul</div>', unsafe_allow_html=True)
        st.markdown('''<div class="member-position">
            Division of
Blood Transfusion Science,<br> Faculty of Associated Medical Sciences<br>
            Chiang Mai University<br>
            chonthicha.sir@cmu.ac.th
        </div>''', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col3:
        st.markdown('<div class="member-card">', unsafe_allow_html=True)
        try:
            from PIL import Image
            img = Image.open("utils/img/RW.jpg")
            st.image(img, width=200)
        except Exception as e:
            st.image("https://via.placeholder.com/200", width=200)
        st.markdown('<div class="member-name">Ratsameetip Wita</div>',
                    unsafe_allow_html=True)
        st.markdown('''<div class="member-position">
            Department of Computer Science,<br> Faculty of Science<br>
            Chiang Mai University<br>
            ratsameetip.wit@cmu.ac.th
        </div>''', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    # Student members table - read from CSV file

    st.markdown("##### Student Contributors")
    import pandas as pd
    try:
        student_data = pd.read_csv("utils/data/students.csv")
        st.dataframe(student_data, hide_index=True)
    except FileNotFoundError:
        st.warning(
            "Student data file not found. Please add utils/data/students.csv")
    except Exception as e:
        st.error(f"Error loading student data: {e}")

if analyze_button:
    if not fwd_ab1 and not fasta_files:
        st.warning("Please upload at least one file before analyzing.")
    else:
        st.success("Files uploaded successfully! Starting analysis...")

        if fasta_files:
            exons_ref = []
            processed_FASTA = {}
            for ref_file in fasta_files:
                fasta_detected, exons_ref = process_fasta_file(
                    ref_file, exon_start, exon_end)

                if fasta_detected:
                    if processed_FASTA == {}:
                        processed_FASTA = fasta_detected
                    else:
                        for i in fasta_detected['exon_combination']:
                            if i not in processed_FASTA['exon_combination']:
                                processed_FASTA['exon_combination'].append(i)
                                processed_FASTA['query_length'] += fasta_detected['query_length']
                                processed_FASTA['exon_alignments'].extend(
                                    # type: ignore
                                    fasta_detected['exon_alignments'])

        processed_AB1, hets = process_ab1_files(
            fwd_ab1, exons_ref, threshold_ratio) if fwd_ab1 else (None, None)

        possible_alleles, unknown_alleles_to_display, variants_name, node_iupac_map = identify_abo_alleles(
            processed_FASTA) if processed_FASTA else []

        tested_exons = [exon['exon'] for exon in exons_ref]

        with tab1:
            st.subheader("Chromatogram Check for Heterozygotes")
            st.write("### Heterozygote Positions Detected: ",
                     len(hets) if hets else 0)
            st.write("👉 Display chromatogram analysis results here")
            hetero_sites = [
                (31, {"A": 0, "C": 0, "G": 9, "T": 55}),
                (78, {"A": 200, "C": 180, "G": 10, "T": 0})
            ]
            if processed_AB1:
                for i in processed_AB1:  # type: ignore
                    x = i['exon']
                    cds_start, cds_end = get_cds(x)

                    fig = plot_chromatogram_plotly(
                        i, base_width=2,  cds_start=cds_start, cds_end=cds_end, hetero_sites=hetero_sites)
                    st.plotly_chart(fig, width='stretch')

            if hets:
                st.write("### Detected Heterozygous Positions:")
                het_data = {
                    "Position": [h['position'] for h in hets],
                    "Ref Base": [h['ref_base'] for h in hets],
                    "Alt Base": [h['alt_base'] for h in hets],
                    "Ratio": [round(h['ratio'], 2) for h in hets]
                }
                het_df = pd.DataFrame(het_data)
                st.table(het_df)

        with tab2:
            st.subheader("Exon-based SNP")

            st.write("👉 Display exon SNP comparison or table results here")
            st.markdown(
                "**Reference Gene:** [NG_006669.1](https://www.ncbi.nlm.nih.gov/nuccore/NG_006669.1)")
            st.write("### Exon Alignments Summary")
            st.write("Total Length of Sequence Analyzed: ",
                     processed_FASTA['query_length'])
            st.write("Total Exons Detected: ",
                     len(processed_FASTA['exon_alignments']))
            st.write("Exon Processing Results:")
            for exon in processed_FASTA['exon_alignments']:
                genomic_pos = []
                isbt_pos = []
                type = []
                ref_base = []
                alt_base = []

                # Display alignment visualization if available
                if 'aligned_query' in exon and 'aligned_reference' in exon:
                    st.write("---")
                    display_alignment_with_snps(
                        aligned_query=exon['aligned_query'],
                        aligned_reference=exon['aligned_reference'],
                        cds_start=exon.get('cds_start'),
                        cds_end=exon.get('cds_end'),
                        variants=exon['variants'],
                        exon_number=exon['exon_number']
                    )

                for var in exon['variants']:
                    # Debug: print available keys
                    # st.write(f"DEBUG: Variant keys: {list(var.keys())}")

                    genomic_pos.append(var.get('genomic_pos', 'N/A'))
                    isbt_pos.append(var.get('isbt_pos', 'N/A'))
                    type.append(var.get('type', 'N/A'))
                    if var.get('type') == 'deletion':
                        ref_base.append(get_display_base(
                            var.get('deleted_sequence')))
                        alt_base.append('-')
                    elif var.get('type') == 'insertion':
                        ref_base.append('-')
                        alt_base.append(get_display_base(
                            var.get('inserted_sequence')))
                    else:
                        ref_base.append(get_display_base(
                            var.get('ref_base', var.get('ref', 'N/A'))))
                        alt_base.append(get_display_base(
                            var.get('alt_base', var.get('alt', 'N/A'))))

                if exon['variants']:
                    st.write("**Variant Summary Table:**")
                    variant_data = {
                        "Genomic Position": genomic_pos,
                        "ISBT Position": isbt_pos,
                        "Type": type,
                        "Reference Base": ref_base,
                        "Alternate Base": alt_base
                    }
                    df = pd.DataFrame(variant_data)

                    st.dataframe(df, hide_index=True)

        with tab3:
            st.subheader("Allele Prediction")
            st.write("👉 Display predicted alleles or summary results here")
            st.markdown(
                "**Reference:** [ISBT ABO Alleles Table](https://www.isbtweb.org/resource/001aboalleles.html)")

            # Color code legend
            st.markdown("""
            #### 🎨 Color Code Legend:
            """)

            legend_html = """
            <table style="border-collapse: collapse; margin-bottom: 20px;">
                <tr>
                    <td style="background-color: #FEC98F; padding: 8px 12px; border: 2px solid #999; font-weight: bold;">Heterozygous Variant</td>
                    <td style="padding: 8px 12px; border: 2px solid #999;">Variant detected with IUPAC ambiguity code (heterozygous)</td>
                </tr>
                <tr>
                    <td style="background-color: #E0FFCC; padding: 8px 12px; border: 2px solid #999; font-weight: bold;">Found in Tested Exon</td>
                    <td style="padding: 8px 12px; border: 2px solid #999;">Variant confirmed in the tested exon</td>
                </tr>
                <tr>
                    <td style="background-color: #FFD6C9; padding: 8px 12px; border: 2px solid #999; font-weight: bold;">Not Found in Tested Exon</td>
                    <td style="padding: 8px 12px; border: 2px solid #999;">Variant expected but not detected in the tested exon</td>
                </tr>
                <tr>
                    <td style="background-color: #EFECE6; padding: 8px 12px; border: 2px solid #999; font-weight: bold;">Not in Tested Exon</td>
                    <td style="padding: 8px 12px; border: 2px solid #999;">Variant in exon that was not included in the test</td>
                </tr>
            </table>
            """
            st.markdown(legend_html, unsafe_allow_html=True)

            st.write("### Predicted ABO Alleles:")

            if possible_alleles:
                # allele_variants_df = pd.DataFrame(possible_alleles)
                # st.table(allele_variants_df)
                st.markdown(
                    """
                    <style>
                    table {
                        border-collapse: collapse;
                        width: 100%;
                        border: 3px solid #999;
                    }
                    th, td {
                        border: 3px solid #ccc;
                        padding: 8px 12px;
                        text-align: center;
                        vertical-align: middle;
                    }
                    </style>
                    """,
                    unsafe_allow_html=True
                )
                html_string = "<table>"
                html_string += "<tr><th>Allele</th><th>Variant Name</th><th>Exon</th><th>Location</th><th>Change</th></tr>"
                print("possible alleles = ", possible_alleles)
                print("node_iupac_map = ", node_iupac_map)
                print("tested_exons = ", tested_exons)
                for allele_data in possible_alleles:
                    for allele_name, variants in allele_data.items():
                        num_variants = len(variants)
                        html_string += f"<tr style='border-top: 3px solid #999;'><td rowspan='{num_variants}'>{allele_name}</td>"
                        for i, variant in enumerate(variants):
                            hets = False
                            if variant['name'] in node_iupac_map:
                                iupac_code = node_iupac_map[variant['name']]
                                if iupac_code not in ['A', 'T', 'C', 'G']:
                                    hets = True

                            if i > 0:
                                html_string += "<tr>"
                            if variant['exon'] in tested_exons:
                                if variant['name'] in variants_name and variant['exon'] in tested_exons:
                                    if hets:
                                        iupac_code = node_iupac_map[variant['name']]
                                        html_string += f"<td style='background-color: #FEC98F;'>{variant['name']} ({iupac_code})</td><td style='background-color: #FEC98F;'>{variant['exon']}</td><td style='background-color: #FEC98F;'>{variant['location']}</td><td style='background-color: #FEC98F;'>{get_display_iupac_change(variant['change'], iupac_code)}</td></tr>"
                                    else:
                                        html_string += f"<td style='background-color: #E0FFCC;'>{variant['name']}</td><td style='background-color: #E0FFCC;'>{variant['exon']}</td><td style='background-color: #E0FFCC;'>{variant['location']}</td><td style='background-color: #E0FFCC;'>{variant['change']}</td></tr>"
                                else:
                                    html_string += f"<td style='background-color: #FFD6C9;'>{variant['name']}</td><td style='background-color: #FFD6C9;'>{variant['exon']}</td><td style='background-color: #FFD6C9;'>{variant['location']}</td><td style='background-color: #FFD6C9;'>{variant['change']}</td></tr>"
                            else:
                                html_string += f"<td style='background-color: #EFECE6;'>{variant['name']}</td><td style='background-color: #EFECE6;'>{variant['exon']}</td><td style='background-color: #EFECE6;'>{variant['location']}</td><td style='background-color: #EFECE6;'>{variant['change']}</td></tr>"

                html_string += "</table>"
                st.markdown(html_string, unsafe_allow_html=True)

            if unknown_alleles_to_display:
                st.write("### Unknown ABO Alleles:")

                unknown_alleles_df = pd.DataFrame(unknown_alleles_to_display)
                st.dataframe(unknown_alleles_df, hide_index=True)

else:
    st.info("Upload files and click **Analyze** to start.")
