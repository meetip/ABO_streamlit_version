import networkx as nx
import json
from pathlib import Path
import utils.referece_loader as rl
ABO_IDENTIFIER_GRAPH_PATH = './data/ABO_enhanced_subgraph.gml'
ABO='./data/abo_reference_ng006669.json'





class ABOIdentifier:
    """Class to identify ABO alleles from a NetworkX graph."""

    def __init__(self,Gene="ABO"):
        if Gene=="ABO":
            self.graph = rl.ReferenceLoader().load_enhanced_graph()
            self.ABORef = rl.ReferenceLoader().load_abo_reference()
    
 


    def get_variant_node(self, position: int, from_base: str, to_base: str):
        """Retrieve a variant node from the graph based on its attributes."""
        if not self.graph:
            print("❌ Graph not loaded, cannot retrieve node.")
            return None

        for node, data in self.graph.nodes(data=True):
            if (data.get('type') == 'Variant' and
                data.get('position') == position and
                data.get('from_base') == from_base and
                data.get('to_base') == to_base):
                return (node, data)
        return None

    def get_variants_for_allele(self, allele_name: str) -> list:
        """Retrieve all variant nodes connected to a specific allele node."""
        if not self.graph:
            print("❌ Graph not loaded, cannot retrieve variants.")
            return []

        # Find the allele node
        allele_node = None
        for node, data in self.graph.nodes(data=True):
            if data.get('type') == 'Allele' and node == allele_name:
                allele_node = node
                break

        if not allele_node:
            print(f"❌ Allele '{allele_name}' not found in the graph.")
            return []

        # Get all neighbors of the allele node that are variant nodes
        variant_nodes = []
       # reversed_neighbors = list(G.predecessors(node_n)) 
        for neighbor in self.graph.predecessors(allele_node):
            if self.graph.nodes[neighbor].get('type') == 'Variant':
                variant_nodes.append((neighbor, self.graph.nodes[neighbor]))

        return variant_nodes

    def identify_alleles(self, variant_nodes: list) -> list:
        """Identify ABO alleles based on variant nodes."""
        identified_alleles = set()

        if not self.graph:
            print("❌ Graph not loaded, cannot identify alleles.")
            return list(identified_alleles)

        for node, data in variant_nodes:
            connected_alleles = [
                neighbor for neighbor in self.graph.neighbors(node)
                if self.graph.nodes[neighbor].get('type') == 'Allele'
            ]
            identified_alleles.update(connected_alleles)

        return list(identified_alleles)

    def get_exon(self,pos: int) -> int:
        """Get the exon number for a given position."""
        for exon in self.ABORef['exons']: 
            if exon['cds_start'] <= int(pos) <= exon['cds_end']: 
                return exon['exon_number']
        return None