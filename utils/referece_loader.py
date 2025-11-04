from pathlib import Path
import json
import networkx as nx
CUR_DIR = Path(__file__).parent.absolute()

ABO = CUR_DIR / "data" / "abo_reference_ng006669.json"

ABO_IDENTIFIER_GRAPH_PATH = CUR_DIR / "data" / "ABO_enhanced_subgraph.gml"

class ReferenceLoader:
    def __init__(self):
        pass
    
    def load_abo_reference(self):
        """Load the NCBI ABO reference data"""
        try:
            # current_dir = Path(__file__).parent
            reference_file = ABO

            with open(reference_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Warning: Could not load ABO reference: {e}")
            return {}
        
    def load_enhanced_graph(self):
        """Load the enhanced graph from GML file."""
        return self.load_graph(ABO_IDENTIFIER_GRAPH_PATH)

    def load_graph(self, graph_path):
        """Load a graph from a GML file."""
        try:
            G = nx.read_gml(graph_path)
            print(f"✅ Loaded graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
            return G
        except Exception as e:
            print(f"❌ Error loading graph: {e}")
            return None 
