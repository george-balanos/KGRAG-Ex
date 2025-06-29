import csv
import networkx as nx
import pickle
import heapq
from colors import *
import matplotlib.pyplot as plt
import os


class KnowledgeGraphCreator:
    
    def __init__(self):
        self.csv_file = "Graph/arch_kg.csv"
        self.kg_file = "Graph/knowledge_graph3.pkl"
        self.graph = nx.DiGraph()
        self.categories = ['Diseases', 'Medications', 'Symptoms', 'Treatments', 
                           'Risk Factors', 'Diagnostic Tests', 'Body Parts']
        self.betweennessMap = {}

        self.load_betweenness_map()

    def read_csv_and_create_graph(self):
        """Reads the CSV file and constructs a directed knowledge graph with entity labels and categories."""
        with open(self.csv_file, newline='', encoding='utf-8') as file:
            reader = csv.reader(file)
            next(reader, None)  
            for row in reader:
                if len(row) != 7:
                    print(f"Skipping invalid row: {row}")
                    continue
                
                entity1, label1, relationship, entity2, label2, source_article, chunk_id = row
                
                label1 = label1.strip().replace('*', '')
                label2 = label2.strip().replace('*', '')
                
                self.create_edges(self.graph, entity1, label1, entity2, label2, source_article, chunk_id)
        
        self.save_graph(self.graph)

    def create_edges(self, graph, entity1, label1, entity2, label2, source_article, chunk_id):
        """Creates edges between entities and assigns category labels correctly."""
        
        def safe_update_node(entity, label):
            if label in self.categories:
                if entity not in graph.nodes:
                    graph.add_node(entity, label=label, source_article={source_article}, chunk_id={chunk_id})
                else:
                    if 'source_article' not in graph.nodes[entity]:
                        graph.nodes[entity]['source_article'] = set()
                    if 'chunk_id' not in graph.nodes[entity]:
                        graph.nodes[entity]['chunk_id'] = set()
                    
                    graph.nodes[entity]['source_article'].add(source_article)
                    graph.nodes[entity]['chunk_id'].add(chunk_id)

        safe_update_node(entity1, label1)
        safe_update_node(entity2, label2)

        fixed_relationship = self.get_fixed_relationship(label1, label2)
        graph.add_edge(
            entity1, entity2,
            relationship=fixed_relationship,
            source_article=source_article,
            chunk_id=chunk_id
        )

        reverse_relationship = self.get_reverse_relationship(fixed_relationship)
        if reverse_relationship != fixed_relationship:
            graph.add_edge(
                entity2, entity1,
                relationship=reverse_relationship,
                source_article=source_article,
                chunk_id=chunk_id
            )

    def get_fixed_relationship(self, label1, label2):
        fixed_relationships = {
            ('Diseases', 'Diseases'): 'HAS_RISK_FACTOR',
            ('Diseases', 'Medications'): 'TREATED_WITH',
            ('Diseases', 'Symptoms'): 'HAS_SYMPTOM',
            ('Diseases', 'Treatments'): 'TREATED_WITH',
            ('Diseases', 'Risk Factors'): 'HAS_RISK_FACTOR',
            ('Diseases', 'Diagnostic Tests'): 'DIAGNOSED_BY',
            ('Diseases', 'Body Parts'): 'AFFECTS',
            
            ('Medications', 'Diseases'): 'TREATS',
            ('Medications', 'Medications'): 'HAS_INTERACTION',
            ('Medications', 'Symptoms'): 'ALLEVIATES',
            ('Medications', 'Treatments'): 'USED_FOR',
            ('Medications', 'Risk Factors'): 'USED_FOR',
            ('Medications', 'Diagnostic Tests'): 'USED_IN',
            ('Medications', 'Body Parts'): 'AFFECTS',
            
            ('Symptoms', 'Diseases'): 'IS_SYMPTOM_OF',
            ('Symptoms', 'Medications'): 'TREATED_BY',
            ('Symptoms', 'Symptoms'): 'RELATED_TO',
            ('Symptoms', 'Treatments'): 'ALLEVIATED_BY',
            ('Symptoms', 'Risk Factors'): 'INDICATES',
            ('Symptoms', 'Diagnostic Tests'): 'DETECTED_BY',
            ('Symptoms', 'Body Parts'): 'AFFECTS',
            
            ('Treatments', 'Diseases'): 'TREATS',
            ('Treatments', 'Medications'): 'USED_WITH',
            ('Treatments', 'Symptoms'): 'ALLEVIATES',
            ('Treatments', 'Treatments'): 'SIMILAR_TO',
            ('Treatments', 'Risk Factors'): 'MANAGES',
            ('Treatments', 'Diagnostic Tests'): 'USED_IN',
            ('Treatments', 'Body Parts'): 'TARGETS',
            
            ('Risk Factors', 'Diseases'): 'IS_RISK_FACTOR_FOR',
            ('Risk Factors', 'Medications'): 'USED_FOR',
            ('Risk Factors', 'Symptoms'): 'INDICATES',
            ('Risk Factors', 'Treatments'): 'MANAGED_BY',
            ('Risk Factors', 'Risk Factors'): 'RELATED_TO',
            ('Risk Factors', 'Diagnostic Tests'): 'INCREASES_RISK_FOR',
            ('Risk Factors', 'Body Parts'): 'AFFECTS',
            
            ('Diagnostic Tests', 'Diseases'): 'DETECTS',
            ('Diagnostic Tests', 'Medications'): 'USED_IN',
            ('Diagnostic Tests', 'Symptoms'): 'DETECTED_BY',
            ('Diagnostic Tests', 'Treatments'): 'USED_IN',
            ('Diagnostic Tests', 'Risk Factors'): 'USED_IN',
            ('Diagnostic Tests', 'Diagnostic Tests'): 'RELATED_TO',
            ('Diagnostic Tests', 'Body Parts'): 'ASSESSES',
            
            ('Body Parts', 'Diseases'): 'AFFECTS',
            ('Body Parts', 'Medications'): 'AFFECTS',
            ('Body Parts', 'Symptoms'): 'AFFECTS',
            ('Body Parts', 'Treatments'): 'TARGETED_BY',
            ('Body Parts', 'Risk Factors'): 'AFFECTS',
            ('Body Parts', 'Diagnostic Tests'): 'ASSESSED_BY',
            ('Body Parts', 'Body Parts'): 'PART_OF',
        }
        return fixed_relationships.get((label1, label2), f"{label1, label2}")

    def get_reverse_relationship(self, relationship):
        reverse_map = {
            "HAS_SYMPTOM": "IS_SYMPTOM_OF",
            "AFFECTS": "IS_AFFECTED_BY",
            "DIAGNOSED_BY": "DIAGNOSES",
            "TREATED_WITH": "TREATS",
            "MANAGED_BY": "MANAGES",
            "HAS_RISK_FACTOR": "IS_RISK_FACTOR_FOR",
            "INDICATES": "IS_INDICATED_BY",
            "OCCURS_IN": "HAS_OCCURRENCE_OF",
            "USED_FOR": "HAS_USE",
            "CAUSES": "CAUSED_BY",
            "INVOLVES_MEDICATION": "IS_INVOLVED_IN_TREATMENT",
            "HAS_SIDE_EFFECT": "IS_SIDE_EFFECT_OF",
            "CONTRAINDICATED_FOR": "HAS_CONTRAINDICATION",
            "DETECTS": "IS_DETECTED_BY",
            "MEASURES": "IS_MEASURED_BY",
            "INCREASES_RISK_OF": "HAS_INCREASED_RISK_DUE_TO",
            "PART_OF": "HAS_PART",
            "CAN_BE_AFFECTED_BY": "CAN_AFFECT"
        }
        return reverse_map.get(relationship, relationship)

    def save_graph(self, graph):
        with open(self.kg_file, 'wb') as file:
            pickle.dump(graph, file)
        print(f"Graph saved to {self.kg_file}")

    def load_graph(self):
        with open(self.kg_file, 'rb') as file:
            self.graph = pickle.load(file)

    def view_node_metadata(self, node_name):
        node_data = self.graph.nodes.get(node_name, None)
        if node_data is not None:
            print(f"\nNode: {node_name}")
            source_articles = node_data.get('source_article', [])
            chunk_ids = node_data.get('chunk_id', [])
            print(f"Source Articles: {', '.join(source_articles) if source_articles else 'N/A'}")
            print(f"Chunk IDs: {', '.join(chunk_ids) if chunk_ids else 'N/A'}")
        else:
            print(f"Node '{node_name}' not found in the graph.")

    def list_all_occurrences(self):
        print("\nListing all node occurrences (source_article, chunk_id):")
        for node, data in self.graph.nodes(data=True):
            print(f"\nNode: {node}")
            source_articles = data.get('source_article', [])
            chunk_ids = data.get('chunk_id', [])
            print(f"Source Articles: {', '.join(source_articles) if source_articles else 'N/A'}")
            print(f"Chunk IDs: {', '.join(chunk_ids) if chunk_ids else 'N/A'}")

    def find_shortest_path_and_check_chunk_id(self, node1, node2):
        try:
            shortest_path = nx.shortest_path(self.graph, source=node1, target=node2)
            print(f"Shortest path between '{node1}' and '{node2}': {shortest_path}")
            chunk_ids_set = set()
            for node in shortest_path:
                node_data = self.graph.nodes.get(node)
                if node_data is not None:
                    chunk_ids_set.update(node_data.get('chunk_id', []))
            if len(chunk_ids_set) == 1:
                print(f"All nodes in the shortest path come from the same chunk ID: {chunk_ids_set}")
            else:
                print("The nodes in the shortest path come from different chunk IDs.")
        except nx.NetworkXNoPath:
            print(f"No path exists between '{node1}' and '{node2}' in the graph.")
        except Exception as e:
            print(f"An error occurred: {str(e)}")
    
    def search_entity_relationship(self, entity1, entity2, method):
        """Finds and returns a simple relationship path and a detailed source+chunk trace."""
        e1, e2 = entity1, entity2
        entity1 = entity1.lower()
        entity2 = entity2.lower()
        linked_path_src = ""
        pertubation_used = []
        betweenness_score = float("-inf")
        max_deg_value = float("-inf")
        important_entities = []

        # Case-insensitive node lookup
        entity1 = next((node for node in self.graph.nodes if node.lower() == entity1), None)
        entity2 = next((node for node in self.graph.nodes if node.lower() == entity2), None)

        if not entity1 or not entity2:
            print(f"One or both entities not found: '{e1}', '{e2}'")
            return None, None, None, []

        try:
            path = nx.shortest_path(self.graph, source=entity1, target=entity2)

            deg_centralities = nx.degree_centrality(self.graph)


            nodes = []
            max_deg_node = None
            if method == "edge" or method == "subpath":
                nodes_to_score = {"tmp" : 0}

                for i in range(len(path) - 1):
                    node1, node2 = path[i], path[i + 1] 

                    tmp_score = self.find_betweenness_score(node1, node2)
                    if tmp_score > max(nodes_to_score.values()):
                        nodes_to_score = {}
                        nodes_to_score[(node1, node2)] = tmp_score

                    nodes.append(node1)
                    nodes.append(node2)
                
                nodesBetweenness = list(nodes_to_score.keys())[0]  # Get the first (and only) key
                betweenness_score = max(nodes_to_score.values())
                print(betweenness_score)
                # print("exiting")
                # exit(1)
                node1b, node2b = nodesBetweenness                    # Unpack the tuple
                print(nodesBetweenness, node1b, node2b)

                res = f"\nPath found between '{entity1}' and '{entity2}':"
                linked_path_src = f"\nDetailed relationship trace between '{entity1}' and '{entity2}':"
            # print(nodes_to_score)
            # print(path)
            # exit(1)

            else:

                max_deg_node = None

                for i in range(len(path) - 1):
                    node1, node2 = path[i], path[i + 1]
                    if deg_centralities[node1] > max_deg_value:
                        max_deg_node = node1
                        max_deg_value = deg_centralities[node1]
                    if deg_centralities[node2] > max_deg_value:
                        max_deg_node = node2
                        max_deg_value = deg_centralities[node2]
                    nodes.append(node1)
                    nodes.append(node2)
                print(f"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa {max_deg_node}")
                print(nodes)
                # exit(1)

            ## target node should be already found here
            # pertubation_methods = ["node", "subpath", "edge", "test"]
            # test = []

            res = f"\nPath found between '{entity1}' and '{entity2}':"
            linked_path_src = f"\nDetailed relationship trace between '{entity1}' and '{entity2}':"

            for i in range(len(path) - 1):
                node1, node2 = path[i], path[i + 1]
                edge_data = self.graph[node1][node2]
                relationship = edge_data.get("relationship", "related to")
                source_article = edge_data.get("source_article", "N/A")
                chunk_id = edge_data.get("chunk_id", "N/A")

                label1 = self.graph.nodes[node1].get('label', 'Unknown')
                label2 = self.graph.nodes[node2].get('label', 'Unknown')

                removed_string = ""

                if method == "node":
                    node1, label1, node2, label2, removed_string, important_entities = self.remove_node(max_deg_node, node1, label1, node2, label2)
                    # print(important_entities)
                elif method == "subpath":
                    print("1")
                    if node1 == node1b and node2 == node2b: 
                        node1, label1, relationship, node2, label2, removed_string, important_entities = self.remove_subpath(max_deg_node, node1, label1, relationship, node2, label2)
                        # print(important_entities)
                elif method == "edge":
                    if node1 == node1b and node2 == node2b:
                        relationship, removed_string, important_entities = self.remove_edge(node1b, relationship, node2b)
                        # print(important_entities)
                else:
                    pass
                # else:
                #     for t in test:
                #         print(f"{BG_RED}{t}{RESET}")
                #     exit(1)
                # print("aaaa")

                res += f"\n - {node1} {relationship} {node2}"        

                # metrics here 
                linked_path_src += (
                    f"\n - {node1} ({label1}) --[{relationship}]--> "
                    f"{node2} ({label2}) | Source: {source_article}, Chunk ID: {chunk_id}"
                )
            
            if method == "node":
                pertubation_used.append(f"{removed_string}\n{method}: {res}\nnode centrality degree: {max_deg_value}")
            else:
                pertubation_used.append(f"{removed_string}\n{method}: {res}\nbetweenness score: {betweenness_score}")

                # test.append(f"{method}: {res}")    
            # print(linked_path_src)
            # print(res)
            # exit(1)

            print("ret")
            print(important_entities)
            return res, linked_path_src, pertubation_used, important_entities

        except nx.NetworkXNoPath:
            print(f"No relationship path found between '{entity1}' and '{entity2}'.")
            return None, None, None, []

    def remove_node(self, target, node1, label1, node2, label2):
        
        removed_string = ""
        important_entities = [target]

        if node1 == target:
            removed_string = f"Removed: {node1}{label1}"
            node1 = ""
            label1 = ""
        if node2 == target:
            removed_string = f"Removed: {node2}{label2}"
            node2 = ""
            label2 = ""

        return node1, label1 , node2, label2, removed_string, important_entities
    
    def remove_subpath(self, target, node1, label1, relationship, node2, label2):
        
        removed_string = ""
        important_entities = [node1, node2]

        # if node1 == target or node2 == target:
        removed_string = f"Removed: {node1}{label1}, {relationship}, {node2}{label2}"
        node1 = ""
        label1 = ""
        node2 = ""
        label2 = ""
        relationship = ""
        
        return node1, label1, relationship, node2, label2, removed_string, important_entities
    
    def remove_edge(self, node1, relationship, node2):
        important_entities = [node1, node2]

        removed_string = f"Removed: {relationship}"
        return "", removed_string, important_entities


    def print_graph_summary(self):
        """Prints the number of nodes and edges in the graph."""
        num_nodes = self.graph.number_of_nodes()
        num_edges = self.graph.number_of_edges()
        print(f"\nGraph Summary:")
        print(f"Total Nodes: {num_nodes}")
        print(f"Total Edges: {num_edges}")
    
    def find_top_nodes_by_edge_count(self, number_of_top_nodes):
        """Finds and prints the top 5 nodes with the most total edges using a max heap."""
        res = []

        if self.graph.number_of_nodes() == 0:
            print("The graph is empty.")
            return

        max_heap = []

        for node in self.graph.nodes:
            in_deg = self.graph.in_degree(node)
            out_deg = self.graph.out_degree(node)
            total_edges = in_deg + out_deg
            heapq.heappush(max_heap, (-total_edges, node, in_deg, out_deg))

        print("\nTop 5 Nodes by Total Edge Count (In-Degree + Out-Degree):")
        for _ in range(min(number_of_top_nodes, len(max_heap))):
            neg_total, node, in_deg, out_deg = heapq.heappop(max_heap)
            label = self.graph.nodes[node].get('label', 'Unknown')
            print(f"Node: {node}")
            print(f"  Label: {label}")
            print(f"  Total Edges: {-neg_total}")
            print(f"  In-Degree: {in_deg}")
            print(f"  Out-Degree: {out_deg}")
            print()
            res.append(node)

        return res
    
    def print_sample_triplets(self, count=10):
        """Prints sample triplets with their chunk IDs and source articles."""
        print(f"\nPrinting {count} sample triplets from the graph:")
        printed = 0
        for u, v, data in self.graph.edges(data=True):
            relationship = data.get('relationship', 'related_to')
            chunk_id = data.get('chunk_id', 'N/A')
            source_article = data.get('source_article', 'N/A')
            print(f"({u}) -[{relationship}]-> ({v}) | Chunk ID: {chunk_id}, Source Article: {source_article}")
            printed += 1
            if printed >= count:
                break
    
    def print_all_paths_with_metadata(self, source, target, max_paths=5):
        """Prints all simple paths (up to `max_paths`) from source to target with edge relationships and metadata."""
        source = next((node for node in self.graph.nodes if node.lower() == source.lower()), None)
        target = next((node for node in self.graph.nodes if node.lower() == target.lower()), None)

        if not source or not target:
            print(f"One or both entities not found: '{source}', '{target}'")
            return

        print(f"\nShowing up to {max_paths} paths from '{source}' to '{target}':\n")
        try:
            paths = nx.all_simple_paths(self.graph, source=source, target=target, cutoff=6)
            count = 0
            for path in paths:
                if count >= max_paths:
                    break
                print(f"Path {count+1}:")
                for i in range(len(path) - 1):
                    u, v = path[i], path[i+1]
                    edge_data = self.graph.get_edge_data(u, v)
                    # Handle case where multiple edges exist (as dict of dicts)
                    if isinstance(edge_data, dict) and 0 in edge_data:
                        for key in edge_data:
                            rel = edge_data[key].get('relationship', 'related_to')
                            chunk = edge_data[key].get('chunk_id', 'N/A')
                            source_article = edge_data[key].get('source_article', 'N/A')
                            print(f"  ({u}) -[{rel}]-> ({v}) | Chunk ID: {chunk}, Article: {source_article}")
                    else:
                        rel = edge_data.get('relationship', 'related_to')
                        chunk = edge_data.get('chunk_id', 'N/A')
                        source_article = edge_data.get('source_article', 'N/A')
                        print(f"  ({u}) -[{rel}]-> ({v}) | Chunk ID: {chunk}, Article: {source_article}")
                print()
                count += 1
        except nx.NetworkXNoPath:
            print("No path exists.")
    
    def remove_nodes(self, number_of_nodes_to_remove):

        nodes_to_remove = self.find_top_nodes_by_edge_count(number_of_nodes_to_remove)

        for node in nodes_to_remove:
            if node in self.graph:
                self.graph.remove_node(node)
                print(f"Removed node: {node}")
            else:
                print(f"Node not found: {node}")
    
    def calculate_edge_betweenness(self):
        edge_betweenness = nx.edge_betweenness_centrality(self.graph)

        with open("edge_betweenness.csv", "w", newline="") as f_csv:
            writer = csv.writer(f_csv)
            writer.writerow(["Edge", "Betweenness"])
            for edge, centrality in edge_betweenness.items():
                
                writer.writerow([edge, centrality])

    def load_betweenness_map(self):
        if not os.path.exists('edge_betweenness.csv'):
            print("📊 edge_betweenness.csv not found. Generating it now...")
            self.calculate_edge_betweenness()

        with open('edge_betweenness.csv', newline='') as csvfile:
            reader = csv.reader(csvfile)
            next(reader)
            for row in reader:
                self.betweennessMap[row[0]] = float(row[1])
    
    def find_betweenness_score(self, entity1, entity2):
        entities_to_string = f"('{entity1}', '{entity2}')"
        # if entities_to_string in self.betweennessMap:
        #     print("found", self.betweennessMap[entities_to_string])

        return self.betweennessMap[entities_to_string]
    
    def get_avg_node_degree(self):
        avg_in = sum(dict(self.graph.in_degree()).values()) / self.graph.number_of_nodes()
        avg_out = sum(dict(self.graph.out_degree()).values()) / self.graph.number_of_nodes()
        avg_total = sum(dict(self.graph.degree()).values()) / self.graph.number_of_nodes()

        print(f"Average in-degree: {avg_in}")
        print(f"Average out-degree: {avg_out}")
        print(f"Average total-degree: {avg_total}")

        return avg_in, avg_out, avg_total

    
    def get_number_of_chunks_entities_appear(self):
        """
        Returns a dictionary mapping each entity to the set of chunk IDs in which it appears.
        """
        freq_map = {}

        for u, v, data in self.graph.edges(data=True):
            # Get the source chunk ID from edge data
            chunk_id = data.get("chunk_id")
            if chunk_id is None:
                continue

            for entity in (u, v):
                if entity in freq_map:
                    freq_map[entity].add(chunk_id)
                else:
                    freq_map[entity] = {chunk_id}

        # Convert sets to counts if you only want counts
        entity_chunk_counts = {entity: len(chunks) for entity, chunks in freq_map.items()}
        print(entity_chunk_counts)

        return entity_chunk_counts
    
    def plot_chunks_per_entity(self, save_path="chunks_per_entity.png", top_n=500):
        """
        Plots and saves a vertical bar chart of the top N entities by chunk frequency.
        The figure width is scaled to fit many entities with minimal spacing.
        """
        chunk_counts = self.get_number_of_chunks_entities_appear()
        sorted_entities = sorted(chunk_counts.items(), key=lambda x: x[1], reverse=True)

        # Limit to top N for the plot
        top_entities = sorted_entities[:top_n]
        entities, counts = zip(*top_entities)

        # Use a tighter width-per-entity factor to fit more in less space
        plt.figure(figsize=(max(12, len(entities) * 0.1), 8))
        plt.bar(entities, counts, color="skyblue")
        plt.ylabel("Number of Unique Chunks")
        plt.xlabel("Entities")
        plt.title(f"Top {top_n} Entities by Chunk Appearance")
        plt.xticks(rotation=90, fontsize=6)  # Smaller font to fit more labels
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"Saved plot to {save_path}")
    
    def get_graph_label_percentage(self):
        res = {}
        instances = {}

        for node in self.graph.nodes:
            if self.graph.nodes[node].get('label', 'Unknown') in instances:
                instances[self.graph.nodes[node].get('label', 'Unknown')] += 1
            else:
                instances[self.graph.nodes[node].get('label', 'Unknown')] = 1
        
        print(instances)
        allval = sum(instances.values())

        for k, v in instances.items():
            print(f"{k}: {v / allval}")

        return 0



kgc = KnowledgeGraphCreator()
kgc.read_csv_and_create_graph()
kgc.load_graph()
# kgc.get_graph_label_percentage()
# kgc.calculate_edge_betweenness()
# kgc.load_betweenness_map()
# kgc.find_betweenness_score("Paroxysmal spells", "psychological disorders")
# kgc.remove_nodes(100)
kgc.print_graph_summary()
# ind, out, total = kgc.get_avg_node_degree()
# kgc.plot_chunks_per_entity("chunk_frequency_plot.png")
res = kgc.find_top_nodes_by_edge_count(100)
# kgc.print_sample_triplets(res)
# kgc.print_all_paths_with_metadata("Neuronal ischemia", "Myoclonic jerking")
# res, paths, links, important_entities = kgc.search_entity_relationship("Neuronal ischemia", "Myoclonic jerking", "edge")
# print("end:", important_entities)
# print(res)
# print()
# print(paths)


# Example usage:
# kgc = KnowledgeGraphCreator()
# kgc.read_csv_and_create_graph()
# kgc.load_graph()
# # kgc.list_all_occurrences()
# # kgc.view_node_metadata("Neuronal ischemia")
# kgc.find_shortest_path_and_check_chunk_id("Neuronal ischemia", "Myoclonic jerking")
