import pandas as pd
from dt import DT
from icecream import ic
# Load the CSV file
import os

path = './Data'
file_name = 'titanic-homework.csv'
full_path = os.path.join(path, file_name)

if os.path.exists(full_path):
    print(f"File {file_name} exists.")
    df = pd.read_csv(full_path)

    # Remove the 'Name' column from DataFrame, some data are unnecessary.
    df = df.drop('Name', axis=1)

    ############

    dt = DT(df=df, column_to_predict='Survived', ignore_columns=['PassengerId'])
    
    dupa=dt.visualize_tree()
    import json

    visualise = False  # domyślnie ustawione na False

    # Konwersja słownika na JSON
    wyniki = json.dumps(dupa)
    
    # Zapis JSON do pliku
    with open('wyniki.json', 'w') as file:
        file.write(wyniki)
    
    if visualise:
        import matplotlib.pyplot as plt
        import networkx as nx
        # Funkcja do tworzenia grafu z JSON
        def create_graph_from_json(json_str):
            data = json.loads(json_str)
            G = nx.DiGraph()
            
            def add_nodes_edges(data, parent=None):
                node_label = data.get('LEAF', data.get('NODE'))
                if parent:
                    G.add_edge(parent, node_label)
                else:
                    G.add_node(node_label)
                for child in data.get('children', []):
                    add_nodes_edges(child, node_label)
                    
            add_nodes_edges(data)
            return G
        
        # Tworzenie grafu
        G = create_graph_from_json(wyniki)
        
        # Rysowanie grafu
        pos = nx.spring_layout(G)
        nx.draw(G, pos, with_labels=True, arrows=True)
        plt.show()
else:
    print(f"File {file_name} does not exist.")


