import pandas as pd
from dt import DT
from icecream import ic
# Load the CSV file
import os

path = './Data'
file_name = 'przedmioty.csv'
full_path = os.path.join(path, file_name)

if os.path.exists(full_path):
    print(f"File {file_name} exists.")
    df = pd.read_csv(full_path)

    # Remove the 'Name' column from DataFrame, some data are unnecessary.
    #df = df.drop('Name', axis=1)

    ############

    dt = DT(df=df, column_to_predict='Decyzja', ignore_columns=['Uczeń'])
    dupa=dt.visualize_tree()
    import json

    visualise = False  # domyślnie ustawione na False

    # Konwersja słownika na JSON
    dupa_json = json.dumps(dupa)
    
    # Zapis JSON do pliku
    with open('wynik_dupa.json', 'w') as file:
        file.write(dupa_json)
    
    if visualise:
        import matplotlib.pyplot as plt
        import networkx as nx
        # Funkcja do tworzenia grafu z JSON
        def create_graph_from_json(json_str):
            data = json.loads(json_str)
            G = nx.DiGraph()
            
            def add_nodes_edges(data, parent=None):
                if parent is not None:
                    G.add_edge(parent, data['name'])
                else:
                    G.add_node(data['name'])
                for child in data.get('children', []):
                    add_nodes_edges(child, data['name'])
                    
            add_nodes_edges(data)
            return G
        
        # Tworzenie grafu
        G = create_graph_from_json(dupa_json)
        
        # Rysowanie grafu
        pos = nx.spring_layout(G)
        nx.draw(G, pos, with_labels=True, arrows=True)
        plt.show()
else:
    print(f"File {file_name} does not exist.")


