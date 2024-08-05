import pickle

# Pfad zur .pkl Datei
file_path = 'path_to_your_file.pkl'

# Öffnen der .pkl Datei
with open('./results_test/last.pkl', 'rb') as file:
    data = pickle.load(file)

# Anzeigen des Inhalts
print(data)