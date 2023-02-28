import pickle
from GraphClasses import ClipGraph
from MGAnnotations import MGAnnotations


# python 3 (should work on most versions!)
# numpy
# networkx=1.11  NOTE: networkx-2 will NOT work because they changed many functions!


with open('2017-11-02-51-7637_py3.pkl', 'rb') as fid:
    all_mg = pickle.load(fid, encoding='latin1')
    # latin1 is important here

# all_mg is a dictionary of MovieGraph objects
# indexed by imdb unique movie identifiers


num_movies = len(all_mg.keys())
print('Found {} movies with graphs'.format(num_movies))


# mg = all_mg['tt0822832']  # Marley and Me
mg = all_mg['tt0073486'] #One Flew Over the Cuckoo's Nest 1975
print('Selected movie: {}'.format(mg.imdb_key))
print()

print(type(mg))
print(mg.imdb_key)
print(mg.castlist)
#print(mg.mergers)
#print("\n Scene Gt : ")
#print(mg.scenes_gt)
#print(mg.clip_graphs[0] )
print(f"Total number of clip graphs: {len(mg.clip_graphs)}")
print(mg.clip_graphs)

my_ClipGraph = mg.clip_graphs[6] 
print("*"*20)
print("ClipGraph is a : ", type(my_ClipGraph))
print(my_ClipGraph.situation)
print(my_ClipGraph.scene_label)
print(my_ClipGraph.description)
print(type(my_ClipGraph.G))
my_ClipGraph.convert_to_nx_graph()
print("*"*20)
my_ClipGraph.pprint()
my_ClipGraph.visualize_graph()

print('Cast in this movie:')
for character in mg.castlist:
    print(character['chid'], character['name'])

# mg.clip_graphs is a list of ClipGraph objects
print()
print(f"Number of clip graphs: {len(mg.clip_graphs)}")
print('Selected one clip graph')
cg = mg.clip_graphs[4]
print(type(cg))
# cg.pprint()
# cg.visualize_graph()


