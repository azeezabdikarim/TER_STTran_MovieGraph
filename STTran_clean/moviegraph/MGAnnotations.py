import re
import json
import pickle
# from moviegraph.GraphClasses import ClipGraph
# python 3 (should work on most versions!)
# numpy
# networkx=1.11 # NOTE: networkx-2 will NOT work because they changed many functions!
import os
print(os.getcwd())

class MGAnnotations:
    def __init__(self, annotation_path):
        with open(annotation_path, 'rb') as fid:
            all_mg = pickle.load(fid, encoding='latin1')
            self.mg = all_mg['tt0073486'] #One Flew Over the Cuckoo's Nest 1975
            print(f"Total number of clip graphs: {len(self.mg.clip_graphs)}")
           
            self.mg_mapping = movieGraphSceneMapping(self.mg)
            self.ag_to_mg_pred_mapping = {
                "lookingat": {"looks at", "watches", "observes", "glances at", "sees", "stares at", "spots"},
                "notlookingat": {"ignores", "averts gaze", "turns away from", "looks away from"},
                "unsure": {"uncertain", "unsure", "confused", "puzzled", "doubtful", "perplexed", 
                           "hesitant", "unclear", "unconvinced", "indecisive"},
                "above": {"above", "over", "higher than", "superior", "on top of", "up"},
                "beneath": {"beneath", "below", "lower than", "inferior", "under", "down"},
                "infrontof": {"in front of", "facing", "ahead of", "before", "forefront"},
                "behind": {"behind", "at the back of", "trailing", "rear"},
                "onthesideof": {"beside", "next to", "adjacent to", "alongside", "on the side of"},
                "in": {"inside", "within", "in", "in the middle of", "in the center of", "in the midst of", "amidst"},
                "carrying": {"carrying", "transporting", "holding", "bearing"},
                "coveredby": {"covered by", "hidden under", "camouflaged by", "obscured by"},
                "drinkingfrom": {"drinking from", "sipping from", "imbibing from", "swigging from", "quaffing from"},
                "eating": {"eating", "chewing", "consuming", "devouring"},
                "haveitontheback": {"wearing on the back", "carrying on the back", "having on the back", "sporting on the back"},
                "holding": {"holding", "gripping", "clutching", "grasping", "embracing"},
                "leaningon": {"leaning on", "resting on", "supported by", "propped up by", "relying on"},
                "lyingon": {"lying on", "resting on", "stretched out on", "lying prone on"},
                "notcontacting": {"not contacting", "not touching", "not reaching", "out of reach of", "too far away from"},
                "otherrelationship": {"related in other ways", "associated in other ways", "connected in other ways", 
                                      "linked in other ways", "affiliated in other ways"},
                "sittingon": {"sitting on", "perched on", "astride", "straddling", "seated on"},
                "standingon": {"standing on", "perched on", "upright on", "balanced on"},
                "touching": {"touching", "in contact with", "grazing", "brushing", "pressing against"},
                "twisting": {"twisting", "contorting", "wringing", "writhing", "torquing"},
                "wearing": {"wearing", "sporting", "having on", "clad in", "dressed in"},
                "wiping": {"wiping", "drying", "cleaning", "clearing", "sponging"}
            }
            self.mg_to_ag_pred_mapping = {}

            for mk in self.ag_to_mg_pred_mapping.keys():
                for v in self.ag_to_mg_pred_mapping[mk]:
                    self.mg_to_ag_pred_mapping[v] = mk
            
            self.scene_triplets = {}
            for k in self.mg_mapping.keys():
                clip = self.mg_mapping[k]
                sc_triplets = getClipTriplets(clip, mapping = self.mg_to_ag_pred_mapping)
                self.scene_triplets[k] = sc_triplets
                
            self.total_triplets = 0
            self.triplets_w_predicate_match = 0
            for k in self.scene_triplets.keys():
                st = self.scene_triplets[k]
                for time in st.keys():
                    trips = st[time]
                    for t in trips:
                        self.total_triplets += 1
                        if "__" in t[1]:
                            self.triplets_w_predicate_match += 1
            print(f"Out of the {self.total_triplets} total triplets constructed in for the movie, {self.triplets_w_predicate_match} have a predicate match with ActionGenome")


    def getAnnotation(self, scene_id):
        return self.scene_triplets[scene_id]
    
    def keys(self):
        return self.scene_triplets.keys()

def getClipTriplets(clip, mapping = None):
    # time nodes have no neighbors, they are children nodes to a root node. so to build an interaction graph we find 
    # all the time nodes, and then iterate through the node graph to find which nodes have time nodes as children
    # we call the nodes that are found 'root' nodes
    root_nodes_of_time = []
    clip.convert_to_nx_graph()
    for node_id in clip.G.nodes:
        time_neighbors = list(clip.get_neighbors(node_id, ntypes = 'time'))
        if len(time_neighbors) != 0:
            root_nodes_of_time.append(node_id)

    #root nodes only have direction in one way. They seem almost like predicates linking to subject2, but subject1 is missing
    # so we follow the same methodlogy as above, the find node that have the 'root' node as a child
    # the new nodes found are called root precursor nodes 
    root_node_precursors = {}
    for node_id in clip.G.nodes:
        node_neighbors = list(clip.get_neighbors(node_id))
        for root in root_nodes_of_time:
            if root in node_neighbors:
                if root in root_node_precursors.keys():
                    root_node_precursors[root].append(node_id)
                else:
                    root_node_precursors[root] = [node_id]

    mini_scenes = {}
    for root_id in root_nodes_of_time:
        root_node = clip.G.nodes[root_id]
        mini_scenes[root_id] = {}
        mini_scenes[root_id]['sub1'] = []
        for n in root_node_precursors[root_id]:
            mini_scenes[root_id]['sub1'].append(clip.G.nodes[n])
    #         print("Precursor Node to the Root Time Node:\n", prettyPrintDict(clip.G.nodes[n],1))
        mini_scenes[root_id]['sub2'] = []
        for n in clip.get_neighbors(root_id):
            mini_scenes[root_id]['sub2'].append(clip.G.nodes[n])
    return getMiniSceneTriplets(clip, mini_scenes, mapping)


def getMiniSceneTriplets(clip, mini_scenes, mapping = None):
    mini_scene_triplets = {}
    for key in mini_scenes.keys():
        values = mini_scenes[key]
        predicate = clip.G.nodes[key]
        relationship = predicate['origtext']
        if mapping != None:
            if relationship in mapping.keys():
                relationship = f"{mapping[relationship]}"
            else:
                relationship = f'{relationship}'
        subj1 = values['sub1']
        subj2 = values['sub2']
        triplets = []
        time = ''
        for s1 in subj1:
            if s1['type'] == 'entity':
                s1_subj = 'person'
            else:
                s1_subj = s1['origtext']

            for s2 in subj2:
                if s2['type'] == 'entity':
                        s2_subj = 'person'
                else:
                        s2_subj = s2['origtext']
                if s2['type'] != 'time':
                    trip = [s1_subj, relationship, s2_subj]
                    triplets.append(trip)
                else:
                    time = s2['origtext']
        mini_scene_triplets[time] = triplets
    return mini_scene_triplets
    
def extract_scene_number(cg):
    file_name = cg.video['fname'][0]
    match = re.search(r'scene-(\d+)\.', file_name)
    if match:
        return int(match.group(1))
    else:
        return -1

def movieGraphSceneMapping(mg):
    mapping = {}
    for i in range(len(mg.clip_graphs)):
        try:
            clip_graph = mg.clip_graphs[i]
            scene_num = extract_scene_number(clip_graph)
            mapping[scene_num] = clip_graph
        except KeyError:
            continue
    return mapping

def getSetOfRelationships(mg):
    relationships = {}

    for clip_id in mg.clip_graphs:
        clip = mg.clip_graphs[clip_id]
        clip.convert_to_nx_graph()
        for n in clip.G.nodes():
#             print(clip.G.nodes[n].keys())
            node_type = clip.G.nodes[n]['type']
            name = clip.G.nodes[n]['name']
            if node_type in relationships.keys():
                relationships[node_type].add(name)
            else:
                relationships[node_type] = set()
                relationships[node_type].add(name)
#             print("\t",clip.G.nodes[n]['type'], clip.G.nodes[n]['name'], 
#                   clip.G.nodes[n]['origtext'])
#         print(clip.get_node_type_dict())
    return relationships