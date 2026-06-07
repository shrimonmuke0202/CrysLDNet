"""Module to generate networkx graphs."""
"""Implementation based on the template of Matformer."""
from multiprocessing.context import ForkContext
from re import X
import numpy as np
import pandas as pd
from jarvis.core.specie import chem_data, get_node_attributes
import amd

from collections import defaultdict
from typing import List, Tuple, Sequence, Optional
import torch
from torch_geometric.data import Data
from torch_geometric.transforms import LineGraph
from torch_geometric.data.batch import Batch
import itertools
import pdb

try:
    import torch
    from tqdm import tqdm
except Exception as exp:
    print("torch/tqdm is not installed.", exp)
    pass


chemical_symbols = [
    # 0
    'X',
    # 1
    'H', 'He',
    # 2
    'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',
    # 3
    'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar',
    # 4
    'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
    'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr',
    # 5
    'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd',
    'In', 'Sn', 'Sb', 'Te', 'I', 'Xe',
    # 6
    'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy',
    'Ho', 'Er', 'Tm', 'Yb', 'Lu',
    'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi',
    'Po', 'At', 'Rn',
    # 7
    'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk',
    'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr',
    'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn', 'Nh', 'Fl', 'Mc',
    'Lv', 'Ts', 'Og']

def angle_from_array(a, b, lattice):
    a_new = np.dot(a, lattice)
    b_new = np.dot(b, lattice)
    assert a_new.shape == a.shape
    value = sum(a_new * b_new)
    length = (sum(a_new ** 2) ** 0.5) * (sum(b_new ** 2) ** 0.5)
    cos = value / length
    angle = np.arccos(cos)
    return angle / np.pi * 180.0

def correct_coord_sys(a, b, c, lattice):
    a_new = np.dot(a, lattice)
    b_new = np.dot(b, lattice)
    c_new = np.dot(c, lattice)
    assert a_new.shape == a.shape
    plane_vec = np.cross(a_new, b_new)
    value = sum(plane_vec * c_new)
    length = (sum(plane_vec ** 2) ** 0.5) * (sum(c_new ** 2) ** 0.5)
    cos = value / length
    angle = np.arccos(cos)
    return (angle / np.pi * 180.0 <= 90.0)

def same_line(a, b):
    a_new = a / (sum(a ** 2) ** 0.5)
    b_new = b / (sum(b ** 2) ** 0.5)
    flag = False
    if abs(sum(a_new * b_new) - 1.0) < 1e-5:
        flag = True
    elif abs(sum(a_new * b_new) + 1.0) < 1e-5:
        flag = True
    else:
        flag = False
    return flag

def same_plane(a, b, c):
    flag = False
    if abs(np.dot(np.cross(a, b), c)) < 1e-5:
        flag = True
    return flag


# pyg dataset
class PygStructureDataset(torch.utils.data.Dataset):
    """Dataset of crystal DGLGraphs."""

    def __init__(
        self,
        df: pd.DataFrame,
        graphs: Sequence[Data],
        target: str,
        atom_features="atomic_number",
        transform=None,
        line_graph=False,
        classification=False,
        id_tag="jid",
        neighbor_strategy="",
        nolinegraph=False,
        lineControl=True,
        mean_train=None,
        std_train=None,
        pre_train=False,
        masks=None,
        targets_mlm=None,
        targets_lattice=None,
        targets_position=None,
    ):
        """Pytorch Dataset for atomistic graphs.

        `df`: pandas dataframe from e.g. jarvis.db.figshare.data
        `graphs`: DGLGraph representations corresponding to rows in `df`
        `target`: key for label column in `df`
        """
        self.masks = masks
        self.targets_mlm = targets_mlm,
        self.targets_lattice=targets_lattice,
        self.targets_position=targets_position,
        self.df = df
        self.graphs = graphs
        self.target = target
        self.line_graph = line_graph
        self.pre_train = pre_train
        self.ids = self.df[id_tag]
        self.atoms = self.df['atoms']
        
        self.labels = torch.tensor(self.df[target]).type(
            torch.get_default_dtype()
        )
        print("mean %f std %f"%(self.labels.mean(), self.labels.std()))
        if mean_train == None:
            mean = self.labels.mean()
            std = self.labels.std()
            self.labels = (self.labels - mean) / std
            print("normalize using training mean but shall not be used here %f and std %f" % (mean, std))
        else:
            self.labels = (self.labels - mean_train) / std_train
            print("normalize using training mean %f and std %f" % (mean_train, std_train))

        self.transform = transform

        features = self._get_attribute_lookup(atom_features)

        

        self.prepare_batch = prepare_pyg_batch
        
        if line_graph and pre_train:
            self.prepare_batch = prepare_pyg_line_graph_batch_pre_train
            print("pre_train")
            if lineControl == False:
                self.line_graphs = []
                self.graphs = []
                for g in tqdm(graphs):
                    linegraph_trans = LineGraph(force_directed=True)
                    g_new = Data()
                    g_new.x, g_new.edge_index, g_new.edge_attr = g.x, g.edge_index, g.edge_attr
                    try:
                        lg = linegraph_trans(g)
                    except Exception as exp:
                        print(g.x, g.edge_attr, exp)
                        pass
                    lg.edge_attr = pyg_compute_bond_cosines(lg) # old cosine emb

                    self.graphs.append(g_new)
                    self.line_graphs.append(lg)
            else:
                if neighbor_strategy == "pairwise-k-nearest":
                    self.graphs = []
                    labels = []
                    idx_t = 0
                    filter_out = 0
                    max_size = 0
                    for g in tqdm(graphs):
                        g.edge_attr = g.edge_attr.float()
                        if g.x.size(0) > max_size:
                            max_size = g.x.size(0)
                        if g.x.size(0) < 200:
                            self.graphs.append(g)
                            labels.append(self.labels[idx_t])
                        else:
                            filter_out += 1
                        idx_t += 1
                    print("filter out %d samples because of exceeding threshold of 200 for nn based method" % filter_out)
                    print("dataset max atom number %d" % max_size)
                    self.line_graphs = self.graphs
                    self.labels = labels
                    self.labels = torch.tensor(self.labels).type(
                                    torch.get_default_dtype()
                                )
                else:
                    self.graphs = []
                    for g in tqdm(graphs):
                        g.edge_attr = g.edge_attr.float()
                        self.graphs.append(g)
                    self.line_graphs = self.graphs        
        elif line_graph:
            self.prepare_batch = prepare_pyg_line_graph_batch
            print("building line graphs")
            if lineControl == False:
                self.line_graphs = []
                self.graphs = []
                for g in tqdm(graphs):
                    linegraph_trans = LineGraph(force_directed=True)
                    g_new = Data()
                    g_new.x, g_new.edge_index, g_new.edge_attr = g.x, g.edge_index, g.edge_attr
                    try:
                        lg = linegraph_trans(g)
                    except Exception as exp:
                        print(g.x, g.edge_attr, exp)
                        pass
                    lg.edge_attr = pyg_compute_bond_cosines(lg) # old cosine emb
                    # lg.edge_attr = pyg_compute_bond_angle(lg)
                    self.graphs.append(g_new)
                    self.line_graphs.append(lg)
            else:
                if neighbor_strategy == "pairwise-k-nearest":
                    self.graphs = []
                    labels = []
                    idx_t = 0
                    filter_out = 0
                    max_size = 0
                    for g in tqdm(graphs):
                        g.edge_attr = g.edge_attr.float()
                        if g.x.size(0) > max_size:
                            max_size = g.x.size(0)
                        if g.x.size(0) < 200:
                            self.graphs.append(g)
                            labels.append(self.labels[idx_t])
                        else:
                            filter_out += 1
                        idx_t += 1
                    print("filter out %d samples because of exceeding threshold of 200 for nn based method" % filter_out)
                    print("dataset max atom number %d" % max_size)
                    self.line_graphs = self.graphs
                    self.labels = labels
                    self.labels = torch.tensor(self.labels).type(
                                    torch.get_default_dtype()
                                )
                else:
                    self.graphs = []
                    for g in tqdm(graphs):
                        g.edge_attr = g.edge_attr.float()
                        self.graphs.append(g)
                    self.line_graphs = self.graphs


        if classification:
            self.labels = self.labels.view(-1).long()
            print("Classification dataset.", self.labels)


    @staticmethod  
    def _get_attribute_lookup(atom_features: str = "cgcnn"):
        """Build a lookup array indexed by atomic number."""
        max_z = max(v["Z"] for v in chem_data.values())  

        # get feature shape (referencing Carbon)
        template = get_node_attributes("C", atom_features)

        features = np.zeros((1 + max_z, len(template)))

        for element, v in chem_data.items():
            z = v["Z"]
            x = get_node_attributes(element, atom_features)

            if x is not None:
                features[z, :] = x

        return features

    def __len__(self):
        """Get length."""
        return self.labels.shape[0]

    def __getitem__(self, idx):
        """Get StructureDataset sample."""
        # g = self.graphs[idx]
        # label = self.labels[idx]
        # pdb.set_trace()
        g = self.graphs[idx]
        label = self.labels[idx]
        return_dict={}
        if self.targets_mlm[0] is not None:
            return_dict["mask"] = self.masks[idx]
            return_dict["atoms"] = self.targets_mlm[0][idx]
        if self.targets_lattice[0] is not None:
            return_dict["lattice"] = self.targets_lattice[0][idx]
        if self.targets_position[0] is not None:
            return_dict["positions"] = self.targets_position[0][idx].t()
        
        if self.pre_train and self.line_graph:
            return g, self.line_graphs[idx], return_dict
        
        elif self.pre_train:
            return g, self.masks[idx], self.targets_mlm[0][idx]

        if self.line_graph:
            return g, self.line_graphs[idx], label, label

        return g, label

    @staticmethod
    def collate(samples: List[Tuple[Data, torch.Tensor]]):
        """Dataloader helper to batch graphs cross `samples`."""
        graphs, labels = map(list, zip(*samples))
        batched_graph = Batch.from_data_list(graphs)
        return batched_graph, torch.tensor(labels)

    @staticmethod
    def collate_line_graph(
        samples: List[Tuple[Data, Data, torch.Tensor, torch.Tensor]]
    ):
        """Dataloader helper to batch graphs cross `samples`."""
        graphs, line_graphs, lattice, labels = map(list, zip(*samples))
        batched_graph = Batch.from_data_list(graphs)
        batched_line_graph = Batch.from_data_list(line_graphs)
        if len(labels[0].size()) > 0:
            return batched_graph, batched_line_graph, batched_line_graph, torch.stack(labels)
        else:
            return batched_graph, batched_line_graph, batched_line_graph, torch.tensor(labels)
        
        
    @staticmethod
    def collate_line_graph_pretrain(
        samples: List[Tuple[Data, Data, dict]]
    ):
        # pdb.set_trace()
        graphs, line_graphs, return_dict = map(list, zip(*samples))
        batched_graph = Batch.from_data_list(graphs)
        batched_line_graph = Batch.from_data_list(line_graphs)
        target_dict = {}
        if "mask" in return_dict[0]:
            target_mask = []
            target_atom = []
            target_dict["mask"] = target_mask
            target_dict["atoms"] = target_atom
        if "positions" in return_dict[0]:
            target_position = []
            target_dict["positions"] = target_position
        if "lattice" in return_dict[0]:
            target_lattice = []
            target_dict["lattice"] = target_lattice
        for key in target_dict.keys():
            for item in return_dict:
                #print(key, type(item[key]))
                target_dict[key].append(item[key])
            target_dict[key] = torch.hstack(target_dict[key])
        return batched_graph, batched_line_graph, target_dict

def canonize_edge(
    src_id,
    dst_id,
    src_image,
    dst_image,
):
    """Compute canonical edge representation.

    Sort vertex ids
    shift periodic images so the first vertex is in (0,0,0) image
    """
    # store directed edges src_id <= dst_id
    if dst_id < src_id:
        src_id, dst_id = dst_id, src_id
        src_image, dst_image = dst_image, src_image

    # shift periodic images so that src is in (0,0,0) image
    if not np.array_equal(src_image, (0, 0, 0)):
        shift = src_image
        src_image = tuple(np.subtract(src_image, shift))
        dst_image = tuple(np.subtract(dst_image, shift))

    assert src_image == (0, 0, 0)

    return src_id, dst_id, src_image, dst_image


def nearest_neighbor_edges_submit(
    atoms=None,
    cutoff=8,
    max_neighbors=12,
    id=None,
    use_canonize=False,
    use_lattice=False,
    use_angle=False,
):
    """Construct k-NN edge list."""
    # returns List[List[Tuple[site, distance, index, image]]]
    lat = atoms.lattice
    all_neighbors_now = atoms.get_all_neighbors(r=cutoff)
    min_nbrs = min(len(neighborlist) for neighborlist in all_neighbors_now)

    attempt = 0
    if min_nbrs < max_neighbors:
        lat = atoms.lattice
        if cutoff < max(lat.a, lat.b, lat.c):
            r_cut = max(lat.a, lat.b, lat.c)
        else:
            r_cut = 2 * cutoff
        attempt += 1
        return nearest_neighbor_edges_submit(
            atoms=atoms,
            use_canonize=use_canonize,
            cutoff=r_cut,
            max_neighbors=max_neighbors,
            id=id,
            use_lattice=use_lattice,
        )

    edges = defaultdict(set)
    # lattice correction process
    r_cut = max(lat.a, lat.b, lat.c) + 1e-2
    all_neighbors = atoms.get_all_neighbors(r=r_cut)
    neighborlist = all_neighbors[0]
    neighborlist = sorted(neighborlist, key=lambda x: x[2])
    ids = np.array([nbr[1] for nbr in neighborlist])
    images = np.array([nbr[3] for nbr in neighborlist])
    images = images[ids == 0]
    lat1 = images[0]
    # finding lat2
    start = 1
    for i in range(start, len(images)):
        lat2 = images[i]
        if not same_line(lat1, lat2):
            start = i
            break
    # finding lat3
    for i in range(start, len(images)):
        lat3 = images[i]
        if not same_plane(lat1, lat2, lat3):
            break
    # find the invariant corner
    if angle_from_array(lat1,lat2,lat.matrix) > 90.0:
        lat2 = - lat2
    if angle_from_array(lat1,lat3,lat.matrix) > 90.0:
        lat3 = - lat3
    # find the invariant coord system
    if not correct_coord_sys(lat1, lat2, lat3, lat.matrix):
        lat1 = - lat1
        lat2 = - lat2
        lat3 = - lat3

    atom_coords = atoms.coords
    
    n_atoms = len(atom_coords)

    for site_idx, neighborlist in enumerate(all_neighbors_now):

        # sort on distance
        neighborlist = sorted(neighborlist, key=lambda x: x[2])
        distances = np.array([nbr[2] for nbr in neighborlist])
        ids = np.array([nbr[1] for nbr in neighborlist])
        images = np.array([nbr[3] for nbr in neighborlist])

        # find the distance to the k-th nearest neighbor
        max_dist = distances[max_neighbors - 1]
        ids = ids[distances <= max_dist]
        images = images[distances <= max_dist]
        distances = distances[distances <= max_dist]
        for dst, image in zip(ids, images):
            src_id, dst_id, src_image, dst_image = canonize_edge(
                site_idx, dst, (0, 0, 0), tuple(image)
            )
            if use_canonize:  # True
                edges[(src_id, dst_id)].add(dst_image)
            else:
                edges[(site_idx, dst)].add(tuple(image))

        if use_lattice:
            edges[(site_idx, site_idx)].add(tuple(lat1))
            edges[(site_idx, site_idx)].add(tuple(lat2))
            edges[(site_idx, site_idx)].add(tuple(lat3))

    return edges, lat1, lat2, lat3


def compute_bond_cosine(v1, v2):
    """Compute bond angle cosines from bond displacement vectors."""
    v1 = torch.tensor(v1).type(torch.get_default_dtype())
    v2 = torch.tensor(v2).type(torch.get_default_dtype())
    bond_cosine = torch.sum(v1 * v2) / (
        torch.norm(v1) * torch.norm(v2)
    )
    bond_cosine = torch.clamp(bond_cosine, -1, 1)
    return bond_cosine


def build_undirected_edgedata(
    atoms=None,
    edges={},
    a=None,
    b=None,
    c=None,
):
    """Build undirected graph data from edge set.

    edges: dictionary mapping (src_id, dst_id) to set of dst_image
    r: cartesian displacement vector from src -> dst
    """
    # second pass: construct *undirected* graph
    # import pprint
    u, v, r, l, nei, angle, atom_lat = [], [], [], [], [], [], []
    v1, v2, v3 = atoms.lattice.cart_coords(a), atoms.lattice.cart_coords(b), atoms.lattice.cart_coords(c)
    # atom_lat.append([v1, v2, v3, -v1, -v2, -v3])
    atom_lat.append([v1, v2, v3])
    for (src_id, dst_id), images in edges.items():

        for dst_image in images:
            # fractional coordinate for periodic image of dst
            dst_coord = atoms.frac_coords[dst_id] + dst_image
            # cartesian displacement vector pointing from src -> dst
            d = atoms.lattice.cart_coords(
                dst_coord - atoms.frac_coords[src_id]
            )
            for uu, vv, dd in [(src_id, dst_id, d), (dst_id, src_id, -d)]:
                u.append(uu)
                v.append(vv)
                r.append(dd)
                nei.append([v1, v2, v3])

    u = torch.tensor(u)  
    v = torch.tensor(v)  
    r = torch.tensor(np.array(r)).type(torch.get_default_dtype())  
    l = torch.tensor(l).type(torch.int)  
    nei = torch.tensor(np.array(nei)).type(torch.get_default_dtype())  
    atom_lat = torch.tensor(np.array(atom_lat)).type(torch.get_default_dtype())  
    
    return u, v, r, l, nei, atom_lat

def distance(pos1, pos2, lattice_matrix):
    
    delta = pos2 - pos1
    delta = np.dot(delta, np.linalg.inv(lattice_matrix))
    delta -= np.round(delta)
    delta = np.dot(delta, lattice_matrix)
    dist = np.linalg.norm(delta)
    return dist

class PygGraph(object):
    """Generate a graph object."""

    def __init__(
        self,
        nodes=[],
        node_attributes=[],
        edges=[],
        edge_attributes=[],
        color_map=None,
        labels=None,
    ):
        """
        Initialize the graph object.

        Args:
            nodes: IDs of the graph nodes as integer array.

            node_attributes: node features as multi-dimensional array.

            edges: connectivity as a (u,v) pair where u is
                   the source index and v the destination ID.

            edge_attributes: attributes for each connectivity.
                             as simple as euclidean distances.
        """
        self.nodes = nodes
        self.node_attributes = node_attributes
        self.edges = edges
        self.edge_attributes = edge_attributes
        self.color_map = color_map
        self.labels = labels

    @staticmethod
    def atom_dgl_multigraph(
        atoms=None,
        neighbor_strategy="k-nearest",
        cutoff=4.0,
        max_neighbors=12,
        atom_features="cgcnn",
        max_attempts=3,
        id: Optional[str] = None,
        compute_line_graph: bool = True,
        use_canonize: bool = False,
        use_lattice: bool = False,
        use_angle: bool = False,
    ):
        lattice_matrix = atoms.lattice_mat
        atom_coords = atoms.coords
        
        n_atoms = len(atom_coords)
        


        if neighbor_strategy == "k-nearest":
            edges, a, b, c = nearest_neighbor_edges_submit(
                atoms=atoms,
                cutoff=cutoff,
                max_neighbors=max_neighbors,
                id=id,
                use_canonize=use_canonize,
                use_lattice=use_lattice,
                use_angle=use_angle,
            )
            u, v, r, l, nei, atom_lat = build_undirected_edgedata(atoms, edges, a, b, c)
        else:
            raise ValueError("Not implemented yet", neighbor_strategy)


        # build up atom attribute tensor
        sps_features = []
        sps_features_ = []
        
        for ii, s in enumerate(atoms.elements):
            feat = list(get_node_attributes(s, atom_features=atom_features))
            sps_features_.append(feat)
        
        for ii, s in enumerate(atoms.elements):
            one_hot = np.zeros(119)
            one_hot[chemical_symbols.index(s)] = 1
            feat = list(one_hot)
            sps_features.append(feat)
        
        sps_features_ = np.array(sps_features_)
        node_features_ = torch.tensor(sps_features_).type(
            torch.get_default_dtype()
        )
        
        sps_features = np.array(sps_features)
        node_features = torch.tensor(sps_features).type(
            torch.get_default_dtype()
        )

        node_features_ = torch.nan_to_num(node_features_, nan=0.0)

        total_sum = node_features_.sum().float()
        atomic_weight = node_features_.float() / total_sum

        k = 92

        periodic_sets = amd.periodicset_from_pymatgen_structure(atoms.pymatgen_converter())

        pdd = amd.PDD(
                periodic_sets,             
                k,                         
                lexsort=False,              
                collapse=False,             
                collapse_tol=1e-4,         
            )

        atomic_weight = atomic_weight.reshape(n_atoms,1)

        pdd[:, 0] = atomic_weight[:, 0]
        
        pdd = torch.tensor(pdd.astype(np.float32))

        atom_lat = atom_lat.repeat(node_features.shape[0], 1, 1)
        edge_index = torch.cat((u.unsqueeze(0), v.unsqueeze(0)), dim=0).long()
        g = Data(x=node_features, edge_index=edge_index, edge_attr=r, adj=pdd,atomic_n=node_features_)
        
        return g


def prepare_pyg_batch(
    batch: Tuple[Data, torch.Tensor], device=None, non_blocking=False
):
    """Send batched dgl crystal graph to device."""

    g, t = batch
    batch = (
        g.to(device),
        t.to(device, non_blocking=non_blocking),
    )

    return batch

def prepare_pyg_line_graph_batch_pre_train(
    batch: Tuple[Tuple[Data, Data], dict],
    device=None,
    non_blocking=False,
):
    """Send line graph batch to device.

    Note: the batch is a nested tuple, with the graph and line graph together
    """
    g, lg, return_dict = batch
    for i, item in enumerate(return_dict):
        return_dict[item]  = return_dict[item].to(device, non_blocking=non_blocking)
    batch = (
        (
            g.to(device),
            lg.to(device),
        ),
        (
            return_dict,
        )
    )

    return batch

def prepare_pyg_line_graph_batch(
    batch: Tuple[Tuple[Data, Data, torch.Tensor], torch.Tensor],
    device=None,
    non_blocking=False,
):
    """Send line graph batch to device.

    Note: the batch is a nested tuple, with the graph and line graph together
    """
    g, lg, lattice, t = batch
    batch = (
        (
            g.to(device),
            lg.to(device),
            lattice.to(device, non_blocking=non_blocking),
        ),
        t.to(device, non_blocking=non_blocking),
    )

    return batch

