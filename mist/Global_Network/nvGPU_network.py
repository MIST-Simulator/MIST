import csv
from dataclasses import dataclass
from typing import Optional
import enum
import math
import pandas as pd

# Inspired from Calculon.

class NetworkConnectionType(enum.Enum):
    NODE = enum.auto() # node-level
    RACK = enum.auto() # rack-level
    POD = enum.auto() # pod-level or a cabinet group of racks
    # TODO: More types to be used

@dataclass
class GPUNetworkSpec:
    intra_node_bw: Optional[float] = None # In GB/s
    intra_node_latency: Optional[float] = None # In msec
    intra_node_sz: Optional[int] = None # E.g. Number of GPUs inside a node
    intra_node_eff: Optional[float] = None
    inter_node_bw: Optional[float] = None # In GB/s
    inter_node_latency: Optional[float] = None # In msec
    inter_node_eff: Optional[float] = None
    intra_rack_size: Optional[float] = None # 8 nodes inside a rack
    inter_rack_bw: Optional[float] = None #  nodes inside a rack
    inter_rack_latency: Optional[float] = None #  nodes inside a rack
    inter_rack_eff: Optional[float] = None # Nodes per rack

    # inter node values are also for intra rack s


def get_network_spec(machine_model:str):
    if machine_model == "H100_nvl8":
        network_spec = GPUNetworkSpec()
        # Values initialization inside a node
        network_spec.intra_node_bw = 450
        network_spec.intra_node_latency = 0.01
        network_spec.intra_node_sz = 8
        network_spec.intra_node_eff = 0.65 # We have more accurate data from trace

        # Values initialization inside a rack (between nodes)
        network_spec.inter_node_bw = 50
        network_spec.inter_node_latency = 0.02
        network_spec.inter_node_eff = 0.9
        network_spec.intra_rack_size = 8 #nodes inside a rack

        # Values initialization between racks
        network_spec.inter_rack_bw = 12.5 #  1/4 of internode bw
        network_spec.inter_rack_latency = 0.2
        network_spec.inter_rack_eff = 0.6
    elif machine_model == "A100_nvl8":
        network_spec = GPUNetworkSpec()
        network_spec.intra_node_bw = 300
        network_spec.intra_node_latency = 0.01
        network_spec.intra_node_sz = 8
        network_spec.intra_node_eff = 0.65 # We have more accurate data from trace

        # Values initialization inside a rack (between nodes)
        network_spec.inter_node_bw = 25
        network_spec.inter_node_latency = 0.02
        network_spec.inter_node_eff = 0.9
        network_spec.intra_rack_size = 8 # Nodes per rack

        # Values initialization between racks
        network_spec.inter_rack_bw = 6.25 # 1/4 of the internode
        network_spec.inter_rack_latency = 0.2
        network_spec.inter_rack_eff = 0.6
    else:
        network_spec = GPUNetworkSpec() # Return default None

    return network_spec




def get_network_bw_between_engines(global_size, network_spec: GPUNetworkSpec, intra_domain_sz:int):
    data_cols = ['src', 'dst', 'latency(msec)', 'BW(GB/s)']
    data = []
    """
        Determine the engine level communication bw and latency value for the three tiered hierarchy
        of network connections
    """
    lowest_connection = NetworkConnectionType.NODE

    # Determine the starting unit of communication (reduced by the intra domain size)
    intra_node_group = network_spec.intra_node_sz // intra_domain_sz

    if intra_node_group <= 1:
        # If the already assigned intra-domain is already larger than intra node
        intra_rack_group = network_spec.intra_rack_size*network_spec.intra_node_sz // intra_domain_sz
        lowest_connection = NetworkConnectionType.RACK
    else:
        # Then the intra_rack
        intra_rack_group = network_spec.intra_rack_size * intra_node_group


    if intra_rack_group <= 1:
        # This means all the engines should be connected with inter rack
        lowest_connection = NetworkConnectionType.POD

    print(f"lowest-level connection: {lowest_connection}")

    # Assume sending i to j is the same as sending j to i
    for i in range(global_size):
        for j in range(i+1, global_size):
            if (lowest_connection == NetworkConnectionType.NODE
                and ((i // intra_node_group) == (j // intra_node_group))):
                # Within node-level groups
                bw = network_spec.intra_node_bw * network_spec.intra_node_eff
                latency = network_spec.intra_node_latency
            elif ((lowest_connection == NetworkConnectionType.RACK) or (lowest_connection == NetworkConnectionType.NODE)) \
                  and  (i // intra_rack_group) == (j // intra_rack_group):
                # Within rack-level groups
                bw = network_spec.inter_node_bw * network_spec.inter_node_eff
                latency = network_spec.inter_node_latency
            else:
                # Outside rack, at pod level
                bw = network_spec.inter_rack_bw * network_spec.inter_rack_eff
                latency = network_spec.inter_rack_latency

            # Append bidirectional connections
            cur_data = [
                {"src": i, "dst": j, "latency(msec)": latency, "BW(GB/s)": bw},
                {"src": j, "dst": i, "latency(msec)": latency, "BW(GB/s)": bw},
            ]
            data.extend(cur_data)

    # Add self connection
    for i in range(global_size):
        data.append({"src": i, "dst": i, "latency(msec)": 0, "BW(GB/s)": 0})
    data = sorted(data, key=lambda x: x["src"])

    df = pd.DataFrame(data, columns=data_cols)

    return df
    # Save to CSV
    csv_filename = "network_bw.csv"
    with open(csv_filename, mode="w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=data_cols)
        writer.writeheader()
        writer.writerows(data)

    print(f"CSV file '{csv_filename}' created successfully.")


if __name__ == "__main__":
    network_spec = get_network_spec("H100_nvl8")
    global_size = 16    ## Total number of GPUs = Num Engines * TP

    # Value used for the already used intra domain parallelization
    intra_domain_sz = 4 # This can also be treated as TP or others

    # Group the size in the intra-domain
    global_size = math.ceil(global_size / intra_domain_sz)

    get_network_bw_between_engines(global_size, network_spec, intra_domain_sz)