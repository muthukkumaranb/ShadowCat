"""
Unit tests for node mapping logic.
"""
import pytest
import numpy as np
from gnn.node_mapping import NodeMapper, GlobalNodeMapper


def test_node_mapper_fit_transform():
    mapper = NodeMapper()
    src_ips = np.array(["192.168.1.1", "192.168.1.2", "192.168.1.1"])
    dst_ips = np.array(["10.0.0.1", "10.0.0.2", "10.0.0.1"])

    mapper.fit(src_ips, dst_ips)
    assert mapper.num_nodes == 4  # 192.168.1.1, 192.168.1.2, 10.0.0.1, 10.0.0.2

    src_idx = mapper.transform(src_ips)
    dst_idx = mapper.transform(dst_ips)

    assert len(src_idx) == 3
    assert len(dst_idx) == 3
    assert src_idx[0] == src_idx[2]
    assert dst_idx[0] == dst_idx[2]


def test_node_mapper_bijective():
    mapper = NodeMapper()
    src_ips = np.array(["A", "B"])
    dst_ips = np.array(["C", "D"])
    mapper.fit(src_ips, dst_ips)

    for ip in ["A", "B", "C", "D"]:
        idx = mapper.get_idx(ip)
        assert mapper.get_ip(idx) == ip
