import struct

import numpy as np

REGULAR = np.array(
    [[0, 0, 0], [1, 0, 0], [0.5, np.sqrt(3) / 2, 0], [0.5, np.sqrt(3) / 6, np.sqrt(2 / 3)]], float
)


def write_gmsh(path, vertices, binary=True, endian="<"):
    ids = np.arange(len(vertices), dtype=int) * 7 + 10
    with open(path, "wb") as h:
        h.write(f"$MeshFormat\n2.2 {int(binary)} 8\n".encode())
        if binary:
            h.write(struct.pack(endian + "i", 1) + b"\n")
        h.write(f"$EndMeshFormat\n$Nodes\n{len(vertices)}\n".encode())
        for i, v in zip(ids, vertices):
            h.write(
                struct.pack(endian + "i3d", int(i), *v)
                if binary
                else (f"{i} " + " ".join(map(str, v)) + "\n").encode()
            )
        if binary:
            h.write(b"\n")
        n = len(vertices) // 4
        h.write(f"$EndNodes\n$Elements\n{n}\n".encode())
        if binary:
            h.write(struct.pack(endian + "3i", 4, n, 2))
        for j in range(n):
            row = [j + 100, j + 1, j + 1, *ids[4 * j : 4 * j + 4]]
            h.write(
                struct.pack(endian + "7i", *row)
                if binary
                else (
                    " ".join(map(str, [j + 100, 4, 2, j + 1, j + 1, *ids[4 * j : 4 * j + 4]]))
                    + "\n"
                ).encode()
            )
        if binary:
            h.write(b"\n")
        h.write(b"$EndElements\n")
    return path
