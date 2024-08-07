type Vector = [f32];
type Edges = [u32];
type SerializedVector = [u8];
type SerializedEdges = [u8];

pub fn serialize_node(vector: &Vector, edges: &Edges) -> Vec<u8> {
    let serialize_vector: &[u8] = serialize_vector(vector);
    let (serialize_edges, serialize_edges_len) = serialize_edges(edges);
    let mut combined = Vec::with_capacity(serialize_vector.len() + serialize_edges.len());
    combined.extend_from_slice(serialize_vector);
    combined.extend_from_slice(&serialize_edges_len);
    combined.extend_from_slice(serialize_edges);

    combined
}

pub fn serialize_vector(vector: &Vector) -> &SerializedVector {
    let serialize_vector: &SerializedVector = bytemuck::cast_slice(vector)
        .try_into()
        .expect("Failed to try into &[u8; DIM*4]");
    serialize_vector
}

pub fn serialize_edges(edges: &Edges) -> (&SerializedEdges, [u8; 4]) {
    let serialize_edges_len = (edges.len() as u32).to_le_bytes();
    let serialize_edges: &SerializedEdges = bytemuck::cast_slice(edges)
        .try_into()
        .expect("Failed to try into &[u8; DIGREE*4]");
    (serialize_edges, serialize_edges_len)
}

pub fn deserialize_node(
    bytes: &[u8],
    vector_dim: usize,
    edge_max_digree: usize,
) -> (&Vector, &Edges) {
    let vector_end = vector_dim * 4;
    let edges_start = vector_end + 4;
    let edges_len = u32::from_le_bytes(bytes[vector_end..edges_start].try_into().unwrap()) as usize;
    let edges_end = edges_start + std::cmp::min(edge_max_digree, edges_len) * 4;

    let vector: &Vector = deserialize_vector(&bytes[..vector_end]);
    let edges: &Edges = deserialize_edges(&bytes[edges_start..edges_end]);

    (vector, edges)
}

pub fn deserialize_vector(serialize_vector: &SerializedVector) -> &Vector {
    let vector: &Vector = bytemuck::try_cast_slice(serialize_vector)
        .expect("Failed to deserialize embeddings")
        .try_into()
        .expect("Failed to try into &[f32; DIM]");

    vector
}

pub fn deserialize_edges(serialize_edges: &SerializedEdges) -> &Edges {
    let edges: &Edges = bytemuck::try_cast_slice(serialize_edges)
        .expect("Failed to deserialize embeddings")
        .try_into()
        .expect("Failed to try into &[f32; DIM]");

    edges
}

pub fn node_byte_size(vector_dim: usize) -> usize {
    let node_byte_size = vector_dim * 4 + 140 * 4 + 4;
    node_byte_size
}