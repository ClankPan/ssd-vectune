use crate::graph::*;

use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use std::time::SystemTime;

fn gen_random_vector(rng: &mut SmallRng) -> Vec<f32> {
    (0..384).map(|_| rng.gen::<f32>()).collect()
}

fn gen_random_edges(rng: &mut SmallRng) -> Vec<Id> {
    (0..70).map(|_| rng.gen::<Id>()).collect()
}

#[test]
fn serializing() {
    let mut rng = SmallRng::seed_from_u64(123456789);
    let vector: Vector = gen_random_vector(&mut rng)
        .try_into()
        .expect("cannot convert to [f32; DIM]");
    let edges: Edges = gen_random_edges(&mut rng)
        .try_into()
        .expect("cannot convert to [Id; DIGREE");
    let (seri_vector, seri_edges) = serialize_node(&vector, &edges);
    let (deseri_vector, deseri_edges) = deserialize_node(seri_vector, seri_edges);
    assert_eq!(&vector, deseri_vector);
    assert_eq!(&edges, deseri_edges);
}

struct TestStorage {
    memory: Vec<u8>,
}
impl TestStorage {
    pub fn new(num_sector: usize, rng: &mut SmallRng) -> Self {
        let vector: Vector = gen_random_vector(rng)
            .try_into()
            .expect("cannot convert to [f32; DIM]");
        let edges: Edges = gen_random_edges(rng)
            .try_into()
            .expect("cannot convert to [Id; DIGREE");
        let (seri_vector, seri_edges): (&SerializedVector, &SerializedEdges) =
            serialize_node(&vector, &edges);
        let mut memory = vec![0; SECTOR_SIZE * num_sector];

        let store_index = 10;
        let (sector_index, offset_in_sector) = store_index_to_sector_index_and_offset(&store_index);

        let sector_offset = sector_index * SECTOR_SIZE;
        let offset = sector_offset + offset_in_sector;
        memory[offset..offset + VECTOR_SIZE].copy_from_slice(seri_vector);
        memory[offset + VECTOR_SIZE..offset + VECTOR_SIZE + EDGES_SIZE].copy_from_slice(seri_edges);

        Self { memory }
    }
}
impl Storage for TestStorage {
    fn read(&self, offset: usize, dst: &mut [u8]) {
        let end = offset + dst.len();
        dst.copy_from_slice(&self.memory[offset..end]);
    }

    fn write(&mut self, offset: usize, src: &[u8]) {
        let end = offset + src.len();
        self.memory[offset..end].copy_from_slice(src);
    }
}

#[test]
fn reading_from_cache() {
    let num_sector = 100;
    let mut rng = SmallRng::seed_from_u64(123456789);
    let mut cache = Cache {
        cache_table: vec![(usize::MAX, SystemTime::now()); num_sector],
        cache: vec![[0; SECTOR_SIZE]; num_sector],
        storage: TestStorage::new(num_sector, &mut rng),
    };

    let a = cache.read_node(&10);

    println!("{:?}", a);
}

#[test]
fn rewrite_edges() {
    let num_sector = 100;
    let mut rng = SmallRng::seed_from_u64(123456789);
    let mut cache = Cache {
        cache_table: vec![(usize::MAX, SystemTime::now()); num_sector],
        cache: vec![[0; SECTOR_SIZE]; num_sector],
        storage: TestStorage::new(num_sector, &mut rng),
    };

    let (_, original_edges) = cache.read_node(&10);
    let original_edges: Edges = original_edges.to_vec()[..].try_into().unwrap();
    println!("{:?}", original_edges);

    cache.write_edges(&10, &original_edges);

    let (_, rewrited_edges) = cache.read_node(&10);

    assert_eq!(rewrited_edges, &original_edges);

    let random_edges: Edges = gen_random_edges(&mut rng)[..].try_into().unwrap();
    cache.write_edges(&10, &random_edges);
    let (_, random_rewrited_edges) = cache.read_node(&10);

    println!("{:?}", random_rewrited_edges);
    assert_eq!(random_rewrited_edges, &random_edges);
}
