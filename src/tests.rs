use core::num;

use super::*;

use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

fn gen_random_point(rng: &mut SmallRng) -> Vec<f32> {
    (0..384).map(|_| rng.gen::<f32>()).collect()
}

fn gen_random_edges(rng: &mut SmallRng) -> Vec<Id> {
    (0..70).map(|_| rng.gen::<Id>()).collect()
}

#[test]
fn serializing() {
    let mut rng = SmallRng::seed_from_u64(123456789);
    let point: Point = gen_random_point(&mut rng)
        .try_into()
        .expect("cannot convert to [f32; DIM]");
    let edges: Edges = gen_random_edges(&mut rng)
        .try_into()
        .expect("cannot convert to [Id; DIGREE");
    let (seri_point, seri_edges) = serialize_node(&point, &edges);
    let (deseri_point, deseri_edges) = deserialize_node(seri_point, seri_edges);
    assert_eq!(&point, deseri_point);
    assert_eq!(&edges, deseri_edges);
}

#[test]
fn reading_from_cache() {
  struct TestStorage {
    memory: Vec<u8>
  }
  impl TestStorage {
    pub fn new(num_sector: usize) -> Self {

      let mut rng = SmallRng::seed_from_u64(123456789);
      let point: Point = gen_random_point(&mut rng)
          .try_into()
          .expect("cannot convert to [f32; DIM]");
      let edges: Edges = gen_random_edges(&mut rng)
          .try_into()
          .expect("cannot convert to [Id; DIGREE");
      let (seri_point, seri_edges): (&SerializedPoint, &SerializedEdges) = serialize_node(&point, &edges);
      let mut memory = vec![0; SECTOR_SIZE*num_sector];

      let store_index = 10;
      let (sector_index, offset_in_sector) = store_index_to_sector_index_and_offset(&store_index);

      let sector_offset = sector_index * SECTOR_SIZE;
      let offset = sector_offset + offset_in_sector;
      memory[offset..offset+POINT_SIZE].copy_from_slice(seri_point);
      memory[offset+POINT_SIZE..offset+POINT_SIZE+EDGES_SIZE].copy_from_slice(seri_edges);

      Self {
        memory,
      }
    }
  }
  impl Storage for TestStorage {
    fn read(&self, offset: u64, dst: &mut [u8]) {
      let offset = offset as usize;
      let end = offset + dst.len();
      dst.copy_from_slice(&self.memory[offset..end]);
    }
  }

  let num_sector = 100;
  let mut cache = Cache {
    cache_table: vec![(usize::MAX, SystemTime::now()); num_sector],
    cache: vec![[0; SECTOR_SIZE]; num_sector],
    storage: TestStorage::new(num_sector),
  };

  let a = cache.read(&10);

  println!("{:?}", a);

}