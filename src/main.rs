#![feature(portable_simd)]



#[cfg(test)]
mod tests;

pub mod graph;
pub mod k_means;
pub mod node_reader;
pub mod original_vector_reader;
pub mod point;
pub mod sharded_index;

use std::{io, path};

use anyhow::Result;
use bit_vec::BitVec;
use bytesize::MB;
use ext_sort::{buffer::LimitedBufferBuilder, ExternalSorter, ExternalSorterBuilder};
use itertools::Itertools;
use k_means::on_disk_k_means;
use node_reader::{EdgesIterator, GraphOnStorage, GraphOnStorageTrait};
use original_vector_reader::{OriginalVectorReader, OriginalVectorReaderTrait};
use rand::{rngs::SmallRng, SeedableRng};
use sharded_index::sharded_index;

type VectorIndex = usize;

fn main() -> Result<()> {
    let seed = 01234;
    let mut rng = SmallRng::seed_from_u64(seed);

    /* k-meansをSSD上で行う */
    // 1. memmap
    // 2. 1つの要素ずつアクセスする関数を定義
    println!("reading vector file");
    let path = "test_vectors/base.10M.fbin";
    let vector_reader = OriginalVectorReader::new(path)?;

    // 3. k個のindexをランダムで決めて、Vec<(ClusterPoint, PointSum, NumInCluster)>
    println!("k-menas on disk");
    let num_clusters: u8 = 100;
    let cluster_labels = on_disk_k_means(&vector_reader, &num_clusters, &mut rng);

    /* sharding */
    println!("sharded indexing");
    let mut graph_on_stroage = GraphOnStorage::new_with_empty_file(
        "test_vectors/unordered_graph.10M.graph",
        vector_reader.get_num_vectors() as u32,
        vector_reader.get_vector_dim() as u32,
        70 * 2,
    )?;
    sharded_index(
        &vector_reader,
        &mut graph_on_stroage,
        &num_clusters,
        &cluster_labels,
        seed,
    );

    /* gordering */
    // 1. node-idでssdから取り出すメソッドを定義する
    // 2. 並び替えの順番をもとに、ssdに書き込む。
    // wip : gordering用のshuffleを消して、別のロジックに書き換える。
    // backlinksを取り出す。
    println!("gordering");
    let edge_iter = EdgesIterator::new(&graph_on_stroage);
    let sorter: ExternalSorter<(u32, u32), std::io::Error, LimitedBufferBuilder> =
        ExternalSorterBuilder::new()
            .with_tmp_dir(path::Path::new("./"))
            // .with_buffer(LimitedBufferBuilder::new((50 * MB) as usize, false))
            .build()
            .unwrap();
    let sorted = sorter.sort_by(edge_iter, |a, b| a.0.cmp(&b.0)).unwrap();
    let backlinks: Vec<Vec<u32>> = sorted
        .into_iter()
        .map(Result::unwrap)
        .chunk_by(|&(key, _)| key)
        .into_iter()
        .map(|(_key, group)| group.map(|(_, edge)| edge).collect())
        .collect();

    let get_backlinks = |id: &u32| -> Vec<u32> { backlinks[*id as usize].clone() };

    let get_edges =
        |id: &u32| -> Vec<u32> { graph_on_stroage.read_edges(&(*id as usize)).unwrap() };

    let target_node_bit_vec = BitVec::from_elem(vector_reader.get_num_vectors(), true);
    let window_size = 10;
    let reordered_node_ids = vectune::gorder(
        get_edges,
        get_backlinks,
        target_node_bit_vec,
        window_size,
        &mut rng,
    );

    Ok(())
}

/*
Note:
storageに書き込まれているedgesは、確保されている要素数に満たない場合があり、それはどうdeserializeされるのか。

*/
