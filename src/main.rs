#[cfg(test)]
mod tests;

pub mod graph;
pub mod k_means;
pub mod node_reader;
pub mod original_vector_reader;
pub mod point;
pub mod sharded_index;

use anyhow::Result;
use k_means::on_disk_k_means;
use node_reader::GraphOnStorage;
use original_vector_reader::{OriginalVectorReader, OriginalVectorReaderTrait};
use sharded_index::sharded_index;

type VectorIndex = usize;

fn main() -> Result<()> {
    let seed = 01234;

    /* k-meansをSSD上で行う */
    // 1. memmap
    // 2. 1つの要素ずつアクセスする関数を定義
    let path = "test_vectors/base.10M.fbin";
    let vector_reader = OriginalVectorReader::new(path)?;

    // 3. k個のindexをランダムで決めて、Vec<(ClusterPoint, PointSum, NumInCluster)>
    let num_clusters: u8 = 16;
    let cluster_labels = on_disk_k_means(&vector_reader, &num_clusters, &seed);

    /* sharding */
    let mut graph_on_stroage = GraphOnStorage::new_with_empty_file(
        "",
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

    Ok(())
}
