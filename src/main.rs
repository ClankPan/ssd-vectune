#[cfg(test)]
mod tests;

pub mod graph;
pub mod point;
pub mod original_vector_reader;
pub mod k_means;

use anyhow::Result;
use bit_set::BitSet;
use point::Point;
use rayon::iter::IndexedParallelIterator;
use rayon::iter::IntoParallelIterator;
use rayon::iter::IntoParallelRefIterator;
use rayon::iter::ParallelIterator;
use original_vector_reader::OriginalVectorReader;
use k_means::on_disk_k_means;

type VectorIndex = usize;

fn main() -> Result<()> {

    let seed = 01234;

    /* k-meansをSSD上で行う */
    // 1. memmap
    // 2. 1つの要素ずつアクセスする関数を定義
    let path = "deep1M.fbin";
    let vector_reader = OriginalVectorReader::new(path)?;

    // 3. k個のindexをランダムで決めて、Vec<(ClusterPoint, PointSum, NumInCluster)>
    let num_clusters: u8 = 16;
    let cluster_labels = on_disk_k_means(&vector_reader,  &num_clusters, &seed);

    /* sharding */
    // let mut node_written_ssd_bitmap = BitVec::from_elem(vector_reader.num_vectors, false);
    let mut node_written_ssd_bitmap = BitSet::new();

    for cluster_label in 0..(num_clusters as u8) {
        // 1. a cluster labelを持つpointをlist upして、vectune::indexに渡す。
        let table_for_shard_id_to_node_id: Vec<VectorIndex> = cluster_labels
            .iter()
            .enumerate()
            .filter(|(_, (first, second))| *first == cluster_label || *second == cluster_label)
            .map(|(id, _)| id)
            .collect();
        let shard: Vec<Point> = table_for_shard_id_to_node_id
            .par_iter()
            .map(|node_id| Point::from_f32_vec(vector_reader.read(node_id).unwrap()))
            .collect();
        // 2. vectune::indexに渡すノードのindexとidとのtableを作る
        let (indexed_shard, start_shard_id): (Vec<(Point, Vec<u32>)>, u32) =
            vectune::Builder::default().set_seed(seed).build(shard);
        let _start_node_id = table_for_shard_id_to_node_id[start_shard_id as usize];

        // 3. idをもとにssdに書き込む。
        // 4. bitmapを持っておいて、idがtrueの時には、すでにあるedgesをdeserializeして、extend, dup。
        indexed_shard.into_par_iter().enumerate().for_each(
            |(shard_id, (_point, shard_id_edges))| {
                let node_id = table_for_shard_id_to_node_id[shard_id as usize];
                let _edges: Vec<VectorIndex> = shard_id_edges
                    .into_iter()
                    .map(|edge_shard_id| table_for_shard_id_to_node_id[edge_shard_id as usize])
                    .collect();

                if node_written_ssd_bitmap.contains(node_id) {
                    // Pointは書き込まず、元のedgesだけ復号して、追加する
                } else {
                    // Pointとedgesをserializeしてssdに書き込む
                }
            },
        );
        table_for_shard_id_to_node_id.iter().for_each(|node_id| {
            node_written_ssd_bitmap.insert(*node_id);
        });
    }

    /* gordering */
    // 1. node-idでssdから取り出すメソッドを定義する
    // 2. 並び替えの順番をもとに、ssdに書き込む。
    // wip : gordering用のshuffleを消して、別のロジックに書き換える。

    Ok(())
}
