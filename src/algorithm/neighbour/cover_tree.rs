use crate::math;
use crate::algorithm::neighbour::KNNAlgorithm;
use std::collections::HashMap;
use std::fmt::Debug;

pub struct CoverTree<'a, T> 
where T: Debug
{

    base: f64,
    max_level: i8,
    min_level: i8,
    distance: &'a Fn(&T, &T) -> f64,
    nodes: Vec<Node<T>>
}

impl<'a, T> CoverTree<'a, T> 
where T: Debug
{

    pub fn new(data: Vec<T>, distance: &'a Fn(&T, &T) -> f64) -> CoverTree<T> {
        let mut tree = CoverTree {
            base: 2f64,
            max_level: 10,
            min_level: 10,
            distance: distance,
            nodes: Vec::new()
        };

        for p in data {
            tree.insert(p);
        }

        tree
        
    }

    pub fn new_node(&mut self, data: T) -> NodeId {
        let next_index = self.nodes.len();
        let node_id = NodeId { index: next_index };
        self.nodes.push(
            Node {
                index: node_id,
                data: data,
                parent: None,
                children: HashMap::new()                               
            });        
        node_id
    }    

    fn insert(&mut self, p: T) {
        if self.nodes.is_empty(){
            self.new_node(p);
        } else {            
            let mut parent: Option<NodeId> = Option::None;
            let mut p_i = 0;
            let mut qi_p_ds = vec!((self.root(), (self.distance)(&p, &self.root().data)));
            let mut i = self.max_level;
            loop {                
                let i_d = self.base.powf(i as f64);
                let q_p_ds = self.get_children_dist(&p, &qi_p_ds, i);
                let d_p_Q = self.min_ds(&q_p_ds);                
                if d_p_Q < math::small_e {
                    return
                } else if d_p_Q > i_d {
                    break;
                }                      
                if self.min_ds(&qi_p_ds) <= self.base.powf(i as f64){
                    parent = q_p_ds.iter().find(|(_, d)| d <= &i_d).map(|(n, d)| n.index);
                    p_i = i;
                }
                
                qi_p_ds = q_p_ds.into_iter().filter(|(n, d)| d <= &i_d).collect();
                i -= 1;                
            }

            let new_node = self.new_node(p);            
            self.nodes.get_mut(parent.unwrap().index).unwrap().children.insert(p_i, new_node);
            self.min_level = i8::min(self.min_level, p_i-1);
        }
    }

    fn root(&self) -> &Node<T> {
        self.nodes.first().unwrap()
    }

    fn get_children_dist<'b>(&'b self, p: &T, qi_p_ds: &Vec<(&'b Node<T>, f64)>, i: i8) -> Vec<(&'b Node<T>, f64)> {

        let mut children = Vec::<(&'b Node<T>, f64)>::new();

        children.extend(qi_p_ds.iter().cloned());

        let q: Vec<&Node<T>> = qi_p_ds.iter().flat_map(|(n, _)| self.get_child(n, i)).collect();        

        children.extend(q.into_iter().map(|n| (n, (self.distance)(&n.data, &p))));

        children
        
    }

    fn min_ds(&self, q_p_ds: &Vec<(&Node<T>, f64)>) -> f64 {
        q_p_ds.into_iter().min_by(|(_, d1), (_, d2)| d1.partial_cmp(d2).unwrap()).unwrap().1        
    }

    fn min_p_ds(&self, q_p_ds: &mut Vec<(&Node<T>, f64)>, k: usize) -> f64 {
        q_p_ds.sort_by(|(_, d1), (_, d2)| d1.partial_cmp(d2).unwrap());
        q_p_ds[..usize::min(q_p_ds.len(), k)].last().unwrap().1
    }

    fn get_child(&self, node: &Node<T>, i: i8) -> Option<&Node<T>> {
        node.children.get(&i).and_then(|n_id| self.nodes.get(n_id.index))       
    }

}

impl<'a, T> KNNAlgorithm<T> for CoverTree<'a, T>
where T: Debug
{
    fn find(&self, p: &T, k: usize) -> Vec<usize>{
        let mut qi_p_ds = vec!((self.root(), (self.distance)(&p, &self.root().data)));
        for i in (self.min_level..self.max_level+1).rev() {
            let i_d = self.base.powf(i as f64);
            let mut q_p_ds = self.get_children_dist(&p, &qi_p_ds, i);
            let d_p_q = self.min_p_ds(&mut q_p_ds, k);            
            qi_p_ds = q_p_ds.into_iter().filter(|(n, d)| d <= &(d_p_q + i_d)).collect();
        }        
        qi_p_ds.sort_by(|(_, d1), (_, d2)| d1.partial_cmp(d2).unwrap());
        qi_p_ds[..usize::min(qi_p_ds.len(), k)].iter().map(|(n, _)| n.index.index).collect()
    }
}

#[derive(Debug, Clone, Copy)]
pub struct NodeId {
    index: usize,
}

#[derive(Debug)]
struct Node<T> {
    index: NodeId,
    data: T,
    children: HashMap<i8, NodeId>,
    parent: Option<NodeId>
}

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn cover_tree_test() {  
        let data = vec!(1, 2, 3, 4, 5, 6, 7, 8, 9);          
        let distance = |a: &i32, b: &i32| -> f64 {
            (a - b).abs() as f64
        };            
        let tree = CoverTree::<i32>::new(data, &distance);
        let nearest_3 = tree.find(&5, 3);
        assert_eq!(vec!(4, 5, 3), nearest_3);
    }

}