extern crate vec;
use vec::mat_opp::*;
use vec::vec_opp::*;
/*
 * activity function
 */
#[warn(non_snake_case)]
pub fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}
pub fn step(x: f64) -> f64 {
    match x > 0.0 {
        true  => 1.0,
        false => 0.0
    }
}
fn max(x: f64,y: f64) -> f64 {
    match x > y {
        true  => x,
        false => y
    }
}
pub fn ReLU(x: f64) -> f64 {
    max(0.0,x)
}


pub fn softmax(x : &vec::mat_opp::Matrix) -> Matrix {
    let mut _sum = 0.0;
    let tmp_matrix = x.mat.clone();
    for vector in tmp_matrix {
        for val in vector {
            _sum += val.exp();
        }
    }
    
    let _ans = 
    (0..x.col).map(|i|{
        (0..x.row).map(|j|
            x.mat[i][j].exp() / _sum
        ).collect()
    }).collect();
    Matrix {
        mat: _ans,
        row: x.row,
        col: x.col
    }
}
#[cfg(test)]
mod activity_function_tests {
    use super::*;

    #[test]
    fn sigmoid_work(){
        assert_eq!(sigmoid(-1.0),0.2689414213699951);
    }
    #[test]
    fn step_works(){
         assert_eq!(step(0.0),0.0);
         assert_eq!(step(5.0),1.0);
         assert_eq!(step(-100.0),0.0);
    }
    #[test]
    fn ReLU_work(){
         assert_eq!(ReLU(10.0),10.0);
         assert_eq!(ReLU(0.0),0.0);
         assert_eq!(ReLU(-100.0),0.0);
    }
    #[test]
    fn softmax_work() {
        let test = Matrix {
            mat: vec![vec![0.3,2.9,4.0]],
            row: 3,
            col: 1 
        };
        let ans = Matrix {
            mat: vec![vec![0.018211273295547534, 0.2451918129350739, 0.7365969137693785]],
            row: 3,
            col: 1
        };
        matrix_test(&softmax(&test),&ans);
    }
}

