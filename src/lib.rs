extern crate vec;
use vec::VecOpp;
use vec::MatOpp;
use nearly_eq::*;
use std::cmp;

struct Percept;
impl Percept{
    pub fn and(x:&Vec<f64>) -> f64 {
        let b = -0.7;
        let _w = vec![0.5,0.5];
        let tmp   = VecOpp::in_mul(&x,&_w).unwrap() + b;
        let out   = if tmp <= 0.0 {
            0.0
        } else {
            1.0
        };
        out
    }

    pub fn nand(x:&Vec<f64>) -> f64 {
        let b = 0.7;
        let _w = vec![-0.5,-0.5];
        let tmp   = VecOpp::in_mul(&x,&_w).unwrap() + b;
        let out   = if tmp <= 0.0 {
            0.0
        } else {
            1.0
        };
        out
    }

    pub fn or(x:&Vec<f64>) -> f64 {
        let b = -0.2;
        let _w = vec![0.5,0.5];
        let tmp   = VecOpp::in_mul(&x,&_w).unwrap() + b;
        let out   = if tmp <= 0.0 {
            0.0
        } else {
            1.0
        };
        out
    }

    pub fn xor (x:&Vec<f64>) -> f64 {
        let out = Percept::and(&vec![Percept::nand(&x),Percept::or(&x)]);
        out
    }
}

struct AcFn;
impl AcFn {
    pub fn step_function(x:&Vec<f64>) -> Vec<f64> {
        let mut out: Vec<f64> = Vec::new();
        for i in x {
            match (0.0 as f64) < *i {
                true  => out.push(1.0),
                false => out.push(0.0),
            }
        }
        out
    }

    pub fn sigmoid(x:&Vec<f64>) -> Vec<f64> {
        let mut out: Vec<f64> = Vec::new();
        for i in x {
            let tmp: f64 = 1.0 / (1.0 + (-i).exp());
            out.push(tmp);
        }
        out
    }

    pub fn ReLU(x:&Vec<f64>) -> Vec<f64> {
        let mut out: Vec<f64> = Vec::new();
        for i in x {
            out.push(i.max(0.0));
        }
        out
    }
}


struct TheeLayerNn;
impl TheeLayerNn {
    pub fn NnCal(x: &Vec<Vec<f64>>,w: &Vec<Vec<f64>>,b: &Vec<Vec<f64>>,h: fn(&Vec<f64>) -> Vec<f64> ) -> Vec<Vec<f64>>{
        let z = MatOpp::mul(&x,&w);
        let a: Vec<Vec<f64>> = MatOpp::add(&z,&b).unwrap();
        vec![h(&a[0])]
    }

}
//-------------ここからテストです------------------
#[test]
pub fn perceptron_works(){
    let x1 = vec![0.0,0.0];
    let x2 = vec![0.0,1.0];
    let x3 = vec![1.0,0.0];
    let x4 = vec![1.0,1.0];
    

    assert_eq!(Percept::and(&x1),0.0);
    assert_eq!(Percept::and(&x2),0.0);
    assert_eq!(Percept::and(&x3),0.0);
    assert_eq!(Percept::and(&x4),1.0);
    
    assert_eq!(Percept::nand(&x1),1.0);
    assert_eq!(Percept::nand(&x2),1.0);
    assert_eq!(Percept::nand(&x3),1.0);
    assert_eq!(Percept::nand(&x4),0.0);

    assert_eq!(Percept::or(&x1),0.0);
    assert_eq!(Percept::or(&x2),1.0);
    assert_eq!(Percept::or(&x3),1.0);
    assert_eq!(Percept::or(&x4),1.0);

    assert_eq!(Percept::xor(&x1),0.0);
    assert_eq!(Percept::xor(&x2),1.0);
    assert_eq!(Percept::xor(&x3),1.0);
    assert_eq!(Percept::xor(&x4),0.0);
}

#[test]
pub fn AcFn_works() {
    let _a = vec![ -1.0, 1.0, 2.0];
    let _b = vec![  0.0, 1.0, 1.0];
    assert_eq!(AcFn::step_function(&_a),_b);
    
    let _c = vec![0.2689414213699951, 0.7310585786300049, 0.8807970779778823];
    let _d = AcFn::sigmoid(&_a);
    for i in 0.._c.len() {
        assert_nearly_eq!(_c[i],_d[i]);
    }

    let _e = vec![ 0.0, 1.0, 2.0];
    assert_eq!(_e,AcFn::ReLU(&_a));
}

#[test]
pub fn NN_works() {
    let _x = vec![vec![1.0,0.5]];
    let w1 = vec![vec![0.1,0.3,0.5],
                  vec![0.2,0.4,0.6]
    ];
    let b1 = vec![vec![0.1,0.2,0.3]];
    let a1 = vec![vec![0.574442516811659, 0.6681877721681662, 0.7502601055951177]];
    assert_nearly_eq!(a1,TheeLayerNn::NnCal(&_x,&w1,&b1,AcFn::sigmoid));
}
