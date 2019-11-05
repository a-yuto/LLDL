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

    pub fn identity(x:&Vec<f64>) -> Vec<f64> {
        let out = x.clone();
        out
    }
}


struct ThreeLayerNn;
impl ThreeLayerNn {
    pub fn NnCal(input: &Vec<Vec<f64>>,w: &Vec<Vec<f64>>,b: &Vec<Vec<f64>>,h: fn(&Vec<f64>) -> Vec<f64> ) -> Vec<Vec<f64>>{
        let a = MatOpp::mul(&input,&w);
        let out: Vec<Vec<f64>> = MatOpp::add(&a,&b).unwrap();
        vec![h(&out[0])]
    }
    
    pub fn L1FromInput(input: &Vec<Vec<f64>>,w: &Vec<Vec<f64>>,b: &Vec<Vec<f64>>,h: fn(&Vec<f64>) -> Vec<f64> ) -> Vec<Vec<f64>>{
        ThreeLayerNn::NnCal(input,w,b,h)
    }

    pub fn L2FromL1(input: &Vec<Vec<f64>>,w: &Vec<Vec<f64>>,b: &Vec<Vec<f64>>,h: fn(&Vec<f64>) -> Vec<f64> ) -> Vec<Vec<f64>>{
        ThreeLayerNn::NnCal(input,w,b,h)
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
    assert_nearly_eq!(_c,_d);

    let _e = vec![ 0.0, 1.0, 2.0];
    assert_eq!(_e,AcFn::ReLU(&_a));
}

#[test]
pub fn NN_works() {
    let x1 = vec![vec![1.0,0.5]];
    let w1 = vec![vec![0.1,0.3,0.5],
                  vec![0.2,0.4,0.6]
    ];
    let b1 = vec![vec![0.1,0.2,0.3]];
    let z1 = vec![vec![0.574442516811659, 0.6681877721681662, 0.7502601055951177]];
    assert_nearly_eq!(z1,ThreeLayerNn::NnCal(&x1,&w1,&b1,AcFn::sigmoid));
    
    let x2 = ThreeLayerNn::L1FromInput(&x1,&w1,&b1,AcFn::sigmoid);
    assert_nearly_eq!(z1,x2);
    let w2 = vec![vec![ 0.2, 0.4],
                  vec![ 0.2, 0.5],
                  vec![ 0.3, 0.6]
    ];
    let b2 = vec![vec![ 0.1, 0.2]];
    let x3 = ThreeLayerNn::L1FromInput(&x2,&w2,&b2,AcFn::sigmoid);

    let w3 = vec![vec![ 0.1, 0.3],
                  vec![ 0.2, 0.4]
    ];
    let b3 = vec![vec![ 0.1, 0.2]];
    let y  = ThreeLayerNn::L1FromInput(&x3,&w3,&b3,AcFn::identity); 
    let z3 = vec![vec![0.3181615777097998, 0.7002825937582768]];
    assert_nearly_eq!(z3,y);
}
