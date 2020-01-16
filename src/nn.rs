extern crate vec;
use nearly_eq::*;
/*
 * perceptron
 */

pub fn and(_x1: f64,_x2: f64) -> f64 {
    let (w1,w2,bias) = (0.5,0.5,-0.7);
    let tmp          = _x1 * w1 + _x2 * w2 + bias;
    match tmp >= 0.0 {
        true  => 1.0,
        false => 0.0
    }
}

pub fn nand(_x1: f64,_x2: f64) -> f64 {
    let (w1,w2,bias) = (-0.5,-0.5,0.7);
    let tmp          = _x1 * w1 + _x2 * w2 + bias;
    match tmp >= 0.0 {
        true  => 1.0,
        false => 0.0
    }
}

pub fn or(_x1: f64,_x2: f64) -> f64 {
    let (w1,w2,bias) = (0.5,0.5,-0.2);
    let tmp          = _x1 * w1 + _x2 * w2 + bias;
    match tmp >= 0.0 {
        true  => 1.0,
        false => 0.0
    }
}

pub fn xor(_x1: f64,_x2: f64) -> f64 {
    let s1 = nand(_x1,_x2);
    let s2 = or(_x1,_x2);
    and(s1,s2)
}
#[cfg(test)]
mod perceptoron_test {
    use super::*;

    #[test]
    fn and_test(){
        assert_eq!(and(0.0,0.0) ,0.0);
        assert_eq!(and(1.0,0.0) ,0.0);
        assert_eq!(and(0.0,1.0) ,0.0);
        assert_eq!(and(1.0,1.0), 1.0);
    }
    #[test]
    fn nand_test(){
        assert_eq!(nand(0.0,0.0) ,1.0);
        assert_eq!(nand(1.0,0.0) ,1.0);
        assert_eq!(nand(0.0,1.0) ,1.0);
        assert_eq!(nand(1.0,1.0), 0.0);
    }
    #[test]
    fn or_test(){
        assert_eq!(or(0.0,0.0) ,0.0);
        assert_eq!(or(1.0,0.0) ,1.0);
        assert_eq!(or(0.0,1.0) ,1.0);
        assert_eq!(or(1.0,1.0), 1.0);
    }
    #[test]
    fn xor_test(){
        assert_eq!(xor(0.0,0.0) ,0.0);
        assert_eq!(xor(1.0,0.0) ,1.0);
        assert_eq!(xor(0.0,1.0) ,1.0);
        assert_eq!(xor(1.0,1.0), 0.0);
    }


}
