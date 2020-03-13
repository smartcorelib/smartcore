pub mod distance;

pub static EPSILON:f64 = 2.2204460492503131e-16_f64;

pub trait NumericExt {
    fn ln_1pe(&self) -> f64;
    fn sigmoid(&self) -> f64;
}

impl NumericExt for f64 {

    fn ln_1pe(&self) -> f64{        
        if *self > 15. {
            return *self;
        } else {
            return self.exp().ln_1p();
        }

    }

    fn sigmoid(&self) -> f64 {
        
        if *self < -40. {
            return 0.;
        } else if *self > 40. {
            return 1.;
        } else {
            return 1. / (1. + f64::exp(-self))
        }
        
    }
}

#[cfg(test)]
mod tests {    
    use super::*;     

    #[test]
    fn sigmoid() { 
        assert_eq!(1.0.sigmoid(), 0.7310585786300049);
        assert_eq!(41.0.sigmoid(), 1.);
        assert_eq!((-41.0).sigmoid(), 0.);
    }
}