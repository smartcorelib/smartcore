extern crate num;
use std::ops::Range;
use std::fmt;
use num::complex::Complex;
use crate::linalg::{Matrix};
use crate::linalg::svd::SVD;
use crate::linalg::evd::EVD;
use crate::math;
use rand::prelude::*;

#[derive(Debug, Clone)]
pub struct DenseMatrix {

    ncols: usize,
    nrows: usize,
    values: Vec<f64> 

}

impl fmt::Display for DenseMatrix {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut rows: Vec<Vec<f64>> = Vec::new();
        for r in 0..self.nrows {
            rows.push(self.get_row_as_vec(r).iter().map(|x| (x * 1e4).round() / 1e4 ).collect());
        }        
        write!(f, "{:?}", rows)
    }
}

impl DenseMatrix {  
    
    fn new(nrows: usize, ncols: usize, values: Vec<f64>) -> DenseMatrix {
        DenseMatrix {
            ncols: ncols,
            nrows: nrows,
            values: values
        }
    } 

    pub fn from_array(values: &[&[f64]]) -> DenseMatrix {
        DenseMatrix::from_vec(&values.into_iter().map(|row| Vec::from(*row)).collect())
    }

    pub fn from_vec(values: &Vec<Vec<f64>>) -> DenseMatrix {
        let nrows = values.len();
        let ncols = values.first().unwrap_or_else(|| panic!("Cannot create 2d matrix from an empty vector")).len();
        let mut m = DenseMatrix {
            ncols: ncols,
            nrows: nrows,
            values: vec![0f64; ncols*nrows]
        };
        for row in 0..nrows {
            for col in 0..ncols {
                m.set(row, col, values[row][col]);
            }
        }
        m
    }      

    pub fn vector_from_array(values: &[f64]) -> DenseMatrix {
        DenseMatrix::vector_from_vec(Vec::from(values)) 
     }

    pub fn vector_from_vec(values: Vec<f64>) -> DenseMatrix {
        DenseMatrix {
            ncols: values.len(),
            nrows: 1,
            values: values
        }
    }

    pub fn div_mut(&mut self, b: DenseMatrix) -> () {
        if self.nrows != b.nrows || self.ncols != b.ncols {
            panic!("Can't divide matrices of different sizes.");
        }

        for i in 0..self.values.len() {
            self.values[i] /= b.values[i];
        }
    }    

    pub fn get_raw_values(&self) -> &Vec<f64> {
        &self.values
    }      

    fn tred2(&mut self, d: &mut Vec<f64>, e: &mut Vec<f64>) {
        
        let n = self.nrows;
        for i in 0..n {
            d[i] = self.get(n - 1, i);
        }

        // Householder reduction to tridiagonal form.
        for i in (1..n).rev() {
            // Scale to avoid under/overflow.
            let mut scale = 0f64;
            let mut h = 0f64;
            for k in 0..i {
                scale = scale + d[k].abs();
            }
            if scale == 0f64 {
                e[i] = d[i - 1];
                for j in 0..i {
                    d[j] = self.get(i - 1, j);
                    self.set(i, j, 0.0);
                    self.set(j, i, 0.0);
                }
            } else {
                // Generate Householder vector.
                for k in 0..i {
                    d[k] /= scale;
                    h += d[k] * d[k];
                }
                let mut f = d[i - 1];
                let mut g = h.sqrt();
                if f > 0f64 {
                    g = -g;
                }
                e[i] = scale * g;
                h = h - f * g;
                d[i - 1] = f - g;
                for j in 0..i {
                    e[j] = 0f64;
                }

                // Apply similarity transformation to remaining columns.
                for j in 0..i {
                    f = d[j];
                    self.set(j, i, f);
                    g = e[j] + self.get(j, j) * f;
                    for k in j + 1..=i - 1 {
                        g += self.get(k, j) * d[k];
                        e[k] += self.get(k, j) * f;
                    }
                    e[j] = g;
                }
                f = 0.0;
                for j in 0..i {
                    e[j] /= h;
                    f += e[j] * d[j];
                }
                let hh = f / (h + h);
                for j in 0..i {
                    e[j] -= hh * d[j];
                }
                for j in 0..i {
                    f = d[j];
                    g = e[j];
                    for k in j..=i-1 {
                        self.sub_element_mut(k, j, f * e[k] + g * d[k]);
                    }
                    d[j] = self.get(i - 1, j);
                    self.set(i, j, 0.0);
                }
            }
            d[i] = h;
        }        

        // Accumulate transformations.
        for i in 0..n-1 {
            self.set(n - 1, i, self.get(i, i));
            self.set(i, i, 1.0);
            let h = d[i + 1];
            if h != 0f64 {
                for k in 0..=i {
                    d[k] = self.get(k, i + 1) / h;
                }
                for j in 0..=i {
                    let mut g = 0f64;
                    for k in 0..=i {
                        g += self.get(k, i + 1) * self.get(k, j);
                    }
                    for k in 0..=i {
                        self.sub_element_mut(k, j,  g * d[k]);
                    }
                }
            }
            for k in 0..=i {
                self.set(k, i + 1, 0.0);
            }
        }
        for j in 0..n {
            d[j] = self.get(n - 1, j);
            self.set(n - 1, j, 0.0);
        }
        self.set(n - 1, n - 1, 1.0);
        e[0] = 0.0;
    }

    fn tql2(&mut self, d: &mut Vec<f64>, e: &mut Vec<f64>) {
        let n = self.nrows;
        for i in 1..n {
            e[i - 1] = e[i];
        }
        e[n - 1] = 0f64;

        let mut f = 0f64;
        let mut tst1 = 0f64;
        for l in 0..n {
            // Find small subdiagonal element
            tst1 = f64::max(tst1, d[l].abs() + e[l].abs());

            let mut m = l;

            loop {
                if m < n {
                    if e[m].abs() <= tst1 * std::f64::EPSILON {
                        break;
                    }
                    m += 1;
                } else {                
                    break;
                }
            }

            // If m == l, d[l] is an eigenvalue,
            // otherwise, iterate.
            if m > l {
                let mut iter = 0;
                loop {
                    iter += 1;
                    if iter >= 30 {
                        panic!("Too many iterations");
                    }

                    // Compute implicit shift
                    let mut g = d[l];
                    let mut p = (d[l + 1] - g) / (2.0 * e[l]);
                    let mut r = p.hypot(1.0);
                    if p < 0f64 {
                        r = -r;
                    }
                    d[l] = e[l] / (p + r);
                    d[l + 1] = e[l] * (p + r);
                    let dl1 = d[l + 1];
                    let mut h = g - d[l];
                    for i in l+2..n {
                        d[i] -= h;
                    }
                    f = f + h;

                    // Implicit QL transformation.
                    p = d[m];
                    let mut c = 1.0;
                    let mut c2 = c;
                    let mut c3 = c;
                    let el1 = e[l + 1];
                    let mut s = 0.0;
                    let mut s2 = 0.0;
                    for i in (l..m).rev() {
                        c3 = c2;
                        c2 = c;
                        s2 = s;
                        g = c * e[i];
                        h = c * p;
                        r = p.hypot(e[i]);
                        e[i + 1] = s * r;
                        s = e[i] / r;
                        c = p / r;
                        p = c * d[i] - s * g;
                        d[i + 1] = h + s * (c * g + s * d[i]);

                        // Accumulate transformation.
                        for k in 0..n {
                            h = self.get(k, i + 1);
                            self.set(k, i + 1, s * self.get(k, i) + c * h);
                            self.set(k, i,     c * self.get(k, i) - s * h);
                        }
                    }
                    p = -s * s2 * c3 * el1 * e[l] / dl1;
                    e[l] = s * p;
                    d[l] = c * p;

                    // Check for convergence.
                    if e[l].abs() <= tst1 * std::f64::EPSILON {
                        break;
                    }
                }
            }
            d[l] = d[l] + f;
            e[l] = 0f64;
        }

        // Sort eigenvalues and corresponding vectors.
        for i in 0..n-1 {
            let mut k = i;
            let mut p = d[i];
            for j in i + 1..n {
                if d[j] > p {
                    k = j;
                    p = d[j];
                }
            }
            if k != i {
                d[k] = d[i];
                d[i] = p;
                for j in 0..n {
                    p = self.get(j, i);
                    self.set(j, i, self.get(j, k));
                    self.set(j, k, p);
                }
            }
        }
    }

    fn balance(A: &mut Self) -> Vec<f64> {
        let radix = 2f64;
        let sqrdx = radix * radix;

        let n = A.nrows;

        let mut scale = vec![1f64; n];        

        let mut done = false;
        while !done {
            done = true;
            for i in 0..n {
                let mut r = 0f64;
                let mut c = 0f64;
                for j in 0..n {
                    if j != i {
                        c += A.get(j, i).abs();
                        r += A.get(i, j).abs();
                    }
                }
                if c != 0f64 && r != 0f64 {
                    let mut g = r / radix;
                    let mut f = 1.0;
                    let s = c + r;
                    while c < g {
                        f *= radix;
                        c *= sqrdx;
                    }
                    g = r * radix;
                    while c > g {
                        f /= radix;
                        c /= sqrdx;
                    }
                    if (c + r) / f < 0.95 * s {
                        done = false;
                        g = 1.0 / f;
                        scale[i] *= f;
                        for j in 0..n {
                            A.mul_element_mut(i, j, g);
                        }
                        for j in 0..n {
                            A.mul_element_mut(j, i, f);
                        }
                    }
                }
            }
        }

        return scale;
    }

    fn elmhes(A: &mut Self) -> Vec<usize> {
        let n = A.nrows;
        let mut perm = vec![0; n];
    
        for m in 1..n-1 {
            let mut x = 0f64;
            let mut i = m;
            for j in m..n {
                if A.get(j, m - 1).abs() > x.abs() {
                    x = A.get(j, m - 1);
                    i = j;
                }
            }            
            perm[m] = i;
            if i != m {
                for j in (m-1)..n {
                    let swap = A.get(i, j);
                    A.set(i, j, A.get(m, j));
                    A.set(m, j, swap);
                }
                for j in 0..n {
                    let swap = A.get(j, i);
                    A.set(j, i, A.get(j, m));
                    A.set(j, m, swap);
                }
            }            
            if x != 0f64 {
                for i in (m + 1)..n {
                    let mut y = A.get(i, m - 1);
                    if y != 0f64 {
                        y /= x;
                        A.set(i, m - 1, y);
                        for j in m..n {
                            A.sub_element_mut(i, j, y * A.get(m, j));
                        }
                        for j in 0..n {
                            A.add_element_mut(j, m, y * A.get(j, i));
                        }
                    }
                }
            }            
        }
    
        return perm;
    }   
    
    fn eltran(A: &Self, V: &mut Self, perm: &Vec<usize>) {
        let n = A.nrows;
        for mp in (1..n - 1).rev() {
            for k in mp + 1..n {
                V.set(k, mp, A.get(k, mp - 1));
            }
            let i = perm[mp];
            if i != mp {
                for j in mp..n {
                    V.set(mp, j, V.get(i, j));
                    V.set(i, j, 0.0);
                }
                V.set(i, mp, 1.0);
            }
        }
    }

    fn hqr2(A: &mut Self, V: &mut Self, d: &mut Vec<f64>, e: &mut Vec<f64>) {
        let n = A.nrows;                                                           
        let mut z = 0f64;        
        let mut s = 0f64;
        let mut r = 0f64;
        let mut q = 0f64;
        let mut p = 0f64;
        let mut anorm = 0f64;
        
        for i in 0..n {            
            for j in i32::max(i as i32 - 1, 0)..n as i32 {
                anorm += A.get(i, j as usize).abs();                
            }
        }        

        let mut nn = n - 1;
        let mut t = 0.0;
        'outer: loop {            
            let mut its = 0;
            loop {
                let mut l = nn;
                while l > 0 {                    
                    s = A.get(l - 1, l - 1).abs() + A.get(l, l).abs();
                    if s == 0.0 {
                        s = anorm;
                    }
                    if A.get(l, l - 1).abs() <= std::f64::EPSILON * s {
                        A.set(l, l - 1, 0.0);
                        break;
                    } 
                    l -= 1;                   
                }
                let mut x = A.get(nn, nn);
                if l == nn {
                    d[nn] = x + t;
                    A.set(nn, nn, x + t);
                    if nn == 0 {
                        break 'outer;
                    } else {
                        nn -= 1;
                    }                                        
                } else {
                    let mut y = A.get(nn - 1, nn - 1);
                    let mut w = A.get(nn, nn - 1) * A.get(nn - 1, nn);
                    if l == nn - 1 {
                        p = 0.5 * (y - x);
                        q = p * p + w;
                        z = q.abs().sqrt();
                        x += t;
                        A.set(nn, nn, x );
                        A.set(nn - 1, nn - 1, y + t);
                        if q >= 0.0 {
                            z = p + z.copysign(p);
                            d[nn - 1] = x + z;
                            d[nn] = x + z;
                            if z != 0.0 {
                                d[nn] = x - w / z;
                            }
                            x = A.get(nn, nn - 1);
                            s = x.abs() + z.abs();
                            p = x / s;
                            q = z / s;
                            r = (p * p + q * q).sqrt();
                            p /= r;
                            q /= r;
                            for j in nn-1..n {
                                z = A.get(nn - 1, j);
                                A.set(nn - 1, j, q * z + p * A.get(nn, j));
                                A.set(nn, j, q * A.get(nn, j) - p * z);
                            }
                            for i in 0..=nn {
                                z = A.get(i, nn - 1);
                                A.set(i, nn - 1, q * z + p * A.get(i, nn));
                                A.set(i, nn, q * A.get(i, nn) - p * z);
                            }
                            for i in 0..n {
                                z = V.get(i, nn - 1);
                                V.set(i, nn - 1, q * z + p * V.get(i, nn));
                                V.set(i, nn, q * V.get(i, nn) - p * z);
                            }
                        } else {
                            d[nn] = x + p;
                            e[nn] = -z;
                            d[nn - 1] = d[nn];
                            e[nn - 1] = -e[nn];
                        }
                        
                        if nn <= 1 {
                            break 'outer;
                        } else {
                            nn -= 2;
                        }                        
                    } else {
                        if its == 30 {
                            panic!("Too many iterations in hqr");
                        }
                        if its == 10 || its == 20 {
                            t += x;
                            for i in 0..nn+1 {
                                A.sub_element_mut(i, i, x);
                            }
                            s = A.get(nn, nn - 1).abs() + A.get(nn - 1, nn - 2).abs();
                            y = 0.75 * s;
                            x = 0.75 * s;
                            w = -0.4375 * s * s;
                        }
                        its += 1;
                        let mut m = nn - 2;
                        while m >= l {
                            z = A.get(m, m);
                            r = x - z;
                            s = y - z;
                            p = (r * s - w) / A.get(m + 1, m) + A.get(m, m + 1);
                            q = A.get(m + 1, m + 1) - z - r - s;
                            r = A.get(m + 2, m + 1);
                            s = p.abs() + q.abs() + r.abs();
                            p /= s;
                            q /= s;
                            r /= s;
                            if m == l {
                                break;
                            }
                            let u = A.get(m, m - 1).abs() * (q.abs() + r.abs());
                            let v = p.abs() * (A.get(m - 1, m - 1).abs() + z.abs() + A.get(m + 1, m + 1).abs());
                            if u <= std::f64::EPSILON * v {
                                break;
                            }
                            m -= 1;
                        }
                        for i in m..nn-1 {
                            A.set(i + 2, i , 0.0);
                            if i != m {
                                A.set(i + 2, i - 1, 0.0);
                            }
                        }
                        for k in m..nn {
                            if k != m {
                                p = A.get(k, k - 1);
                                q = A.get(k + 1, k - 1);
                                r = 0.0;
                                if k + 1 != nn {
                                    r = A.get(k + 2, k - 1);
                                }
                                x = p.abs() + q.abs() +r.abs();
                                if x != 0.0 {
                                    p /= x;
                                    q /= x;
                                    r /= x;
                                }
                            }
                            let s = (p * p + q * q + r * r).sqrt().copysign(p);
                            if s != 0.0 {
                                if k == m {
                                    if l != m {
                                        A.set(k, k - 1, -A.get(k, k - 1));
                                    }
                                } else {
                                    A.set(k, k - 1, -s * x);
                                }
                                p += s;
                                x = p / s;
                                y = q / s;
                                z = r / s;
                                q /= p;
                                r /= p;
                                for j in k..n {
                                    p = A.get(k, j) + q * A.get(k + 1, j);
                                    if k + 1 != nn {
                                        p += r * A.get(k + 2, j);
                                        A.sub_element_mut(k + 2, j, p * z);
                                    }
                                    A.sub_element_mut(k + 1, j, p * y);
                                    A.sub_element_mut(k, j, p * x);
                                }
                                let mmin;
                                if nn < k + 3 {
                                    mmin = nn;
                                } else {
                                    mmin = k + 3;
                                }                        
                                for i in 0..mmin+1 {
                                    p = x * A.get(i, k) + y * A.get(i, k + 1);
                                    if k + 1 != nn {
                                        p += z * A.get(i, k + 2);
                                        A.sub_element_mut(i, k + 2, p * r);
                                    }
                                    A.sub_element_mut(i, k + 1, p * q);
                                    A.sub_element_mut(i, k, p);
                                }
                                for i in 0..n {
                                    p = x * V.get(i, k) + y * V.get(i, k + 1);
                                    if k + 1 != nn {
                                        p += z * V.get(i, k + 2);
                                        V.sub_element_mut(i, k + 2, p * r);
                                    }
                                    V.sub_element_mut(i, k + 1, p * q);
                                    V.sub_element_mut(i, k, p);
                                }
                            }
                        }
                    }
                }
                if l + 1 >= nn {
                    break;
                }
            };
        }

        if anorm != 0f64 {
            for nn in (0..n).rev() {
                p = d[nn];
                q = e[nn];
                let na = nn.wrapping_sub(1);
                if q == 0f64 {
                    let mut m = nn;
                    A.set(nn, nn, 1.0);
                    if nn > 0 {
                        let mut i = nn - 1;
                        loop {
                            let w = A.get(i, i) - p;
                            r = 0.0;
                            for j in m..=nn {
                                r += A.get(i, j) * A.get(j, nn);
                            }
                            if e[i] < 0.0 {
                                z = w;
                                s = r;
                            } else {
                                m = i;

                                if e[i] == 0.0 {
                                    t = w;
                                    if t == 0.0 {
                                        t = std::f64::EPSILON * anorm;
                                    }
                                    A.set(i, nn, -r / t);
                                } else {
                                    let x = A.get(i, i + 1);
                                    let y = A.get(i + 1, i);
                                    q = (d[i] - p).powf(2f64) + e[i].powf(2f64);
                                    t = (x * s - z * r) / q;
                                    A.set(i, nn, t);
                                    if x.abs() > z.abs() {
                                        A.set(i + 1, nn, (-r - w * t) / x);
                                    } else {
                                        A.set(i + 1, nn, (-s - y * t) / z);
                                    }
                                }
                                t = A.get(i, nn).abs();
                                if std::f64::EPSILON * t * t > 1f64 {
                                    for j in i..=nn {
                                        A.div_element_mut(j, nn, t);
                                    }
                                }
                            }
                            if i == 0{
                                break;
                            } else {
                                i -= 1;
                            }
                        }
                    }
                } else if q < 0f64 {
                    let mut m = na;
                    if A.get(nn, na).abs() > A.get(na, nn).abs() {
                        A.set(na, na, q / A.get(nn, na));
                        A.set(na, nn, -(A.get(nn, nn) - p) / A.get(nn, na));
                    } else {
                        let temp = Complex::new(0.0, -A.get(na, nn)) / Complex::new(A.get(na, na) - p, q);                        
                        A.set(na, na, temp.re);
                        A.set(na, nn, temp.im);
                    }
                    A.set(nn, na, 0.0);
                    A.set(nn, nn, 1.0);
                    if nn >= 2 {                                     
                        for i in (0..nn - 1).rev() {
                            let w = A.get(i, i) - p;
                            let mut ra = 0f64;
                            let mut sa = 0f64;
                            for j in m..=nn {
                                ra += A.get(i, j) * A.get(j, na);
                                sa += A.get(i, j) * A.get(j, nn);
                            }
                            if e[i] < 0.0 {
                                z = w;
                                r = ra;
                                s = sa;
                            } else {
                                m = i;
                                if e[i] == 0.0 {
                                    let temp = Complex::new(-ra, -sa) / Complex::new(w, q);                                
                                    A.set(i, na, temp.re);
                                    A.set(i, nn, temp.im);
                                } else {
                                    let x = A.get(i, i + 1);
                                    let y = A.get(i + 1, i);
                                    let mut vr = (d[i] - p).powf(2f64) + (e[i]).powf(2.0) - q * q;
                                    let vi = 2.0 * q * (d[i] - p);
                                    if vr == 0.0 && vi == 0.0 {
                                        vr = std::f64::EPSILON * anorm * (w.abs() + q.abs() + x.abs() + y.abs() + z.abs());
                                    }
                                    let temp = Complex::new(x * r - z * ra + q * sa, x * s - z * sa - q * ra) / Complex::new(vr, vi);
                                    A.set(i, na, temp.re);
                                    A.set(i, nn, temp.im);
                                    if x.abs() > z.abs() + q.abs() {
                                        A.set(i + 1, na, (-ra - w * A.get(i, na) + q * A.get(i, nn)) / x);
                                        A.set(i + 1, nn, (-sa - w * A.get(i, nn) - q * A.get(i, na)) / x);
                                    } else {
                                        let temp = Complex::new(-r - y * A.get(i, na), -s - y * A.get(i, nn)) / Complex::new(z, q);
                                        A.set(i + 1, na, temp.re);
                                        A.set(i + 1, nn, temp.im);
                                    }
                                }
                            }
                            t = f64::max(A.get(i, na).abs(), A.get(i, nn).abs());
                            if std::f64::EPSILON * t * t > 1f64 {
                                for j in i..=nn {
                                    A.div_element_mut(j, na, t);
                                    A.div_element_mut(j, nn, t);
                                }
                            }
                        }
                    }
                }
            }            

            for j in (0..n).rev() {
                for i in 0..n {
                    z = 0f64;
                    for k in 0..=j {
                        z += V.get(i, k) * A.get(k, j);
                    }
                    V.set(i, j, z);
                }
            }
        }        
    }

    fn balbak(V: &mut Self, scale: &Vec<f64>) {
        let n = V.nrows;
        for i in 0..n {
            for j in 0..n {
                V.mul_element_mut(i, j, scale[i]);
            }
        }
    }

    fn sort(d: &mut Vec<f64>, e: &mut Vec<f64>, V: &mut Self) {         
        let n = d.len();
        let mut temp = vec![0f64; n];
        for j in 1..n {
            let real = d[j];
            let img = e[j];
            for k in 0..n {
                temp[k] = V.get(k, j);
            }
            let mut i = j as i32 - 1;
            while i >= 0 {
                if d[i as usize] >= d[j] {
                    break;
                }
                d[i as usize + 1] = d[i as usize];
                e[i as usize + 1] = e[i as usize];
                for k in 0..n {
                    V.set(k, i as usize + 1, V.get(k, i as usize));
                }
                i -= 1;
            }
            d[i as usize + 1] = real;
            e[i as usize + 1] = img;
            for k in 0..n {
                V.set(k, i as usize + 1, temp[k]);
            }
        }
    }
    
}

impl PartialEq for DenseMatrix {
    fn eq(&self, other: &Self) -> bool {
        if self.ncols != other.ncols || self.nrows != other.nrows {
            return false
        }

        let len = self.values.len();
        let other_len = other.values.len();

        if len != other_len {
            return false;
        }

        for i in 0..len {
            if (self.values[i] - other.values[i]).abs() > math::EPSILON {
                return false;
            }
        }

        true
    }
}

impl Into<Vec<f64>> for DenseMatrix {
    fn into(self) -> Vec<f64> {
        self.values
    }
}

impl Matrix for DenseMatrix {   
    
    type RowVector = Vec<f64>;

    fn from_row_vector(vec: Self::RowVector) -> Self{
        DenseMatrix::new(1, vec.len(), vec)
    }

    fn to_row_vector(self) -> Self::RowVector{
        self.to_raw_vector()
    }     

    fn get(&self, row: usize, col: usize) -> f64 {
        self.values[col*self.nrows + row]
    }

    fn get_row_as_vec(&self, row: usize) -> Vec<f64>{
        let mut result = vec![0f64; self.ncols];
        for c in 0..self.ncols {
            result[c] = self.get(row, c);
        }
        result
    }

    fn get_col_as_vec(&self, col: usize) -> Vec<f64>{
        let mut result = vec![0f64; self.nrows];
        for r in 0..self.nrows {
            result[r] = self.get(r, col);
        }        
        result
    }

    fn set(&mut self, row: usize, col: usize, x: f64) {
        self.values[col*self.nrows + row] = x;        
    }

    fn zeros(nrows: usize, ncols: usize) -> DenseMatrix {
        DenseMatrix::fill(nrows, ncols, 0f64)
    }

    fn ones(nrows: usize, ncols: usize) -> DenseMatrix {
        DenseMatrix::fill(nrows, ncols, 1f64)
    }    

    fn eye(size: usize) -> Self {
        let mut matrix = Self::zeros(size, size);

        for i in 0..size {
            matrix.set(i, i, 1.0);
        }

        return matrix;
    }

    fn to_raw_vector(&self) -> Vec<f64>{
        let mut v = vec![0.; self.nrows * self.ncols];

        for r in 0..self.nrows{
            for c in 0..self.ncols {
                v[r * self.ncols + c] = self.get(r, c);
            }
        }
        
        v
    }

    fn shape(&self) -> (usize, usize) {
        (self.nrows, self.ncols)
    }

    fn h_stack(&self, other: &Self) -> Self {
        if self.ncols != other.ncols {
            panic!("Number of columns in both matrices should be equal");
        }
        let mut result = DenseMatrix::zeros(self.nrows + other.nrows, self.ncols);
        for c in 0..self.ncols {
            for r in 0..self.nrows+other.nrows {                 
                if r <  self.nrows {              
                    result.set(r, c, self.get(r, c));
                } else {
                    result.set(r, c, other.get(r - self.nrows, c));
                }
            }
        }    
        result    
    }

    fn v_stack(&self, other: &Self) -> Self{
        if self.nrows != other.nrows {
            panic!("Number of rows in both matrices should be equal");
        }
        let mut result = DenseMatrix::zeros(self.nrows, self.ncols + other.ncols);
        for r in 0..self.nrows {
            for c in 0..self.ncols+other.ncols {                             
                if c <  self.ncols {              
                    result.set(r, c, self.get(r, c));
                } else {
                    result.set(r, c, other.get(r, c - self.ncols));
                }
            }
        }
        result
    }

    fn dot(&self, other: &Self) -> Self {

        if self.ncols != other.nrows {
            panic!("Number of rows of A should equal number of columns of B");
        }
        let inner_d = self.ncols;
        let mut result = DenseMatrix::zeros(self.nrows, other.ncols);

        for r in 0..self.nrows {
            for c in 0..other.ncols {
                let mut s = 0f64;
                for i in 0..inner_d {
                    s += self.get(r, i) * other.get(i, c);
                }
                result.set(r, c, s);
            }
        }

        result
    }    

    fn vector_dot(&self, other: &Self) -> f64 {
        if (self.nrows != 1 || self.nrows != 1) && (other.nrows != 1 || other.ncols != 1) {
            panic!("A and B should both be 1-dimentional vectors.");
        }
        if self.nrows * self.ncols != other.nrows * other.ncols {
            panic!("A and B should have the same size");
        }        

        let mut result = 0f64;
        for i in 0..(self.nrows * self.ncols) {
            result += self.values[i] * other.values[i];            
        }

        result
    }  

    fn slice(&self, rows: Range<usize>, cols: Range<usize>) -> DenseMatrix {

        let ncols = cols.len();
        let nrows = rows.len();

        let mut m = DenseMatrix::new(nrows, ncols, vec![0f64; nrows * ncols]);

        for r in rows.start..rows.end {
            for c in cols.start..cols.end {
                m.set(r-rows.start, c-cols.start, self.get(r, c));
            }
        }

        m
    }    

    fn qr_solve_mut(&mut self, mut b: DenseMatrix) -> DenseMatrix {
        let m = self.nrows;
        let n = self.ncols;
        let nrhs = b.ncols;

        if self.nrows != b.nrows {
            panic!("Dimensions do not agree. Self.nrows should equal b.nrows but is {}, {}", self.nrows, b.nrows);
        }

        let mut r_diagonal: Vec<f64> = vec![0f64; n];

        for k in 0..n {
            let mut nrm = 0f64;
            for i in k..m {
                nrm = nrm.hypot(self.get(i, k));
            }

            if nrm.abs() > math::EPSILON {

                if self.get(k, k) < 0f64 {
                    nrm = -nrm;
                }
                for i in k..m {
                    self.div_element_mut(i, k, nrm);
                }
                self.add_element_mut(k, k, 1f64);

                for j in k+1..n {
                    let mut s = 0f64;
                    for i in k..m {
                        s += self.get(i, k) * self.get(i, j);
                    }
                    s = -s / self.get(k, k);
                    for i in k..m {
                        self.add_element_mut(i, j, s * self.get(i, k));
                    }
                }
            }            
            r_diagonal[k] = -nrm;
        }        
        
        for j in 0..r_diagonal.len() {
            if r_diagonal[j].abs() < math::EPSILON  {
                panic!("Matrix is rank deficient.");                
            }
        }

        for k in 0..n {
            for j in 0..nrhs {
                let mut s = 0f64;
                for i in k..m {
                    s += self.get(i, k) * b.get(i, j);
                }
                s = -s / self.get(k, k);
                for i in k..m {
                    b.add_element_mut(i, j, s * self.get(i, k));
                }
            }
        }         

        for k in (0..n).rev() {
            for j in 0..nrhs {
                b.set(k, j, b.get(k, j) / r_diagonal[k]);
            }
            
            for i in 0..k {
                for j in 0..nrhs {
                    b.sub_element_mut(i, j, b.get(k, j) * self.get(i, k));
                }
            }
        }

        b

    }

    fn svd(&self) -> SVD<Self> {

        let mut U = self.clone();        

        let m = U.nrows;
        let n = U.ncols;
        
        let (mut l, mut nm) = (0usize, 0usize);
        let (mut anorm, mut g, mut scale) = (0f64, 0f64, 0f64);        
        
        let mut v = DenseMatrix::zeros(n, n);
        let mut w = vec![0f64; n];
        let mut rv1 = vec![0f64; n];

        for i in 0..n {
            l = i + 2;
            rv1[i] = scale * g;
            g = 0f64;
            let mut s = 0f64;
            scale = 0f64;

            if i < m {
                for k in i..m {
                    scale += U.get(k, i).abs();
                }

                if scale.abs() > math::EPSILON {

                    for k in i..m {
                        U.div_element_mut(k, i, scale);
                        s += U.get(k, i) * U.get(k, i);
                    }

                    let mut f = U.get(i, i);
                    g = -s.sqrt().copysign(f);                    
                    let h = f * g - s;
                    U.set(i, i, f - g);
                    for j in l - 1..n {
                        s = 0f64;
                        for k in i..m {
                            s += U.get(k, i) * U.get(k, j);
                        }
                        f = s / h;
                        for k in i..m {
                            U.add_element_mut(k, j, f * U.get(k, i));
                        }
                    }
                    for k in i..m {
                        U.mul_element_mut(k, i, scale);
                    }
                }
            }

            w[i] = scale * g;
            g = 0f64;
            let mut s = 0f64;
            scale = 0f64;

            if i + 1 <= m && i + 1 != n {
                for k in l - 1..n {
                    scale += U.get(i, k).abs();
                }

                if scale.abs() > math::EPSILON  {
                    for k in l - 1..n {
                        U.div_element_mut(i, k, scale);
                        s += U.get(i, k) * U.get(i, k);
                    }

                    let f = U.get(i, l - 1);
                    g = -s.sqrt().copysign(f);                    
                    let h = f * g - s;
                    U.set(i, l - 1, f - g);

                    for k in l - 1..n {
                        rv1[k] = U.get(i, k) / h;
                    }

                    for j in l - 1..m {
                        s = 0f64;
                        for k in l - 1..n {
                            s += U.get(j, k) * U.get(i, k);
                        }

                        for k in l - 1..n {
                            U.add_element_mut(j, k, s * rv1[k]);
                        }
                    }

                    for k in l - 1..n {
                        U.mul_element_mut(i, k, scale);
                    }
                }
            }

            
            anorm = f64::max(anorm, w[i].abs() + rv1[i].abs());
        }

        for i in (0..n).rev() {
            if i < n - 1 {
                if g != 0.0 {
                    for j in l..n {
                        v.set(j, i, (U.get(i, j) / U.get(i, l)) / g);
                    }
                    for j in l..n {
                        let mut s = 0f64;
                        for k in l..n {
                            s += U.get(i, k) * v.get(k, j);
                        }
                        for k in l..n {
                            v.add_element_mut(k, j, s * v.get(k, i));
                        }
                    }
                }
                for j in l..n {
                    v.set(i, j, 0f64);
                    v.set(j, i, 0f64);
                }
            }
            v.set(i, i, 1.0);
            g = rv1[i];
            l = i;
        }

        for i in (0..usize::min(m, n)).rev() {
            l = i + 1;
            g = w[i];
            for j in l..n {
                U.set(i, j, 0f64);
            }

            if g.abs() > math::EPSILON {
                g = 1f64 / g;
                for j in l..n {
                    let mut s = 0f64;
                    for k in l..m {
                        s += U.get(k, i) * U.get(k, j);
                    }
                    let f = (s / U.get(i, i)) * g;
                    for k in i..m {
                        U.add_element_mut(k, j, f * U.get(k, i));
                    }
                }
                for j in i..m {
                    U.mul_element_mut(j, i, g);
                }
            } else {
                for j in i..m {
                    U.set(j, i, 0f64);
                }
            }

            U.add_element_mut(i, i, 1f64);
        }

        for k in (0..n).rev() {
            for iteration in 0..30 {
                let mut flag = true;                
                l = k;
                while l != 0 {                               
                    if l == 0 || rv1[l].abs() <= math::EPSILON * anorm {
                        flag = false;   
                        break;
                    }
                    nm = l - 1;
                    if w[nm].abs() <= math::EPSILON * anorm {
                        break;
                    }
                    l -= 1;
                }

                if flag {
                    let mut c = 0.0;
                    let mut s = 1.0;
                    for i in l..k+1 {
                        let f = s * rv1[i];
                        rv1[i] = c * rv1[i];
                        if f.abs() <= math::EPSILON * anorm {
                            break;
                        }
                        g = w[i];
                        let mut h = f.hypot(g);
                        w[i] = h;
                        h = 1.0 / h;
                        c = g * h;
                        s = -f * h;
                        for j in 0..m {
                            let y = U.get(j, nm);
                            let z = U.get(j, i);
                            U.set(j, nm, y * c + z * s);
                            U.set(j,  i, z * c - y * s);
                        }
                    }
                }

                let z = w[k];
                if l == k {
                    if z < 0f64 {
                        w[k] = -z;
                        for j in 0..n {
                            v.set(j, k, -v.get(j, k));
                        }
                    }
                    break;
                }

                if iteration == 29 {
                    panic!("no convergence in 30 iterations");
                }

                let mut x = w[l];
                nm = k - 1;
                let mut y = w[nm];
                g = rv1[nm];
                let mut h = rv1[k];
                let mut f = ((y - z) * (y + z) + (g - h) * (g + h)) / (2.0 * h * y);
                g = f.hypot(1.0);
                f = ((x - z) * (x + z) + h * ((y / (f + g.copysign(f))) - h)) / x;
                let mut c = 1f64;
                let mut s = 1f64;

                for j in l..=nm {
                    let i = j + 1;
                    g = rv1[i];
                    y = w[i];
                    h = s * g;
                    g = c * g;
                    let mut z = f.hypot(h);
                    rv1[j] = z;
                    c = f / z;
                    s = h / z;
                    f = x * c + g * s;
                    g = g * c - x * s;
                    h = y * s;
                    y *= c;

                    for jj in 0..n {
                        x = v.get(jj, j);
                        z = v.get(jj, i);
                        v.set(jj, j, x * c + z * s);
                        v.set(jj, i, z * c - x * s);
                    }

                    z = f.hypot(h);
                    w[j] = z;
                    if z.abs() > math::EPSILON {
                        z = 1.0 / z;
                        c = f * z;
                        s = h * z;
                    }

                    f = c * g + s * y;
                    x = c * y - s * g;
                    for jj in 0..m {
                        y = U.get(jj, j);
                        z = U.get(jj, i);
                        U.set(jj, j, y * c + z * s);
                        U.set(jj, i, z * c - y * s);
                    }
                }

                rv1[l] = 0.0;
                rv1[k] = f;
                w[k] = x;
            }
        }
        
        let mut inc = 1usize;        
        let mut su = vec![0f64; m];
        let mut sv = vec![0f64; n];
        
        loop {
            inc *= 3;
            inc += 1;
            if inc > n {
                break;
            }
        }

        loop {
            inc /= 3;
            for  i in inc..n {                
                let sw = w[i];
                for k in 0..m {
                    su[k] = U.get(k, i);
                }
                for k in 0..n {
                    sv[k] = v.get(k, i);
                }
                let mut j = i;
                while w[j - inc] < sw {
                    w[j] = w[j - inc];
                    for k in 0..m {
                        U.set(k, j, U.get(k, j - inc));
                    }
                    for k in 0..n {
                        v.set(k, j, v.get(k, j - inc));
                    }
                    j -= inc;
                    if j < inc {
                        break;
                    }
                }
                w[j] = sw;
                for k in 0..m {
                    U.set(k, j, su[k]);
                }
                for k in 0..n {
                    v.set(k, j, sv[k]);
                }

            }
            if inc <= 1 {
                break;
            }
        }

        for k in 0..n {
            let mut s = 0.;
            for i in 0..m {
                if U.get(i, k) < 0. {
                    s += 1.;
                }
            }
            for j in 0..n {
                if v.get(j, k) < 0. {
                    s += 1.;
                }
            }
            if s > (m + n) as f64 / 2. {
                for i in 0..m {
                    U.set(i, k, -U.get(i, k));
                }
                for j in 0..n {
                    v.set(j, k, -v.get(j, k));
                }
            }
        }        

        SVD::new(U, v, w)

    }   
    
    fn evd_mut(mut self, symmetric: bool) -> EVD<Self>{
        if self.ncols != self.nrows {
            panic!("Matrix is not square: {} x {}", self.nrows, self.ncols);
        }

        let n = self.nrows;
        let mut d = vec![0f64; n];
        let mut e = vec![0f64; n];        
        
        let mut V;
        if symmetric {            
            V = self;
            // Tridiagonalize.
            V.tred2(&mut d, &mut e);
            // Diagonalize.
            V.tql2(&mut d, &mut e);

        } else {
            let scale = Self::balance(&mut self);            
            
            let perm = Self::elmhes(&mut self);            

            V = Self::eye(n);            

            Self::eltran(&self, &mut V, &perm);            

            Self::hqr2(&mut self, &mut V, &mut d, &mut e);
            Self::balbak(&mut V, &scale);
            Self::sort(&mut d, &mut e, &mut V);
        }

        EVD {
            V: V,
            d: d,
            e: e
        }
    }    

    fn approximate_eq(&self, other: &Self, error: f64) -> bool {
        if self.ncols != other.ncols || self.nrows != other.nrows {
            return false
        }

        for c in 0..self.ncols {
            for r in 0..self.nrows {
                if (self.get(r, c) - other.get(r, c)).abs() > error {
                    return false
                }
            }
        }

        true
    }

    fn fill(nrows: usize, ncols: usize, value: f64) -> Self {
        DenseMatrix::new(nrows, ncols, vec![value; ncols * nrows])
    }

    fn add_mut(&mut self, other: &Self) -> &Self {
        if self.ncols != other.ncols || self.nrows != other.nrows {
            panic!("A and B should have the same shape");
        }        
        for c in 0..self.ncols {
            for r in 0..self.nrows {
                self.add_element_mut(r, c, other.get(r, c));
            }
        }

        self
    }

    fn sub_mut(&mut self, other: &Self) -> &Self {
        if self.ncols != other.ncols || self.nrows != other.nrows {
            panic!("A and B should have the same shape");
        }        
        for c in 0..self.ncols {
            for r in 0..self.nrows {
                self.sub_element_mut(r, c, other.get(r, c));
            }
        }

        self
    }

    fn mul_mut(&mut self, other: &Self) -> &Self {
        if self.ncols != other.ncols || self.nrows != other.nrows {
            panic!("A and B should have the same shape");
        }        
        for c in 0..self.ncols {
            for r in 0..self.nrows {
                self.mul_element_mut(r, c, other.get(r, c));
            }
        }

        self
    }

    fn div_mut(&mut self, other: &Self) -> &Self {
        if self.ncols != other.ncols || self.nrows != other.nrows {
            panic!("A and B should have the same shape");
        }        
        for c in 0..self.ncols {
            for r in 0..self.nrows {
                self.div_element_mut(r, c, other.get(r, c));
            }
        }

        self
    }

    fn div_element_mut(&mut self, row: usize, col: usize, x: f64) {
        self.values[col*self.nrows + row] /= x;
    }

    fn mul_element_mut(&mut self, row: usize, col: usize, x: f64) {
        self.values[col*self.nrows + row] *= x;
    }

    fn add_element_mut(&mut self, row: usize, col: usize, x: f64) {
        self.values[col*self.nrows + row] += x
    }

    fn sub_element_mut(&mut self, row: usize, col: usize, x: f64) {
        self.values[col*self.nrows + row] -= x;
    }  

    fn generate_positive_definite(nrows: usize, ncols: usize) -> Self {
        let m = DenseMatrix::rand(nrows, ncols);
        m.dot(&m.transpose())
    }

    fn transpose(&self) -> Self {
        let mut m = DenseMatrix {
            ncols: self.nrows,
            nrows: self.ncols,
            values: vec![0f64; self.ncols * self.nrows]
        };
        for c in 0..self.ncols {
            for r in 0..self.nrows {
                m.set(c, r, self.get(r, c));
            }
        }
        m

    }

    fn rand(nrows: usize, ncols: usize) -> Self {
        let mut rng = rand::thread_rng();
        let values: Vec<f64> = (0..nrows*ncols).map(|_| {
            rng.gen()
        }).collect();
        DenseMatrix {
            ncols: ncols,
            nrows: nrows,
            values: values
        }
    }    

    fn norm2(&self) -> f64 {
        let mut norm = 0f64;

        for xi in self.values.iter() {
            norm += xi * xi;
        }

        norm.sqrt()
    }

    fn norm(&self, p:f64) -> f64 {

        if p.is_infinite() && p.is_sign_positive() {
            self.values.iter().map(|x| x.abs()).fold(std::f64::NEG_INFINITY, |a, b| a.max(b))
        } else if p.is_infinite() && p.is_sign_negative() {
            self.values.iter().map(|x| x.abs()).fold(std::f64::INFINITY, |a, b| a.min(b))
        } else {

            let mut norm = 0f64;

            for xi in self.values.iter() {
                norm += xi.abs().powf(p);
            }

            norm.powf(1.0/p)
        }
    }

    fn column_mean(&self) -> Vec<f64> {
        let mut mean = vec![0f64; self.ncols];

        for r in 0..self.nrows {
            for c in 0..self.ncols {
                mean[c] += self.get(r, c);
            }
        }

        for i in 0..mean.len() {
            mean[i] /= self.nrows as f64;
        }

        mean
    }

    fn add_scalar_mut(&mut self, scalar: f64) -> &Self {
        for i in 0..self.values.len() {
            self.values[i] += scalar;
        }
        self
    }

    fn sub_scalar_mut(&mut self, scalar: f64) -> &Self {
        for i in 0..self.values.len() {
            self.values[i] -= scalar;
        }
        self
    }

    fn mul_scalar_mut(&mut self, scalar: f64) -> &Self {
        for i in 0..self.values.len() {
            self.values[i] *= scalar;
        }
        self
    }

    fn div_scalar_mut(&mut self, scalar: f64) -> &Self {
        for i in 0..self.values.len() {
            self.values[i] /= scalar;
        }
        self
    }

    fn negative_mut(&mut self) {
        for i in 0..self.values.len() {
            self.values[i] = -self.values[i];
        }
    }    

    fn reshape(&self, nrows: usize, ncols: usize) -> Self {
        if self.nrows * self.ncols != nrows * ncols {
            panic!("Can't reshape {}x{} matrix into {}x{}.", self.nrows, self.ncols, nrows, ncols);
        }
        let mut dst = DenseMatrix::zeros(nrows, ncols);
        let mut dst_r = 0;
        let mut dst_c = 0;
        for r in 0..self.nrows {
            for c in 0..self.ncols {
                dst.set(dst_r, dst_c, self.get(r, c));
                if dst_c + 1 >= ncols {
                    dst_c = 0;
                    dst_r += 1;
                } else {
                    dst_c += 1;
                }
            }
        }
        dst
    }

    fn copy_from(&mut self, other: &Self) {

        if self.nrows != other.nrows || self.ncols != other.ncols {
            panic!("Can't copy {}x{} matrix into {}x{}.", self.nrows, self.ncols, other.nrows, other.ncols);
        }

        for i in 0..self.values.len() {
            self.values[i] = other.values[i];
        }
    }

    fn abs_mut(&mut self) -> &Self{
        for i in 0..self.values.len() {
            self.values[i] = self.values[i].abs();
        }
        self
    }

    fn max_diff(&self, other: &Self) -> f64{
        let mut max_diff = 0f64;
        for i in 0..self.values.len() {
            max_diff = max_diff.max((self.values[i] - other.values[i]).abs());
        }
        max_diff

    }

    fn sum(&self) -> f64 {
        let mut sum = 0.;
        for i in 0..self.values.len() {
            sum += self.values[i];
        }
        sum
    }

    fn softmax_mut(&mut self) {
        let max = self.values.iter().map(|x| x.abs()).fold(std::f64::NEG_INFINITY, |a, b| a.max(b));
        let mut z = 0.;
        for r in 0..self.nrows {
            for c in 0..self.ncols {
                let p = (self.get(r, c) - max).exp();
                self.set(r, c, p);
                z += p;
            }
        }
        for r in 0..self.nrows {
            for c in 0..self.ncols {
                self.set(r, c, self.get(r, c) / z);
            }
        }
    }

    fn pow_mut(&mut self, p: f64) -> &Self {
        for i in 0..self.values.len() {
            self.values[i] = self.values[i].powf(p);
        }
        self
    }

    fn argmax(&self) -> Vec<usize> {

        let mut res = vec![0usize; self.nrows];

        for r in 0..self.nrows {
            let mut max = std::f64::NEG_INFINITY;
            let mut max_pos = 0usize;
            for c in 0..self.ncols {
                let v = self.get(r, c);
                if max < v{
                    max = v;
                    max_pos = c; 
                }
            }
            res[r] = max_pos;
        }

        res

    }

    fn unique(&self) -> Vec<f64> {
        let mut result = self.values.clone();
        result.sort_by(|a, b| a.partial_cmp(b).unwrap());
        result.dedup();
        result
    }

}

#[cfg(test)]
mod tests {    
    use super::*; 

    #[test]
    fn from_to_row_vec() { 

        let vec = vec![ 1.,  2.,  3.];
        assert_eq!(DenseMatrix::from_row_vector(vec.clone()), DenseMatrix::new(1, 3, vec![1., 2., 3.]));
        assert_eq!(DenseMatrix::from_row_vector(vec.clone()).to_row_vector(), vec![1., 2., 3.]);

    }

    #[test]
    fn qr_solve_mut() { 

            let mut a = DenseMatrix::from_array(&[&[0.9, 0.4, 0.7], &[0.4, 0.5, 0.3], &[0.7, 0.3, 0.8]]);
            let b = DenseMatrix::from_array(&[&[0.5, 0.2],&[0.5, 0.8], &[0.5, 0.3]]);
            let expected_w = DenseMatrix::new(3, 2, vec![-0.20, 0.87, 0.47, -1.28, 2.22, 0.66]);
            let w = a.qr_solve_mut(b);   
            assert!(w.approximate_eq(&expected_w, 1e-2));
    }    

    #[test]
    fn h_stack() { 

            let a = DenseMatrix::from_array(
                &[
                    &[1., 2., 3.],
                    &[4., 5., 6.],
                    &[7., 8., 9.]]);
            let b = DenseMatrix::from_array(
                &[
                    &[1., 2., 3.],
                    &[4., 5., 6.]]);
            let expected = DenseMatrix::from_array(
                &[
                    &[1., 2., 3.], 
                    &[4., 5., 6.], 
                    &[7., 8., 9.], 
                    &[1., 2., 3.], 
                    &[4., 5., 6.]]);
            let result = a.h_stack(&b);               
            assert_eq!(result, expected);
    }

    #[test]
    fn v_stack() { 

            let a = DenseMatrix::from_array(
                &[
                    &[1., 2., 3.],
                    &[4., 5., 6.],
                    &[7., 8., 9.]]);
            let b = DenseMatrix::from_array(
                &[
                    &[1., 2.],
                    &[3., 4.],
                    &[5., 6.]]);
            let expected = DenseMatrix::from_array(
                &[
                    &[1., 2., 3., 1., 2.], 
                    &[4., 5., 6., 3., 4.], 
                    &[7., 8., 9., 5., 6.]]);
            let result = a.v_stack(&b);               
            assert_eq!(result, expected);
    }

    #[test]
    fn dot() { 

            let a = DenseMatrix::from_array(
                &[
                    &[1., 2., 3.],
                    &[4., 5., 6.]]);
            let b = DenseMatrix::from_array(
                &[
                    &[1., 2.],
                    &[3., 4.],
                    &[5., 6.]]);
            let expected = DenseMatrix::from_array(
                &[
                    &[22., 28.], 
                    &[49., 64.]]);
            let result = a.dot(&b);               
            assert_eq!(result, expected);
    }

    #[test]
    fn slice() { 

            let m = DenseMatrix::from_array(
                &[
                    &[1., 2., 3., 1., 2.], 
                    &[4., 5., 6., 3., 4.], 
                    &[7., 8., 9., 5., 6.]]);
            let expected = DenseMatrix::from_array(
                &[
                    &[2., 3.], 
                    &[5., 6.]]);
            let result = m.slice(0..2, 1..3);
            assert_eq!(result, expected);
    }
    

    #[test]
    fn approximate_eq() {             
            let m = DenseMatrix::from_array(
                &[
                    &[2., 3.], 
                    &[5., 6.]]);
            let m_eq = DenseMatrix::from_array(
                &[
                    &[2.5, 3.0], 
                    &[5., 5.5]]);
            let m_neq = DenseMatrix::from_array(
                &[
                    &[3.0, 3.0], 
                    &[5., 6.5]]);            
            assert!(m.approximate_eq(&m_eq, 0.5));
            assert!(!m.approximate_eq(&m_neq, 0.5));
    }

    #[test]
    fn rand() {
        let m = DenseMatrix::rand(3, 3);
        for c in 0..3 {
            for r in 0..3 {
                assert!(m.get(r, c) != 0f64);
            }
        }
    }

    #[test]
    fn transpose() {
        let m = DenseMatrix::from_array(&[&[1.0, 3.0], &[2.0, 4.0]]);
        let expected = DenseMatrix::from_array(&[&[1.0, 2.0], &[3.0, 4.0]]);
        let m_transposed = m.transpose();
        for c in 0..2 {
            for r in 0..2 {
                assert!(m_transposed.get(r, c) == expected.get(r, c));
            }
        }
    }

    #[test]
    fn generate_positive_definite() {
        let m = DenseMatrix::generate_positive_definite(3, 3);        
    }

    #[test]
    fn reshape() {
        let m_orig = DenseMatrix::vector_from_array(&[1., 2., 3., 4., 5., 6.]);
        let m_2_by_3 = m_orig.reshape(2, 3);
        let m_result = m_2_by_3.reshape(1, 6);        
        assert_eq!(m_2_by_3.shape(), (2, 3));
        assert_eq!(m_2_by_3.get(1, 1), 5.);
        assert_eq!(m_result.get(0, 1), 2.);
        assert_eq!(m_result.get(0, 3), 4.);
    }

    #[test]
    fn norm() { 

            let v = DenseMatrix::vector_from_array(&[3., -2., 6.]);            
            assert_eq!(v.norm(1.), 11.);
            assert_eq!(v.norm(2.), 7.);
            assert_eq!(v.norm(std::f64::INFINITY), 6.);
            assert_eq!(v.norm(std::f64::NEG_INFINITY), 2.);
    }

    #[test]
    fn softmax_mut() { 

            let mut prob = DenseMatrix::vector_from_array(&[1., 2., 3.]);  
            prob.softmax_mut();            
            assert!((prob.get(0, 0) - 0.09).abs() < 0.01);     
            assert!((prob.get(0, 1) - 0.24).abs() < 0.01);     
            assert!((prob.get(0, 2) - 0.66).abs() < 0.01);            
    }  
    
    #[test]
    fn col_mean(){
        let a = DenseMatrix::from_array(&[
                       &[1., 2., 3.],
                       &[4., 5., 6.], 
                       &[7., 8., 9.]]);  
        let res = a.column_mean();
        assert_eq!(res, vec![4., 5., 6.]);        
    }

    #[test]
    fn eye(){
        let a = DenseMatrix::from_array(&[
                       &[1., 0., 0.],
                       &[0., 1., 0.], 
                       &[0., 0., 1.]]);  
        let res = DenseMatrix::eye(3);
        assert_eq!(res, a);
    }

}
