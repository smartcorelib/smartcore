#![allow(non_snake_case)]

use num::complex::Complex;
use crate::linalg::BaseMatrix;

#[derive(Debug, Clone)]
pub struct EVD<M: BaseMatrix> {        
    pub d: Vec<f64>,
    pub e: Vec<f64>,
    pub V: M
}

impl<M: BaseMatrix> EVD<M> {
    pub fn new(V: M, d: Vec<f64>, e: Vec<f64>) -> EVD<M> {
        EVD {
            d: d,
            e: e,
            V: V
        }
    }
}

pub trait EVDDecomposableMatrix: BaseMatrix {    
    
    fn evd(&self, symmetric: bool) -> EVD<Self>{
        self.clone().evd_mut(symmetric)
    }

    fn evd_mut(mut self, symmetric: bool) -> EVD<Self>{
        let(nrows, ncols) = self.shape();
        if ncols != nrows {
            panic!("Matrix is not square: {} x {}", nrows, ncols);
        }

        let n = nrows;
        let mut d = vec![0f64; n];
        let mut e = vec![0f64; n];        
        
        let mut V;
        if symmetric {            
            V = self;
            // Tridiagonalize.
            tred2(&mut V, &mut d, &mut e);
            // Diagonalize.
            tql2(&mut V, &mut d, &mut e);

        } else {
            let scale = balance(&mut self);            
            
            let perm = elmhes(&mut self);            

            V = Self::eye(n);            

            eltran(&self, &mut V, &perm);            

            hqr2(&mut self, &mut V, &mut d, &mut e);
            balbak(&mut V, &scale);
            sort(&mut d, &mut e, &mut V);
        }

        EVD {
            V: V,
            d: d,
            e: e
        }
    }
}

fn tred2<M: BaseMatrix>(V: &mut M, d: &mut Vec<f64>, e: &mut Vec<f64>) {
    
    let (n, _) = V.shape();
    for i in 0..n {
        d[i] = V.get(n - 1, i);
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
                d[j] = V.get(i - 1, j);
                V.set(i, j, 0.0);
                V.set(j, i, 0.0);
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
                V.set(j, i, f);
                g = e[j] + V.get(j, j) * f;
                for k in j + 1..=i - 1 {
                    g += V.get(k, j) * d[k];
                    e[k] += V.get(k, j) * f;
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
                    V.sub_element_mut(k, j, f * e[k] + g * d[k]);
                }
                d[j] = V.get(i - 1, j);
                V.set(i, j, 0.0);
            }
        }
        d[i] = h;
    }        

    // Accumulate transformations.
    for i in 0..n-1 {
        V.set(n - 1, i, V.get(i, i));
        V.set(i, i, 1.0);
        let h = d[i + 1];
        if h != 0f64 {
            for k in 0..=i {
                d[k] = V.get(k, i + 1) / h;
            }
            for j in 0..=i {
                let mut g = 0f64;
                for k in 0..=i {
                    g += V.get(k, i + 1) * V.get(k, j);
                }
                for k in 0..=i {
                    V.sub_element_mut(k, j,  g * d[k]);
                }
            }
        }
        for k in 0..=i {
            V.set(k, i + 1, 0.0);
        }
    }
    for j in 0..n {
        d[j] = V.get(n - 1, j);
        V.set(n - 1, j, 0.0);
    }
    V.set(n - 1, n - 1, 1.0);
    e[0] = 0.0;
}

fn tql2<M: BaseMatrix>(V: &mut M, d: &mut Vec<f64>, e: &mut Vec<f64>) {
    let (n, _) = V.shape();
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
                        h = V.get(k, i + 1);
                        V.set(k, i + 1, s * V.get(k, i) + c * h);
                        V.set(k, i,     c * V.get(k, i) - s * h);
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
                p = V.get(j, i);
                V.set(j, i, V.get(j, k));
                V.set(j, k, p);
            }
        }
    }
}

fn balance<M: BaseMatrix>(A: &mut M) -> Vec<f64> {
    let radix = 2f64;
    let sqrdx = radix * radix;

    let (n, _) = A.shape();

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

fn elmhes<M: BaseMatrix>(A: &mut M) -> Vec<usize> {
    let (n, _) = A.shape();
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

fn eltran<M: BaseMatrix>(A: &M, V: &mut M, perm: &Vec<usize>) {
    let (n, _) = A.shape();
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

fn hqr2<M: BaseMatrix>(A: &mut M, V: &mut M, d: &mut Vec<f64>, e: &mut Vec<f64>) {
    let (n, _) = A.shape();                                                           
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

fn balbak<M: BaseMatrix>(V: &mut M, scale: &Vec<f64>) {
    let (n, _) = V.shape();
    for i in 0..n {
        for j in 0..n {
            V.mul_element_mut(i, j, scale[i]);
        }
    }
}

fn sort<M: BaseMatrix>(d: &mut Vec<f64>, e: &mut Vec<f64>, V: &mut M) {         
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

#[cfg(test)]
mod tests {    
    use super::*; 
    use crate::linalg::naive::dense_matrix::DenseMatrix;

    #[test]
    fn decompose_symmetric() { 

        let A = DenseMatrix::from_array(&[
            &[0.9000, 0.4000, 0.7000],
            &[0.4000, 0.5000, 0.3000],
            &[0.7000, 0.3000, 0.8000]]);

        let eigen_values = vec![1.7498382, 0.3165784, 0.1335834];

        let eigen_vectors = DenseMatrix::from_array(&[
            &[0.6881997, -0.07121225, 0.7220180],
            &[0.3700456, 0.89044952, -0.2648886],
            &[0.6240573, -0.44947578, -0.6391588]
        ]);

        let evd = A.evd(true);        

        assert!(eigen_vectors.abs().approximate_eq(&evd.V.abs(), 1e-4));        
        for i in 0..eigen_values.len() {
            assert!((eigen_values[i] - evd.d[i]).abs() < 1e-4);
        }
        for i in 0..eigen_values.len() {
            assert!((0f64 - evd.e[i]).abs() < std::f64::EPSILON);
        }

    }

    #[test]
    fn decompose_asymmetric() { 

        let A = DenseMatrix::from_array(&[
            &[0.9000, 0.4000, 0.7000],
            &[0.4000, 0.5000, 0.3000],
            &[0.8000, 0.3000, 0.8000]]);

        let eigen_values = vec![1.79171122, 0.31908143, 0.08920735];

        let eigen_vectors = DenseMatrix::from_array(&[
            &[0.7178958,  0.05322098,  0.6812010],
            &[0.3837711, -0.84702111, -0.1494582],
            &[0.6952105,  0.43984484, -0.7036135]
        ]);

        let evd = A.evd(false);   
        
        assert!(eigen_vectors.abs().approximate_eq(&evd.V.abs(), 1e-4));        
        for i in 0..eigen_values.len() {
            assert!((eigen_values[i] - evd.d[i]).abs() < 1e-4);
        }
        for i in 0..eigen_values.len() {
            assert!((0f64 - evd.e[i]).abs() < std::f64::EPSILON);
        }

    }

    #[test]
    fn decompose_complex() { 

        let A = DenseMatrix::from_array(&[
            &[3.0, -2.0, 1.0, 1.0],
            &[4.0, -1.0, 1.0, 1.0],
            &[1.0, 1.0, 3.0, -2.0],
            &[1.0, 1.0, 4.0, -1.0]]);

        let eigen_values_d = vec![0.0, 2.0, 2.0, 0.0];
        let eigen_values_e = vec![2.2361, 0.9999, -0.9999, -2.2361];

        let eigen_vectors = DenseMatrix::from_array(&[
            &[-0.9159, -0.1378, 0.3816, -0.0806],
            &[-0.6707, 0.1059, 0.901, 0.6289],
            &[0.9159, -0.1378, 0.3816, 0.0806],
            &[0.6707, 0.1059, 0.901, -0.6289]
        ]);

        let evd = A.evd(false);   
        
        assert!(eigen_vectors.abs().approximate_eq(&evd.V.abs(), 1e-4));        
        for i in 0..eigen_values_d.len() {
            assert!((eigen_values_d[i] - evd.d[i]).abs() < 1e-4);
        }
        for i in 0..eigen_values_e.len() {
            assert!((eigen_values_e[i] - evd.e[i]).abs() < 1e-4);
        }

    }
    
}