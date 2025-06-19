
use super::STATION;

#[derive(Default)]
pub struct GpuMatrix {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f32>
}

impl GpuMatrix {
    pub fn get_workgroup_size(&self) -> usize {
        let mut size: usize = match self.cols {
            c if c <= 64 => 64,
            c if c <= 128 => 128,
            c if c <= 256 => 256,
            c if c <= 512 => 512,
            _ => 1024
        };

        if let Ok(adapter) = STATION.get_adapter() {
            if size > adapter.max_workgroup_size as usize {
                size = adapter.max_workgroup_size as usize;
            }
        }

        size
    }
}

impl From<Vec<Vec<f32>>> for GpuMatrix {
    fn from(data: Vec<Vec<f32>>) -> Self {
        Self {
            rows: data.len(),
            cols: data[0].len(),
            data: data.into_iter().flatten().collect()
        }
    }
}

impl From<&Vec<Vec<f32>>> for GpuMatrix {
    fn from(data: &Vec<Vec<f32>>) -> Self {
        Self {
            rows: data.len(),
            cols: data[0].len(),
            data: data.clone().into_iter().flatten().collect()
        }
    }
}



