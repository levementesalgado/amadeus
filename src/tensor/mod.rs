pub mod ops;

/// N-dimensional array (row-major, contiguous).
#[derive(Clone, Debug)]
pub struct Tensor {
    data: Vec<f32>,
    shape: Vec<usize>,
}

impl Tensor {
    pub fn zeros(shape: impl Into<Vec<usize>>) -> Self {
        let shape: Vec<usize> = shape.into();
        let len: usize = shape.iter().product();
        Self {
            data: vec![0.0; len],
            shape,
        }
    }

    pub fn from_vec(data: Vec<f32>, shape: impl Into<Vec<usize>>) -> Self {
        let shape: Vec<usize> = shape.into();
        assert_eq!(data.len(), shape.iter().product::<usize>());
        Self { data, shape }
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    pub fn numel(&self) -> usize {
        self.data.len()
    }

    pub fn as_slice(&self) -> &[f32] {
        &self.data
    }

    pub fn as_mut_slice(&mut self) -> &mut [f32] {
        &mut self.data
    }

    pub fn data(&self) -> &Vec<f32> {
        &self.data
    }

    /// Number of elements in a contiguous row at the innermost dimension.
    pub fn last_dim(&self) -> usize {
        *self.shape.last().unwrap_or(&1)
    }

    /// Reshape (must preserve numel).
    pub fn reshape(mut self, new_shape: impl Into<Vec<usize>>) -> Self {
        let new_shape: Vec<usize> = new_shape.into();
        assert_eq!(self.data.len(), new_shape.iter().product::<usize>());
        self.shape = new_shape;
        self
    }
}

/// View into a tensor (e.g. a row or slice).
/// Used for matmul without allocation.
pub struct TensorView<'a> {
    pub data: &'a [f32],
    pub shape: &'a [usize],
    pub stride_0: usize, // stride of the first dimension
}

impl<'a> TensorView<'a> {
    pub fn row(&self, i: usize) -> &[f32] {
        let start = i * self.stride_0;
        &self.data[start..start + self.shape_last()]
    }

    pub fn shape_last(&self) -> usize {
        *self.shape.last().unwrap_or(&1)
    }
}

impl Tensor {
    pub fn view(&self) -> TensorView<'_> {
        let stride_0 = if self.shape.len() <= 1 {
            self.shape[0]
        } else {
            self.shape[1..].iter().product()
        };
        TensorView {
            data: &self.data,
            shape: &self.shape,
            stride_0,
        }
    }
}
