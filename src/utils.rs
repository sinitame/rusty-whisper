#[derive(Debug, Clone)]
pub struct Options {
    pub eot_token: usize,
    pub sot_prev: usize,
    pub n_ctx: usize,
}

impl Default for Options {
    fn default() -> Options {
        Options {
            eot_token: 50257,
            sot_prev: 50361,
            n_ctx: 448,
        }
    }
}
