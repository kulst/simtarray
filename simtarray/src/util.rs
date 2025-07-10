use crate::{Init, Uninit};

pub trait Sealed {}

impl Sealed for Uninit {}
impl Sealed for Init {}
