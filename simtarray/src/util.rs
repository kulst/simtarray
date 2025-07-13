use crate::{FinallySplit, Init, Uninit};

pub trait Sealed {}

impl Sealed for Uninit {}
impl Sealed for Init {}
impl Sealed for FinallySplit {}
