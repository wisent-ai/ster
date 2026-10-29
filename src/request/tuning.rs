//! Decode the actual CLI enum directly, without rebuilding argv or a JSON tree.

use serde::{
    Deserialize, Deserializer,
    de::{
        DeserializeSeed, EnumAccess, VariantAccess, Visitor,
        value::{EnumAccessDeserializer, StrDeserializer},
    },
};

use super::requests::Validate;
use crate::cli::{input::InputCheck, tune::TuneCommand};

pub(super) fn decode(word: &str, body: &[u8]) -> Result<TuneCommand, serde_json::Error> {
    let mut deserializer = serde_json::Deserializer::from_slice(body);
    let command = TuneCommand::deserialize(EnumAccessDeserializer::new(Operation {
        word,
        body: &mut deserializer,
    }))?;
    deserializer.end()?;
    Ok(command)
}

impl Validate for TuneCommand {
    fn validate(&self) -> Result<(), String> {
        self.check()
    }
}

struct Operation<'word, D> {
    word: &'word str,
    body: D,
}

impl<'de, 'word, D: Deserializer<'de>> EnumAccess<'de> for Operation<'word, D> {
    type Error = D::Error;
    type Variant = Self;

    fn variant_seed<V: DeserializeSeed<'de>>(self, seed: V) -> Result<(V::Value, Self), D::Error> {
        let variant = seed.deserialize(StrDeserializer::<D::Error>::new(self.word))?;
        Ok((variant, self))
    }
}

impl<'de, D: Deserializer<'de>> VariantAccess<'de> for Operation<'_, D> {
    type Error = D::Error;

    fn unit_variant(self) -> Result<(), D::Error> {
        <()>::deserialize(self.body)
    }

    fn newtype_variant_seed<V: DeserializeSeed<'de>>(self, seed: V) -> Result<V::Value, D::Error> {
        seed.deserialize(self.body)
    }

    fn tuple_variant<V: Visitor<'de>>(
        self,
        length: usize,
        visitor: V,
    ) -> Result<V::Value, D::Error> {
        self.body.deserialize_tuple(length, visitor)
    }

    fn struct_variant<V: Visitor<'de>>(
        self,
        fields: &'static [&'static str],
        visitor: V,
    ) -> Result<V::Value, D::Error> {
        self.body
            .deserialize_struct("native tuning operation", fields, visitor)
    }
}
