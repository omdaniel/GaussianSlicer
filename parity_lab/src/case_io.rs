use std::{fs, path::Path};

use anyhow::{anyhow, Context, Result};
use glam::{Mat3, Vec3};

use crate::cpu::{matrix::Mat3Case, spd::Spd3Case, Scalar, ScalarVectorCase};

#[derive(Debug, Clone)]
pub enum CaseSet {
    ScalarVector {
        label: String,
        cases: Vec<ScalarVectorCase>,
    },
    Mat3Ops {
        label: String,
        cases: Vec<Mat3Case>,
    },
    Spd3 {
        label: String,
        cases: Vec<Spd3Case>,
    },
}

impl CaseSet {
    pub fn label(&self) -> &str {
        match self {
            CaseSet::ScalarVector { label, .. }
            | CaseSet::Mat3Ops { label, .. }
            | CaseSet::Spd3 { label, .. } => label,
        }
    }

    pub fn kind(&self) -> &'static str {
        match self {
            CaseSet::ScalarVector { .. } => "scalar_vector",
            CaseSet::Mat3Ops { .. } => "mat3_ops",
            CaseSet::Spd3 { .. } => "spd3",
        }
    }
}

/// Writes labeled case sets to JSON so other runners (Metal, etc.) can consume them.
pub fn export_case_sets_to_json<P: AsRef<Path>>(sets: &[CaseSet], path: P) -> Result<()> {
    let mut json = String::new();
    json.push_str("[\n");
    for (set_idx, set) in sets.iter().enumerate() {
        if set_idx > 0 {
            json.push_str(",\n");
        }
        match set {
            CaseSet::ScalarVector { label, cases } => {
                append_scalar_set(&mut json, label, cases);
            }
            CaseSet::Mat3Ops { label, cases } => {
                append_mat3_set(&mut json, label, cases);
            }
            CaseSet::Spd3 { label, cases } => {
                append_spd3_set(&mut json, label, cases);
            }
        }
    }
    json.push_str("\n]\n");
    fs::write(&path, json)
        .with_context(|| format!("failed to write cases JSON to {}", path.as_ref().display()))?;
    Ok(())
}

fn append_scalar_set(buf: &mut String, label: &str, cases: &[ScalarVectorCase]) {
    buf.push_str(&format!(
        "  {{\"label\": \"{}\", \"kind\": \"scalar_vector\", \"cases\": [\n",
        escape(label)
    ));
    for (idx, case) in cases.iter().enumerate() {
        if idx > 0 {
            buf.push_str(",\n");
        }
        let x = case.x.to_array();
        let y = case.y.to_array();
        buf.push_str(&format!(
            "    {{\"alpha\": {}, \"x\": [{}, {}, {}], \"y\": [{}, {}, {}]}}",
            fmt_f32(case.alpha),
            fmt_f32(x[0]),
            fmt_f32(x[1]),
            fmt_f32(x[2]),
            fmt_f32(y[0]),
            fmt_f32(y[1]),
            fmt_f32(y[2])
        ));
    }
    buf.push_str("\n  ]}");
}

fn append_mat3_set(buf: &mut String, label: &str, cases: &[Mat3Case]) {
    buf.push_str(&format!(
        "  {{\"label\": \"{}\", \"kind\": \"mat3_ops\", \"cases\": [\n",
        escape(label)
    ));
    for (idx, case) in cases.iter().enumerate() {
        if idx > 0 {
            buf.push_str(",\n");
        }
        let a = case.a.to_cols_array();
        let b = case.b.to_cols_array();
        let v = case.v.to_array();
        buf.push_str("    {\"a\": [");
        append_mat_rows(buf, &a);
        buf.push_str("], \"b\": [");
        append_mat_rows(buf, &b);
        buf.push_str("], \"v\": [");
        buf.push_str(&format!(
            "{}, {}, {}",
            fmt_f32(v[0]),
            fmt_f32(v[1]),
            fmt_f32(v[2])
        ));
        buf.push_str("]}");
    }
    buf.push_str("\n  ]}");
}

fn append_spd3_set(buf: &mut String, label: &str, cases: &[Spd3Case]) {
    buf.push_str(&format!(
        "  {{\"label\": \"{}\", \"kind\": \"spd3\", \"cases\": [\n",
        escape(label)
    ));
    for (idx, case) in cases.iter().enumerate() {
        if idx > 0 {
            buf.push_str(",\n");
        }
        let a = case.a.to_cols_array();
        let b = case.b.to_array();
        buf.push_str("    {\"a\": [");
        append_mat_rows(buf, &a);
        buf.push_str("], \"b\": [");
        buf.push_str(&format!(
            "{}, {}, {}",
            fmt_f32(b[0]),
            fmt_f32(b[1]),
            fmt_f32(b[2])
        ));
        buf.push_str("]}");
    }
    buf.push_str("\n  ]}");
}

fn append_mat_rows(buf: &mut String, cols: &[Scalar; 9]) {
    let rows = [
        [cols[0], cols[3], cols[6]],
        [cols[1], cols[4], cols[7]],
        [cols[2], cols[5], cols[8]],
    ];
    for (r_idx, row) in rows.iter().enumerate() {
        if r_idx > 0 {
            buf.push_str(", ");
        }
        buf.push_str("[");
        buf.push_str(&format!(
            "{}, {}, {}",
            fmt_f32(row[0]),
            fmt_f32(row[1]),
            fmt_f32(row[2])
        ));
        buf.push_str("]");
    }
}

fn escape(input: &str) -> String {
    input.replace('"', "\\\"")
}

fn fmt_f32(value: f32) -> String {
    if value.fract() == 0.0 {
        format!("{value:.1}")
    } else {
        format!("{value}")
    }
}

/// Loads case sets from JSON (supports both the new set schema and the legacy scalar-only array).
pub fn import_case_sets_from_json<P: AsRef<Path>>(path: P) -> Result<Vec<CaseSet>> {
    let text = fs::read_to_string(&path)
        .with_context(|| format!("failed to read cases JSON {}", path.as_ref().display()))?;
    Parser::new(&text).parse_document()
}

struct Parser<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> Parser<'a> {
    fn new(text: &'a str) -> Self {
        Self {
            data: text.as_bytes(),
            pos: 0,
        }
    }

    fn parse_document(&mut self) -> Result<Vec<CaseSet>> {
        self.skip_ws();
        self.expect_char('[')?;
        let mut sets = Vec::new();
        let mut legacy_cases = Vec::new();
        loop {
            self.skip_ws();
            if self.peek_char()? == ']' {
                self.advance(1);
                break;
            }
            let entry = self.parse_entry()?;
            match entry {
                Entry::Set(set) => sets.push(set),
                Entry::LegacyScalar(case) => legacy_cases.push(case),
            }
            self.skip_ws();
            if self.peek_char()? == ',' {
                self.advance(1);
            }
        }
        if !legacy_cases.is_empty() {
            if !sets.is_empty() {
                return Err(anyhow!("mixed legacy scalar entries with labeled sets"));
            }
            sets.push(CaseSet::ScalarVector {
                label: "legacy".to_string(),
                cases: legacy_cases,
            });
        }
        Ok(sets)
    }

    fn parse_entry(&mut self) -> Result<Entry> {
        self.skip_ws();
        self.expect_char('{')?;
        self.skip_ws();
        let next_key = self.peek_string()?;
        if next_key == "alpha" {
            let case = self.parse_scalar_body()?;
            self.skip_ws();
            self.expect_char('}')?;
            return Ok(Entry::LegacyScalar(case));
        }
        self.expect_key("label")?;
        let label = self.parse_string()?;
        self.skip_ws();
        self.expect_char(',')?;
        self.skip_ws();
        let mut kind = "scalar_vector".to_string();
        if self.peek_string()? == "kind" {
            self.expect_key("kind")?;
            kind = self.parse_string()?;
            self.skip_ws();
            self.expect_char(',')?;
            self.skip_ws();
        }
        self.expect_key("cases")?;
        let cases = match kind.as_str() {
            "scalar_vector" => {
                let list = self.parse_array(|parser| parser.parse_scalar_case())?;
                CaseSet::ScalarVector { label, cases: list }
            }
            "mat3_ops" => {
                let list = self.parse_array(|parser| parser.parse_mat_case())?;
                CaseSet::Mat3Ops { label, cases: list }
            }
            "spd3" => {
                let list = self.parse_array(|parser| parser.parse_spd_case())?;
                CaseSet::Spd3 { label, cases: list }
            }
            other => return Err(anyhow!("unknown case set kind '{other}'")),
        };
        self.skip_ws();
        self.expect_char('}')?;
        Ok(Entry::Set(cases))
    }

    fn parse_array<T, F>(&mut self, mut parser_fn: F) -> Result<Vec<T>>
    where
        F: FnMut(&mut Parser<'a>) -> Result<T>,
    {
        self.skip_ws();
        self.expect_char('[')?;
        let mut values = Vec::new();
        loop {
            self.skip_ws();
            if self.peek_char()? == ']' {
                self.advance(1);
                break;
            }
            values.push(parser_fn(self)?);
            self.skip_ws();
            if self.peek_char()? == ',' {
                self.advance(1);
            }
        }
        Ok(values)
    }

    fn parse_scalar_case(&mut self) -> Result<ScalarVectorCase> {
        self.expect_char('{')?;
        let case = self.parse_scalar_body()?;
        self.skip_ws();
        self.expect_char('}')?;
        Ok(case)
    }

    fn parse_scalar_body(&mut self) -> Result<ScalarVectorCase> {
        self.expect_key("alpha")?;
        let alpha = self.parse_float()?;
        self.skip_ws();
        self.expect_char(',')?;
        self.skip_ws();
        self.expect_key("x")?;
        let x = self.parse_vec3()?;
        self.skip_ws();
        self.expect_char(',')?;
        self.skip_ws();
        self.expect_key("y")?;
        let y = self.parse_vec3()?;
        Ok(ScalarVectorCase {
            alpha,
            x: Vec3::from_array(x),
            y: Vec3::from_array(y),
        })
    }

    fn parse_mat_case(&mut self) -> Result<Mat3Case> {
        self.expect_char('{')?;
        self.expect_key("a")?;
        let a = self.parse_mat3()?;
        self.skip_ws();
        self.expect_char(',')?;
        self.skip_ws();
        self.expect_key("b")?;
        let b = self.parse_mat3()?;
        self.skip_ws();
        self.expect_char(',')?;
        self.skip_ws();
        self.expect_key("v")?;
        let v = self.parse_vec3()?;
        self.skip_ws();
        self.expect_char('}')?;
        Ok(Mat3Case {
            a,
            b,
            v: Vec3::from_array(v),
        })
    }

    fn parse_spd_case(&mut self) -> Result<Spd3Case> {
        self.expect_char('{')?;
        self.expect_key("a")?;
        let a = self.parse_mat3()?;
        self.skip_ws();
        self.expect_char(',')?;
        self.skip_ws();
        self.expect_key("b")?;
        let b = self.parse_vec3()?;
        self.skip_ws();
        self.expect_char('}')?;
        Ok(Spd3Case {
            a,
            b: Vec3::from_array(b),
        })
    }

    fn parse_mat3(&mut self) -> Result<Mat3> {
        self.skip_ws();
        self.expect_char('[')?;
        let mut rows = [[0.0f32; 3]; 3];
        for row in 0..3 {
            self.skip_ws();
            self.expect_char('[')?;
            rows[row][0] = self.parse_float()?;
            self.expect_char(',')?;
            rows[row][1] = self.parse_float()?;
            self.expect_char(',')?;
            rows[row][2] = self.parse_float()?;
            self.skip_ws();
            self.expect_char(']')?;
            if row < 2 {
                self.skip_ws();
                self.expect_char(',')?;
            }
        }
        self.skip_ws();
        self.expect_char(']')?;
        let cols = [
            Vec3::new(rows[0][0], rows[1][0], rows[2][0]),
            Vec3::new(rows[0][1], rows[1][1], rows[2][1]),
            Vec3::new(rows[0][2], rows[1][2], rows[2][2]),
        ];
        Ok(Mat3::from_cols(cols[0], cols[1], cols[2]))
    }

    fn parse_vec3(&mut self) -> Result<[f32; 3]> {
        self.skip_ws();
        self.expect_char('[')?;
        let x = self.parse_float()?;
        self.expect_char(',')?;
        let y = self.parse_float()?;
        self.expect_char(',')?;
        let z = self.parse_float()?;
        self.skip_ws();
        self.expect_char(']')?;
        Ok([x, y, z])
    }

    fn expect_key(&mut self, expected: &str) -> Result<()> {
        self.skip_ws();
        let actual = self.parse_string()?;
        if actual != expected {
            return Err(anyhow!("expected key '{expected}', found '{actual}'"));
        }
        self.skip_ws();
        self.expect_char(':')?;
        self.skip_ws();
        Ok(())
    }

    fn parse_string(&mut self) -> Result<String> {
        self.skip_ws();
        self.expect_char('"')?;
        let mut out = String::new();
        while self.pos < self.data.len() {
            let c = self.data[self.pos] as char;
            self.pos += 1;
            match c {
                '"' => break,
                '\\' => {
                    if self.pos >= self.data.len() {
                        return Err(anyhow!("unterminated escape in string"));
                    }
                    let esc = self.data[self.pos] as char;
                    self.pos += 1;
                    out.push(match esc {
                        '"' => '"',
                        '\\' => '\\',
                        '/' => '/',
                        'b' => '\u{0008}',
                        'f' => '\u{000c}',
                        'n' => '\n',
                        'r' => '\r',
                        't' => '\t',
                        other => other,
                    });
                }
                other => out.push(other),
            }
        }
        Ok(out)
    }

    fn peek_string(&mut self) -> Result<String> {
        let saved = self.pos;
        let s = self.parse_string()?;
        self.pos = saved;
        Ok(s)
    }

    fn parse_float(&mut self) -> Result<f32> {
        self.skip_ws();
        let start = self.pos;
        while self.pos < self.data.len() {
            let c = self.data[self.pos] as char;
            if c.is_ascii_digit() || matches!(c, '-' | '+' | '.' | 'e' | 'E') {
                self.pos += 1;
            } else {
                break;
            }
        }
        if start == self.pos {
            return Err(anyhow!("expected float literal"));
        }
        let slice = std::str::from_utf8(&self.data[start..self.pos])
            .context("invalid UTF-8 in float literal")?;
        slice
            .parse::<f32>()
            .map_err(|e| anyhow!("invalid float literal {slice}: {e}"))
    }

    fn expect_char(&mut self, expected: char) -> Result<()> {
        self.skip_ws();
        let actual = self.peek_char()?;
        if actual == expected {
            self.advance(1);
            Ok(())
        } else {
            Err(anyhow!("expected '{expected}', found '{actual}'"))
        }
    }

    fn peek_char(&self) -> Result<char> {
        self.data
            .get(self.pos)
            .map(|b| *b as char)
            .ok_or_else(|| anyhow!("unexpected end of input"))
    }

    fn advance(&mut self, amount: usize) {
        self.pos = (self.pos + amount).min(self.data.len());
    }

    fn skip_ws(&mut self) {
        while self.pos < self.data.len() && (self.data[self.pos] as char).is_ascii_whitespace() {
            self.pos += 1;
        }
    }
}

enum Entry {
    Set(CaseSet),
    LegacyScalar(ScalarVectorCase),
}
