//! This module contains utitilities to read-in matrices from csv files.
//! ```rust
//! use smartcore::readers::csv;
//! use smartcore::linalg::basic::matrix::DenseMatrix;
//! use std::fs;
//!
//! fs::write("identity.csv", "header\n1.0,0.0\n0.0,1.0");
//!
//! let mtx = csv::matrix_from_csv_source::<f64, Vec<_>, DenseMatrix<_>>(
//!     fs::File::open("identity.csv").unwrap(),
//!     csv::CSVDefinition::default()
//! )
//! .unwrap();
//! println!("{}", &mtx);
//!
//! fs::remove_file("identity.csv");
//! ```

use crate::linalg::basic::arrays::{Array1, Array2};
use crate::numbers::basenum::Number;
use crate::numbers::realnum::RealNumber;
use crate::readers::ReadingError;
use std::io::Read;

/// Define the structure of a CSV-file so that it can be read.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CSVDefinition<'a> {
    /// How many rows does the header have?
    n_rows_header: usize,
    /// What seperates the fields in your csv-file?
    field_seperator: &'a str,
}
impl<'a> Default for CSVDefinition<'a> {
    fn default() -> Self {
        Self {
            n_rows_header: 1,
            field_seperator: ",",
        }
    }
}

/// Format definition for a single row in a csv file.
/// This is used internally to validate rows of the csv file and
/// be able to fail as early as possible.
#[derive(Clone, Debug, PartialEq, Eq)]
struct CSVRowFormat<'a> {
    field_seperator: &'a str,
    n_fields: usize,
}
impl<'a> CSVRowFormat<'a> {
    fn from_csv_definition(definition: &'a CSVDefinition<'_>, n_fields: usize) -> Self {
        CSVRowFormat {
            field_seperator: definition.field_seperator,
            n_fields,
        }
    }
}

/// Detect the row format for the csv file from the first row.
fn detect_row_format<'a>(
    csv_text: &'a str,
    definition: &'a CSVDefinition<'_>,
) -> Result<CSVRowFormat<'a>, ReadingError> {
    let first_line = csv_text
        .lines()
        .nth(definition.n_rows_header)
        .ok_or(ReadingError::NoRowsProvided)?;

    Ok(CSVRowFormat::from_csv_definition(
        definition,
        first_line.split(definition.field_seperator).count(),
    ))
}

/// Read in a matrix from a source that contains a csv file.
pub fn matrix_from_csv_source<T, RowVector, Matrix>(
    source: impl Read,
    definition: CSVDefinition<'_>,
) -> Result<Matrix, ReadingError>
where
    T: Number + RealNumber + std::str::FromStr,
    RowVector: Array1<T>,
    Matrix: Array2<T>,
{
    let csv_text = read_string_from_source(source)?;
    let rows: Vec<Vec<T>> = extract_row_vectors_from_csv_text::<T, RowVector, Matrix>(
        &csv_text,
        &definition,
        detect_row_format(&csv_text, &definition)?,
    )?;
    let nrows = rows.len();
    let ncols = rows[0].len();

    // TODO: try to return ReadingError
    let array2 = Matrix::from_iterator(rows.into_iter().flatten(), nrows, ncols, 0);

    if array2.shape() != (nrows, ncols) {
        Err(ReadingError::ShapesDoNotMatch { msg: String::new() })
    } else {
        Ok(array2)
    }
}

/// Given a string containing the contents of a csv file, extract its value
/// into row-vectors.
fn extract_row_vectors_from_csv_text<
    'a,
    T: Number + RealNumber + std::str::FromStr,
    RowVector: Array1<T>,
    Matrix: Array2<T>,
>(
    csv_text: &'a str,
    definition: &'a CSVDefinition<'_>,
    row_format: CSVRowFormat<'_>,
) -> Result<Vec<Vec<T>>, ReadingError> {
    csv_text
        .lines()
        .skip(definition.n_rows_header)
        .enumerate()
        .map(|(row_index, line)| {
            enrich_reading_error(
                extract_vector_from_csv_line(line, &row_format),
                format!(", Row: {row_index}."),
            )
        })
        .collect::<Result<Vec<_>, ReadingError>>()
}

/// Read a string from source implementing `Read`.
fn read_string_from_source(mut source: impl Read) -> Result<String, ReadingError> {
    let mut string = String::new();
    source.read_to_string(&mut string)?;
    Ok(string)
}

/// Extract a vector from a single line of a csv file.
fn extract_vector_from_csv_line<T, RowVector>(
    line: &str,
    row_format: &CSVRowFormat<'_>,
) -> Result<RowVector, ReadingError>
where
    T: Number + RealNumber + std::str::FromStr,
    RowVector: Array1<T>,
{
    validate_csv_row(line, row_format)?;
    let extracted_fields: Vec<T> = extract_fields_from_csv_row(line, row_format)?;
    Ok(Array1::from_vec_slice(&extracted_fields[..]))
}

/// Extract the fields from a string containing the row of a csv file.
fn extract_fields_from_csv_row<T>(
    row: &str,
    row_format: &CSVRowFormat<'_>,
) -> Result<Vec<T>, ReadingError>
where
    T: Number + RealNumber + std::str::FromStr,
{
    row.split(row_format.field_seperator)
        .enumerate()
        .map(|(field_number, csv_field)| {
            enrich_reading_error(
                extract_value_from_csv_field(csv_field.trim()),
                format!(" Column: {field_number}"),
            )
        })
        .collect::<Result<Vec<T>, ReadingError>>()
}

/// Ensure that a string containing a csv row conforms to a specified row format.
fn validate_csv_row(row: &str, row_format: &CSVRowFormat<'_>) -> Result<(), ReadingError> {
    let actual_number_of_fields = row.split(row_format.field_seperator).count();
    if row_format.n_fields == actual_number_of_fields {
        Ok(())
    } else {
        Err(ReadingError::InvalidRow {
            msg: format!(
                "{} fields found but expected {}",
                actual_number_of_fields, row_format.n_fields
            ),
        })
    }
}

/// Add additional text to the message of an error.
/// In csv reading it is used to add the line-number / row-number
/// The error occured that is only known in the functions above.
fn enrich_reading_error<T>(
    result: Result<T, ReadingError>,
    additional_text: String,
) -> Result<T, ReadingError> {
    result.map_err(|error| ReadingError::InvalidField {
        msg: format!(
            "{}{additional_text}",
            error.message().unwrap_or("Could not serialize value")
        ),
    })
}

/// Extract the value from a single csv field.
fn extract_value_from_csv_field<T>(value_string: &str) -> Result<T, ReadingError>
where
    T: Number + RealNumber + std::str::FromStr,
{
    // By default, `FromStr::Err` does not implement `Debug`.
    // Restricting it in the library leads to many breaking
    // changes therefore I have to reconstruct my own, printable
    // error as good as possible.
    match value_string.parse::<T>().ok() {
        Some(value) => Ok(value),
        None => Err(ReadingError::InvalidField {
            msg: format!("Value '{value_string}' could not be read.",),
        }),
    }
}

#[cfg(test)]
mod tests {
    mod matrix_from_csv_source {
        use super::super::{read_string_from_source, CSVDefinition, ReadingError};
        use crate::linalg::basic::matrix::DenseMatrix;
        use crate::readers::{csv::matrix_from_csv_source, io_testing};

        #[test]
        fn read_simple_string() {
            assert_eq!(
                read_string_from_source(io_testing::TestingDataSource::new("test-string")),
                Ok(String::from("test-string"))
            )
        }
        #[test]
        fn read_simple_csv() {
            assert_eq!(
                matrix_from_csv_source::<f64, Vec<_>, DenseMatrix<_>>(
                    io_testing::TestingDataSource::new(
                        "'sepal.length','sepal.width','petal.length','petal.width'\n\
                        5.1,3.5,1.4,0.2\n\
                        4.9,3.0,1.4,0.2\n\
                        4.7,3.2,1.3,0.2",
                    ),
                    CSVDefinition::default(),
                ),
                Ok(DenseMatrix::from_2d_array(&[
                    &[5.1, 3.5, 1.4, 0.2],
                    &[4.9, 3.0, 1.4, 0.2],
                    &[4.7, 3.2, 1.3, 0.2],
                ]))
            )
        }
        #[test]
        fn read_csv_semicolon_as_seperator() {
            assert_eq!(
                matrix_from_csv_source::<f64, Vec<_>, DenseMatrix<_>>(
                    io_testing::TestingDataSource::new(
                        "'sepal.length';'sepal.width';'petal.length';'petal.width'\n\
                        'Length of sepals.';'Width of Sepals';'Length of petals';'Width of petals'\n\
                        5.1;3.5;1.4;0.2\n\
                        4.9;3.0;1.4;0.2\n\
                        4.7;3.2;1.3;0.2",
                    ),
                    CSVDefinition {
                        n_rows_header: 2,
                        field_seperator: ";"
                    },
                ),
                Ok(DenseMatrix::from_2d_array(&[
                    &[5.1, 3.5, 1.4, 0.2],
                    &[4.9, 3.0, 1.4, 0.2],
                    &[4.7, 3.2, 1.3, 0.2],
                ]).unwrap())
            )
        }
        #[test]
        fn error_in_colum_1_row_1() {
            assert_eq!(
                matrix_from_csv_source::<f64, Vec<_>, DenseMatrix<_>>(
                    io_testing::TestingDataSource::new(
                        "'sepal.length','sepal.width','petal.length','petal.width'\n\
                        5.1,3.5,1.4,0.2\n\
                        4.9,invalid,1.4,0.2\n\
                        4.7,3.2,1.3,0.2",
                    ),
                    CSVDefinition::default(),
                ),
                Err(ReadingError::InvalidField {
                    msg: String::from("Value 'invalid' could not be read. Column: 1, Row: 1.")
                })
            )
        }
        #[test]
        fn different_number_of_columns() {
            assert_eq!(
                matrix_from_csv_source::<f64, Vec<_>, DenseMatrix<_>>(
                    io_testing::TestingDataSource::new(
                        "'field_1','field_2'\n\
                    5.1,3.5\n\
                    4.9,3.0,1.4",
                    ),
                    CSVDefinition::default(),
                ),
                Err(ReadingError::InvalidField {
                    msg: String::from("3 fields found but expected 2, Row: 1.")
                })
            )
        }
    }
    mod extract_row_vectors_from_csv_text {
        use super::super::{extract_row_vectors_from_csv_text, CSVDefinition, CSVRowFormat};
        use crate::linalg::basic::matrix::DenseMatrix;

        #[test]
        fn read_default_csv() {
            assert_eq!(
                extract_row_vectors_from_csv_text::<f64, Vec<_>, DenseMatrix<_>>(
                    "column 1, column 2, column3\n1.0,2.0,3.0\n4.0,5.0,6.0",
                    &CSVDefinition::default(),
                    CSVRowFormat {
                        field_seperator: ",",
                        n_fields: 3,
                    },
                ),
                Ok(vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]])
            );
        }
    }
    mod test_validate_csv_row {
        use super::super::{validate_csv_row, CSVRowFormat, ReadingError};

        #[test]
        fn valid_row_with_comma() {
            assert_eq!(
                validate_csv_row(
                    "1.0, 2.0, 3.0",
                    &CSVRowFormat {
                        field_seperator: ",",
                        n_fields: 3,
                    },
                ),
                Ok(())
            )
        }
        #[test]
        fn valid_row_with_semicolon() {
            assert_eq!(
                validate_csv_row(
                    "1.0; 2.0; 3.0; 4.0",
                    &CSVRowFormat {
                        field_seperator: ";",
                        n_fields: 4,
                    },
                ),
                Ok(())
            )
        }
        #[test]
        fn invalid_number_of_fields() {
            assert_eq!(
                validate_csv_row(
                    "1.0; 2.0; 3.0; 4.0",
                    &CSVRowFormat {
                        field_seperator: ";",
                        n_fields: 3,
                    },
                ),
                Err(ReadingError::InvalidRow {
                    msg: String::from("4 fields found but expected 3")
                })
            )
        }
    }
    mod extract_fields_from_csv_row {
        use super::super::{extract_fields_from_csv_row, CSVRowFormat};

        #[test]
        fn read_four_values_from_csv_row() {
            assert_eq!(
                extract_fields_from_csv_row(
                    "1.0; 2.0; 3.0; 4.0",
                    &CSVRowFormat {
                        field_seperator: ";",
                        n_fields: 4
                    }
                ),
                Ok(vec![1.0, 2.0, 3.0, 4.0])
            )
        }
    }
    mod detect_row_format {
        use super::super::{detect_row_format, CSVDefinition, CSVRowFormat, ReadingError};

        #[test]
        fn detect_2_fields_with_header() {
            assert_eq!(
                detect_row_format(
                    "header-1\nheader-2\n1.0,2.0",
                    &CSVDefinition {
                        n_rows_header: 2,
                        field_seperator: ","
                    }
                )
                .expect("The row format should be detectable with this input."),
                CSVRowFormat {
                    field_seperator: ",",
                    n_fields: 2
                }
            )
        }
        #[test]
        fn detect_3_fields_no_header() {
            assert_eq!(
                detect_row_format(
                    "1.0,2.0,3.0",
                    &CSVDefinition {
                        n_rows_header: 0,
                        field_seperator: ","
                    }
                )
                .expect("The row format should be detectable with this input."),
                CSVRowFormat {
                    field_seperator: ",",
                    n_fields: 3
                }
            )
        }
        #[test]
        fn detect_no_rows_provided() {
            assert_eq!(
                detect_row_format("header\n", &CSVDefinition::default()),
                Err(ReadingError::NoRowsProvided)
            )
        }
    }
    mod extract_value_from_csv_field {
        use super::super::extract_value_from_csv_field;
        use crate::readers::ReadingError;

        #[test]
        fn deserialize_f64_from_floating_point() {
            assert_eq!(extract_value_from_csv_field::<f64>("1.0"), Ok(1.0))
        }
        #[test]
        fn deserialize_f64_from_negative_floating_point() {
            assert_eq!(extract_value_from_csv_field::<f64>("-1.0"), Ok(-1.0))
        }
        #[test]
        fn deserialize_f64_from_non_floating_point() {
            assert_eq!(extract_value_from_csv_field::<f64>("1"), Ok(1.0))
        }
        #[test]
        fn cant_deserialize_f64_from_string() {
            assert_eq!(
                extract_value_from_csv_field::<f64>("Test"),
                Err(ReadingError::InvalidField {
                    msg: String::from("Value 'Test' could not be read.")
                },)
            )
        }
        #[test]
        fn deserialize_f32_from_non_floating_point() {
            assert_eq!(extract_value_from_csv_field::<f32>("12.0"), Ok(12.0))
        }
    }
    mod extract_vector_from_csv_line {
        use super::super::{extract_vector_from_csv_line, CSVRowFormat, ReadingError};

        #[test]
        fn extract_five_floating_point_values() {
            assert_eq!(
                extract_vector_from_csv_line::<f64, Vec<f64>>(
                    "-1.0,2.0,100.0,12",
                    &CSVRowFormat {
                        field_seperator: ",",
                        n_fields: 4
                    }
                ),
                Ok(vec![-1.0, 2.0, 100.0, 12.0])
            )
        }
        #[test]
        fn cannot_extract_second_value() {
            assert_eq!(
                extract_vector_from_csv_line::<f64, Vec<f64>>(
                    "-1.0,test,100.0,12",
                    &CSVRowFormat {
                        field_seperator: ",",
                        n_fields: 4
                    }
                ),
                Err(ReadingError::InvalidField {
                    msg: String::from("Value 'test' could not be read. Column: 1")
                })
            )
        }
    }
}
