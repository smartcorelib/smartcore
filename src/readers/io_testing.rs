//! This module contains functionality to test IO. It has both functions that write
//! to the file-system for end-to-end tests, but also abstractions to avoid this by
//! reading from strings instead.
use rand::distributions::{Alphanumeric, DistString};
use std::fs;
use std::io::Bytes;
use std::io::Read;
use std::io::{Chain, IoSliceMut, Take, Write};

/// Writing out a temporary csv file at a random location and cleaning
/// it up on `Drop`.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct TemporaryTextFile {
    random_path: String,
}
impl TemporaryTextFile {
    pub fn new(contents: &str) -> std::io::Result<Self> {
        let test_text_file = TemporaryTextFile {
            random_path: Alphanumeric.sample_string(&mut rand::thread_rng(), 16),
        };
        string_to_file(contents, &test_text_file.random_path)?;
        Ok(test_text_file)
    }
    pub fn path(&self) -> &str {
        &self.random_path
    }
}
/// On `Drop` we cleanup the file-system by remove the file.
impl Drop for TemporaryTextFile {
    fn drop(&mut self) {
        fs::remove_file(self.path())
            .unwrap_or_else(|_| panic!("Could not clean up temporary file {}.", self.random_path));
    }
}
/// Write out a string to file.
pub(crate) fn string_to_file(string: &str, file_path: &str) -> std::io::Result<()> {
    let mut csv_file = fs::File::create(file_path)?;
    csv_file.write_all(string.as_bytes())?;
    Ok(())
}

/// This is used an an alternative struct that implements `Read` so
/// that instead of reading from the file-system, we can test the same
///  functionality without any file-system interaction.
pub(crate) struct TestingDataSource {
    text: String,
}
impl TestingDataSource {
    pub(crate) fn new(text: &str) -> Self {
        Self {
            text: String::from(text),
        }
    }
}
/// This is the trait that also `file::File` implements, so by implementing
/// it for `TestingDataSource` we can test functionality that reads from
/// file in a more lightweight way.
impl Read for TestingDataSource {
    fn read(&mut self, _buf: &mut [u8]) -> Result<usize, std::io::Error> {
        unimplemented!()
    }

    fn read_vectored(&mut self, _bufs: &mut [IoSliceMut<'_>]) -> Result<usize, std::io::Error> {
        unimplemented!()
    }

    fn read_to_end(&mut self, _buf: &mut Vec<u8>) -> Result<usize, std::io::Error> {
        unimplemented!()
    }
    fn read_to_string(&mut self, buf: &mut String) -> Result<usize, std::io::Error> {
        <String as std::fmt::Write>::write_str(buf, &self.text).unwrap();
        Ok(0)
    }
    fn read_exact(&mut self, _buf: &mut [u8]) -> Result<(), std::io::Error> {
        unimplemented!()
    }
    fn by_ref(&mut self) -> &mut Self
    where
        Self: Sized,
    {
        unimplemented!()
    }
    fn bytes(self) -> Bytes<Self>
    where
        Self: Sized,
    {
        unimplemented!()
    }
    fn chain<R: Read>(self, _next: R) -> Chain<Self, R>
    where
        Self: Sized,
    {
        unimplemented!()
    }
    fn take(self, _limit: u64) -> Take<Self>
    where
        Self: Sized,
    {
        unimplemented!()
    }
}

#[cfg(test)]
mod test {
    use super::TestingDataSource;
    use super::{string_to_file, TemporaryTextFile};
    use std::fs;
    use std::io::Read;
    use std::path;
    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn test_temporary_text_file() {
        let path_of_temporary_file;
        {
            let hello_world_file = TemporaryTextFile::new("Hello World!")
                .expect("`hello_world_file` should be able to write file.");

            path_of_temporary_file = String::from(hello_world_file.path());
            assert_eq!(
                fs::read_to_string(&path_of_temporary_file).expect(
                    "This field should have been written by the `hello_world_file`-object."
                ),
                "Hello World!"
            )
        }
        // By now `hello_world_file` should have been dropped and the file
        // should have been cleaned up.
        assert!(!path::Path::new(&path_of_temporary_file).exists())
    }
    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn test_string_to_file() {
        let path_of_test_file = "test.file";
        let contents_of_test_file = "Hello IO-World";

        string_to_file(contents_of_test_file, path_of_test_file)
            .expect("The file should have been written out.");
        assert_eq!(
            fs::read_to_string(path_of_test_file)
                .expect("The file we test for should have been written."),
            String::from(contents_of_test_file)
        );

        // Cleanup the temporary file.
        fs::remove_file(path_of_test_file)
            .expect("The test file should exist before and be removed here.");
    }

    #[test]
    fn read_from_testing_data_source() {
        let mut test_buffer = String::new();
        let test_data_content = "Hello non-IO world!";

        TestingDataSource::new(test_data_content)
            .read_to_string(&mut test_buffer)
            .expect("Text should have been written to buffer `test_buffer`.");
        assert_eq!(test_buffer, test_data_content)
    }
}
