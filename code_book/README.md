# Instructions to test `quantecon_book_networks` package in codebooks

---

1. In the current branch, create a testing environment in code_book directory by `conda env create -f test_env.yml`
2. Visit the release_v0.5 branch in [`quantecon_book_networks`](https://github.com/QuantEcon/quantecon-book-networks/tree/release_v0.5) and activate the `test_networks` environment 
3. Use `flit build` to build the package
4. Use `flit install` to install the test package into the test environment
5. Run the notebooks of chapter 1 and 2 using the test environment
6. You can check the test version is installed by using `quantecon_book_networks.__version__ ` once it is imported in Python