// build.rs

fn main() {
    // Adjust the path to where your CUDA library is located
    println!("cargo:rustc-link-search=native=../build");
    println!("cargo:rustc-link-lib=static=xelis_hash_cuda");

    // Linking CUDA runtime library, make sure this path is correct
    println!("cargo:rustc-link-search=native=/usr/local/cuda-12/lib64");
    println!("cargo:rustc-link-lib=dylib=cudart");
    println!("cargo:rustc-link-lib=dylib=cudadevrt"); // Add this line
}
