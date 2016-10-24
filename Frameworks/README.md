This folder contains LLVM module maps for CUDA C libraries. None of these files
contribute to the build, but we need to manually add framework targets that
build from module maps when we use the Xcode Playground.
