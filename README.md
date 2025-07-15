# Frama-C Variable Access Counter Plugin

This plugin enhances the Frama-C static analysis framework to provide comprehensive variable access detection and statistics for C code analysis.

## Features

- **Enhanced Variable Identification**: Variables are identified with unique IDs including filename and function name (e.g., `test.c:main::x`)
- **Comprehensive Access Detection**: Tracks all variable reads and writes including:
  - Function parameters when used
  - Variable declarations with initialization
  - Function call arguments
  - Return statement variable reads
  - All expression contexts
- **Memory Layout Optimization**: Provides detailed statistics for memory layout analysis

## Build Instructions

### Prerequisites
- Frama-C 29.0 (Copper) or compatible version
- OCaml compiler
- Make

### Building
```bash
make
```

### Installation
```bash
make install
```

## Usage

### Basic Analysis
```bash
frama-c -load-module ./var_access_counter.cmo -debug-parser test.c
```

### Test with Sample File
```bash
make test
```

## Expected Output

For the sample `test.c` file:
```c
int add(int a, int b) {
    int s = a + b;
    return s;
}

int main() {
    int x = 10;
    int y = 20;
    int z = add(x, y);
    return z;
}
```

The plugin should output:
```
[kernel] == test.c:add ==
[kernel] test.c:add::a: reads=1  writes=0
[kernel] test.c:add::b: reads=1  writes=0
[kernel] test.c:add::s: reads=1  writes=1
[kernel] == test.c:main ==
[kernel] test.c:main::x: reads=2  writes=1
[kernel] test.c:main::y: reads=1  writes=1
[kernel] test.c:main::z: reads=1  writes=2
```

## Files

- `var_access_counter.ml` - Main plugin implementation
- `test.c` - Sample C file for testing
- `Makefile` - Build configuration
- `README.md` - This documentation

## Technical Details

The plugin uses the Frama-C visitor pattern to traverse the AST and detect:
1. **Variable reads** in expressions
2. **Variable writes** in assignments and function calls
3. **Parameter usage** in function calls
4. **Return statement** variable access
5. **Variable initialization** during declaration

Each variable is uniquely identified by combining the filename, function name, and variable name to support whole-project analysis.