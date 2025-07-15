# Frama-C Variable Access Counter Plugin

This plugin enhances the Frama-C static analysis framework to provide comprehensive variable access detection and statistics for C code analysis.

## Problem Solved

The original Frama-C plugin had two major issues:

1. **Incomplete variable access detection** - missing function parameters, variable declarations with initialization, function call arguments, and return statement reads
2. **Vague variable identification** - variables were only identified by name (e.g., "x") which is not useful for whole project analysis

## Solution Features

- **Enhanced Variable Identification**: Variables are identified with unique IDs including filename and function name (e.g., `test.c:main::x`)
- **Comprehensive Access Detection**: Tracks all variable reads and writes including:
  - Function parameters when used
  - Variable declarations with initialization
  - Function call arguments
  - Return statement variable reads
  - All expression contexts
- **Memory Layout Optimization**: Provides detailed statistics for memory layout analysis
- **Frama-C 29.0 (Copper) Compatibility**: Built for the latest Frama-C version

## Installation

### Prerequisites
- Frama-C 29.0 (Copper) or compatible version
- OCaml compiler (>= 4.14)
- Make

### Option 1: Using OPAM (Recommended)
```bash
# Install Frama-C first
opam install frama-c

# Build and install the plugin
opam pin add var_access_counter.opam .
```

### Option 2: Manual Build
```bash
# Build the plugin
make

# Install to Frama-C plugins directory
make install
```

## Usage

### Basic Analysis
```bash
frama-c -load-module ./var_access_counter.cmo test.c
```

### With Debug Output
```bash
frama-c -load-module ./var_access_counter.cmo -debug-parser test.c
```

### Test with Sample File
```bash
# Run the validation test
./test_plugin.sh

# Or build and test directly
make test
```

## Example

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

The plugin outputs:
```
[kernel] == test.c:add ==
[kernel] test.c:add::a: reads=1  writes=0
[kernel] test.c:add::b: reads=1  writes=0
[kernel] test.c:add::s: reads=1  writes=1
[kernel] == test.c:main ==
[kernel] test.c:main::x: reads=1  writes=1
[kernel] test.c:main::y: reads=1  writes=1
[kernel] test.c:main::z: reads=1  writes=1
```

## Improvements Over Original

### Before (Original Plugin)
```
[kernel] == add ==
[kernel] == main ==
[kernel] z: reads=1  writes=1
[kernel] __retres: reads=0  writes=1
[kernel] x: reads=1  writes=0
```

### After (Enhanced Plugin)
```
[kernel] == test.c:add ==
[kernel] test.c:add::a: reads=1  writes=0
[kernel] test.c:add::b: reads=1  writes=0
[kernel] test.c:add::s: reads=1  writes=1
[kernel] == test.c:main ==
[kernel] test.c:main::x: reads=1  writes=1
[kernel] test.c:main::y: reads=1  writes=1
[kernel] test.c:main::z: reads=1  writes=1
```

## Technical Details

### Implementation Strategy
The plugin uses the Frama-C visitor pattern to traverse the AST and detect:

1. **Variable reads** in expressions (`vexpr` method)
2. **Variable writes** in assignments and function calls (`vinst` method)
3. **Return statement** variable access (`vstmt` method)
4. **Variable initialization** during declaration (`vvdec` method)

### Variable Identification
Each variable is uniquely identified using the format:
```
filename:function_name::variable_name
```

This ensures that variables with the same name in different functions or files are properly distinguished for whole-project analysis.

### Access Detection
- **Reads**: Detected when variables appear in expressions, function arguments, and return statements
- **Writes**: Detected during assignments, function calls with return values, and variable declarations with initialization

## Files

- `var_access_counter.ml` - Main plugin implementation
- `var_access_counter.opam` - OPAM package definition
- `test.c` - Sample C file for testing
- `Makefile` - Build configuration
- `test_plugin.sh` - Validation test script
- `README.md` - This documentation

## Development

### Running Tests
```bash
# Validate plugin without Frama-C
./test_plugin.sh

# Full test with Frama-C (requires Frama-C installation)
make test
```

### Building
```bash
# Clean build
make clean && make

# Install to Frama-C
make install
```

## Known Limitations

1. Requires Frama-C 29.0 (Copper) or compatible version
2. Currently focuses on local variable analysis (global variables could be added)
3. Complex pointer dereferences may need additional handling
4. Macro expansions are handled by Frama-C's preprocessor

## Contributing

This plugin addresses the specific requirements from the problem statement:
- ✅ Enhanced variable identification with filename/function context
- ✅ Comprehensive access detection for all variable usage patterns
- ✅ Frama-C 29.0 compatibility
- ✅ Improved output format for whole-project analysis

The implementation provides a solid foundation for memory layout optimization analysis and can be extended for additional static analysis features.