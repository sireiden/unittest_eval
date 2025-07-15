# Frama-C Variable Access Counter Plugin - Implementation Summary

## Problem Statement Addressed

The original Frama-C plugin had two critical issues:
1. **Incomplete variable access detection** - missing function parameters, variable declarations, function calls, and return statements
2. **Vague variable identification** - variables only identified by name, not useful for whole-project analysis

## Solution Implemented

### 1. Enhanced Variable Identification
- **Before**: Variables identified as simple names (e.g., "x", "z")
- **After**: Unique identifiers with context (e.g., "test.c:main::x", "test.c:add::a")

### 2. Comprehensive Access Detection
The plugin now detects ALL variable accesses:
- ✅ Function parameters when used (`a`, `b` in `add` function)
- ✅ Variable declarations with initialization (`int x = 10`)
- ✅ Function call arguments (`add(x, y)`)
- ✅ Return statement reads (`return s`, `return z`)
- ✅ All expression contexts

### 3. Improved Output Format

**Original Plugin Output (Incomplete):**
```
[kernel] == add ==
[kernel] == main ==
[kernel] z: reads=1  writes=1
[kernel] __retres: reads=0  writes=1
[kernel] x: reads=1  writes=0
```

**Enhanced Plugin Output (Complete):**
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

## Key Improvements

1. **Complete Variable Detection**: Now catches all 6 variables vs only 3 in original
2. **Contextual Identification**: Variables include filename and function for unique identification
3. **Accurate Statistics**: Proper read/write counts for all detected variables
4. **Function Context**: Clear separation between different functions
5. **Whole-Project Support**: Unique IDs enable cross-file analysis

## Implementation Details

- **Language**: OCaml with Frama-C API
- **Compatibility**: Frama-C 29.0 (Copper)
- **Pattern**: Visitor pattern for AST traversal
- **Detection Methods**: 
  - `vexpr` for variable reads in expressions
  - `vinst` for assignments and function calls
  - `vstmt` for return statements
  - `vvdec` for variable declarations

## Files Delivered

- `var_access_counter.ml` - Main plugin implementation
- `test.c` - Test file demonstrating the issue and solution
- `Makefile` - Build configuration
- `var_access_counter.opam` - OPAM package definition
- `test_plugin.sh` - Validation script
- `README.md` - Comprehensive documentation
- `.gitignore` - Build artifacts exclusion

## Usage

```bash
# Build the plugin
make

# Run with test file
frama-c -load-module ./var_access_counter.cmo test.c

# Or use the test script
./test_plugin.sh
```

This implementation fully addresses the problem statement requirements and provides a robust foundation for memory layout optimization analysis.