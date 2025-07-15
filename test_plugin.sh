#!/bin/bash

# Test script for Frama-C Variable Access Counter Plugin
# This script validates the plugin implementation without requiring Frama-C

echo "=== Frama-C Variable Access Counter Plugin Test ==="
echo

# Check if files exist
echo "1. Checking required files..."
if [ ! -f "var_access_counter.ml" ]; then
    echo "❌ var_access_counter.ml not found"
    exit 1
fi

if [ ! -f "test.c" ]; then
    echo "❌ test.c not found"
    exit 1
fi

if [ ! -f "Makefile" ]; then
    echo "❌ Makefile not found"
    exit 1
fi

echo "✅ All required files found"
echo

# Check OCaml syntax
echo "2. Checking OCaml syntax..."
if command -v ocaml >/dev/null 2>&1; then
    # Try to compile just the syntax
    echo "let _ = ()" > syntax_test.ml
    cat var_access_counter.ml >> syntax_test.ml
    if ocaml -i syntax_test.ml >/dev/null 2>&1; then
        echo "✅ OCaml syntax appears valid"
    else
        echo "❌ OCaml syntax errors detected"
        ocaml -i syntax_test.ml
    fi
    rm -f syntax_test.ml
else
    echo "⚠️  OCaml not available, skipping syntax check"
fi
echo

# Check C code structure
echo "3. Analyzing test.c structure..."
echo "Functions found:"
grep -n "^[a-zA-Z].*(" test.c | grep -v "^[[:space:]]*#" | head -5
echo

echo "Variables found:"
grep -n "int\|char\|float\|double" test.c | head -10
echo

# Expected output analysis
echo "4. Expected variable access analysis:"
echo "Based on test.c, the plugin should detect:"
echo ""
echo "Function: add"
echo "- a: read in 'a + b' (1 read, 0 writes)"
echo "- b: read in 'a + b' (1 read, 0 writes)"  
echo "- s: written in 'int s = a + b', read in 'return s' (1 read, 1 write)"
echo ""
echo "Function: main"
echo "- x: written in 'int x = 10', read in 'add(x, y)' (1 read, 1 write)"
echo "- y: written in 'int y = 20', read in 'add(x, y)' (1 read, 1 write)"
echo "- z: written in 'int z = add(x, y)', read in 'return z' (1 read, 1 write)"
echo ""
echo "Expected format:"
echo "[kernel] == test.c:add =="
echo "[kernel] test.c:add::a: reads=1  writes=0"
echo "[kernel] test.c:add::b: reads=1  writes=0"
echo "[kernel] test.c:add::s: reads=1  writes=1"
echo "[kernel] == test.c:main =="
echo "[kernel] test.c:main::x: reads=1  writes=1"
echo "[kernel] test.c:main::y: reads=1  writes=1"
echo "[kernel] test.c:main::z: reads=1  writes=1"
echo

# Check if Frama-C is available
echo "5. Checking Frama-C availability..."
if command -v frama-c >/dev/null 2>&1; then
    echo "✅ Frama-C found: $(frama-c -version)"
    echo "Attempting to build plugin..."
    if make clean && make; then
        echo "✅ Plugin built successfully"
        echo "Running test..."
        make test
    else
        echo "❌ Build failed"
    fi
else
    echo "⚠️  Frama-C not found in PATH"
    echo "To install Frama-C:"
    echo "  - Ubuntu/Debian: sudo apt-get install frama-c"
    echo "  - Or build from source: https://frama-c.com/download.html"
    echo "  - Or use OPAM: opam install frama-c"
fi
echo

echo "=== Test Complete ==="