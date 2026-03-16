# Makefile for CodeGen Blackhole Test

CXX = g++
CXXFLAGS = -std=c++17 \
           -I/root/dev/vibe_dsl \
           -I/root/dev/vibe_dsl/tilelang_repo \
           -I/root/dev/vibe_dsl/tilelang_repo/src \
           -I/root/dev/vibe_dsl/tilelang_repo/3rdparty/tvm/include \
           -I/root/dev/vibe_dsl/tilelang_repo/3rdparty/tvm/src \
           -I/root/dev/vibe_dsl/tilelang_repo/3rdparty/tvm/3rdparty/tvm-ffi/include \
           -I/root/dev/vibe_dsl/tilelang_repo/3rdparty/tvm/3rdparty/dmlc-core/include

TEST_SRC = test_codegen_blackhole.cc
TEST_BIN = test_codegen_blackhole

# Source files from tilelang
CODEGEN_SRC = /root/dev/vibe_dsl/tilelang_repo/src/target/codegen_blackhole.cc
CODEGEN_OBJ = codegen_blackhole.o

all: $(TEST_BIN)

$(CODEGEN_OBJ): $(CODEGEN_SRC)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(TEST_BIN): $(TEST_SRC) $(CODEGEN_OBJ)
	$(CXX) $(CXXFLAGS) $< $(CODEGEN_OBJ) -o $@

test: $(TEST_BIN)
	./$(TEST_BIN)

clean:
	rm -f $(CODEGEN_OBJ) $(TEST_BIN) generated_*.cc

.PHONY: all test clean
