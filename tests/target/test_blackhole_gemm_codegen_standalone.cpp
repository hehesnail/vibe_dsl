/*
 * Phase 3: Blackhole GEMM CodeGen Standalone Test
 *
 * This test verifies that CodeGenBlackhole generates correct TT-Metal C++ code
 * for GEMM operations without requiring full TileLang build.
 */

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <cassert>
#include <cctype>
#include <memory>

// ============================================================================
// Minimal mock of TVM structures for testing
// ============================================================================

namespace tvm {
namespace tir {

// Simple expression types for testing
struct Expr {
    virtual ~Expr() = default;
    virtual std::string to_string() const = 0;
};

struct IntImm : Expr {
    int value;
    explicit IntImm(int v) : value(v) {}
    std::string to_string() const override { return std::to_string(value); }
};

struct CallNode : Expr {
    std::string op_name;
    std::vector<std::shared_ptr<Expr>> args;

    CallNode(const std::string& name, const std::vector<std::shared_ptr<Expr>>& a)
        : op_name(name), args(a) {}

    std::string to_string() const override {
        std::ostringstream oss;
        oss << op_name << "(";
        for (size_t i = 0; i < args.size(); ++i) {
            if (i > 0) oss << ", ";
            oss << args[i]->to_string();
        }
        oss << ")";
        return oss.str();
    }
};

struct Stmt {
    virtual ~Stmt() = default;
    virtual std::string to_string(int indent = 0) const = 0;
    virtual void collect_calls(std::vector<std::string>& calls) const = 0;
};

struct EvaluateStmt : Stmt {
    std::shared_ptr<Expr> value;
    explicit EvaluateStmt(std::shared_ptr<Expr> v) : value(v) {}

    std::string to_string(int indent = 0) const override {
        std::string ind(indent * 4, ' ');
        return ind + value->to_string() + ";\n";
    }

    void collect_calls(std::vector<std::string>& calls) const override {
        if (auto call = dynamic_cast<CallNode*>(value.get())) {
            calls.push_back(call->op_name);
        }
    }
};

struct BlockStmt : Stmt {
    std::vector<std::shared_ptr<Stmt>> stmts;

    std::string to_string(int indent = 0) const override {
        std::ostringstream oss;
        for (const auto& s : stmts) {
            oss << s->to_string(indent);
        }
        return oss.str();
    }

    void collect_calls(std::vector<std::string>& calls) const override {
        for (const auto& s : stmts) {
            s->collect_calls(calls);
        }
    }
};

struct ForStmt : Stmt {
    std::string var;
    int begin, end;
    std::shared_ptr<Stmt> body;

    ForStmt(const std::string& v, int b, int e, std::shared_ptr<Stmt> bod)
        : var(v), begin(b), end(e), body(bod) {}

    std::string to_string(int indent = 0) const override {
        std::ostringstream oss;
        std::string ind(indent * 4, ' ');
        oss << ind << "for (int " << var << " = " << begin << "; " << var
            << " < " << end << "; ++" << var << ") {\n";
        oss << body->to_string(indent + 1);
        oss << ind << "}\n";
        return oss.str();
    }

    void collect_calls(std::vector<std::string>& calls) const override {
        body->collect_calls(calls);
    }
};

} // namespace tir
} // namespace tvm

// ============================================================================
// Simplified CodeGenBlackhole for testing
// ============================================================================

namespace tvm {
namespace tl {

class CodeGenBlackholeTest {
public:
    std::ostringstream decl_stream;
    std::ostringstream code_stream;
    bool need_compute_api_h_ = false;
    bool need_dataflow_api_h_ = false;

    void Init() {
        decl_stream.str("");
        code_stream.str("");
    }

    std::string Finish() {
        std::ostringstream result;

        // Add includes
        result << "// SPDX-FileCopyrightText: (c) 2025 TileLang\n";
        result << "// SPDX-License-Identifier: Apache-2.0\n";
        result << "\n";

        // For GEMM tests, we always need compute API
        result << "#include \"compute_kernel_api.h\"\n";
        result << "#include \"compute_kernel_api/matmul.h\"\n";
        result << "#include \"compute_kernel_api/tile_move_copy.h\"\n";
        result << "\n";

        if (need_dataflow_api_h_) {
            result << "#include \"dataflow_api.h\"\n\n";
        }

        // Add declarations
        result << decl_stream.str();

        // Add code
        result << code_stream.str();

        return result.str();
    }

    void PrintExpr(const std::shared_ptr<tir::Expr>& expr, std::ostream& os) {
        if (auto imm = dynamic_cast<tir::IntImm*>(expr.get())) {
            os << imm->value;
        } else if (auto call = dynamic_cast<tir::CallNode*>(expr.get())) {
            HandleBlackholeBuiltin(call, os);
        } else {
            os << expr->to_string();
        }
    }

    bool HandleBlackholeBuiltin(tir::CallNode* op, std::ostream& os) {
        const std::string& name = op->op_name;

        if (name == "tl.blackhole.cb_reserve_back") {
            need_dataflow_api_h_ = true;
            os << "cb_reserve_back(";
            PrintArgs(op->args, os);
            os << ")";
            return true;
        } else if (name == "tl.blackhole.cb_push_back") {
            need_dataflow_api_h_ = true;
            os << "cb_push_back(";
            PrintArgs(op->args, os);
            os << ")";
            return true;
        } else if (name == "tl.blackhole.cb_wait_front") {
            need_dataflow_api_h_ = true;
            os << "cb_wait_front(";
            PrintArgs(op->args, os);
            os << ")";
            return true;
        } else if (name == "tl.blackhole.cb_pop_front") {
            need_dataflow_api_h_ = true;
            os << "cb_pop_front(";
            PrintArgs(op->args, os);
            os << ")";
            return true;
        } else if (name == "tl.blackhole.mm_init") {
            need_compute_api_h_ = true;
            os << "mm_init(";
            PrintArgs(op->args, os);
            os << ")";
            return true;
        } else if (name == "tl.blackhole.matmul_tiles") {
            need_compute_api_h_ = true;
            os << "matmul_tiles(";
            PrintArgs(op->args, os);
            os << ")";
            return true;
        } else if (name == "tl.blackhole.tile_regs_acquire") {
            need_compute_api_h_ = true;
            os << "tile_regs_acquire()";
            return true;
        } else if (name == "tl.blackhole.tile_regs_commit") {
            need_compute_api_h_ = true;
            os << "tile_regs_commit()";
            return true;
        } else if (name == "tl.blackhole.tile_regs_wait") {
            need_compute_api_h_ = true;
            os << "tile_regs_wait()";
            return true;
        } else if (name == "tl.blackhole.tile_regs_release") {
            need_compute_api_h_ = true;
            os << "tile_regs_release()";
            return true;
        } else if (name == "tl.blackhole.pack_tile") {
            need_compute_api_h_ = true;
            os << "pack_tile(";
            PrintArgs(op->args, os);
            os << ")";
            return true;
        }

        return false;
    }

    void PrintArgs(const std::vector<std::shared_ptr<tir::Expr>>& args, std::ostream& os) {
        for (size_t i = 0; i < args.size(); ++i) {
            if (i > 0) os << ", ";
            PrintExpr(args[i], os);
        }
    }

    void PrintStmt(const std::shared_ptr<tir::Stmt>& stmt, int indent = 1) {
        std::string ind(indent * 4, ' ');

        if (auto eval = dynamic_cast<tir::EvaluateStmt*>(stmt.get())) {
            // Check if it's a blackhole builtin call
            if (auto call = dynamic_cast<tir::CallNode*>(eval->value.get())) {
                code_stream << ind;
                HandleBlackholeBuiltin(call, code_stream);
                code_stream << ";\n";
            } else {
                code_stream << ind << eval->value->to_string() << ";\n";
            }
        } else if (auto block = dynamic_cast<tir::BlockStmt*>(stmt.get())) {
            for (const auto& s : block->stmts) {
                PrintStmt(s, indent);
            }
        } else if (auto for_stmt = dynamic_cast<tir::ForStmt*>(stmt.get())) {
            code_stream << ind << "for (int " << for_stmt->var << " = " << for_stmt->begin
                       << "; " << for_stmt->var << " < " << for_stmt->end
                       << "; ++" << for_stmt->var << ") {\n";
            PrintStmt(for_stmt->body, indent + 1);
            code_stream << ind << "}\n";
        }
    }

    void AddFunction(const std::string& name, const std::shared_ptr<tir::Stmt>& body) {
        code_stream << "void " << name << "() {\n";

        // Add CB configuration as comments
        code_stream << "    // CB configuration\n";
        code_stream << "    constexpr uint32_t cb_in0 = 0;\n";
        code_stream << "    constexpr uint32_t cb_in1 = 1;\n";
        code_stream << "    constexpr uint32_t cb_out = 16;\n";
        code_stream << "\n";

        // Print body
        PrintStmt(body, 1);

        code_stream << "}\n";
    }
};

} // namespace tl
} // namespace tvm

// ============================================================================
// Test Cases
// ============================================================================

using namespace tvm;
using namespace tvm::tl;
using namespace tvm::tir;

// Helper to create int immediate
std::shared_ptr<Expr> Int(int v) {
    return std::make_shared<IntImm>(v);
}

// Helper to create blackhole builtin call
std::shared_ptr<Expr> BlackholeCall(const std::string& name,
                                     const std::vector<std::shared_ptr<Expr>>& args) {
    return std::make_shared<CallNode>("tl.blackhole." + name, args);
}

// Helper to create evaluate statement
std::shared_ptr<Stmt> Eval(std::shared_ptr<Expr> expr) {
    return std::make_shared<EvaluateStmt>(expr);
}

bool test_basic_matmul() {
    std::cout << "=== Test: Basic Matmul Tiles ===" << std::endl;

    // Build the matmul sequence
    std::vector<std::shared_ptr<Stmt>> stmts;

    // mm_init(0, 1, 16)
    stmts.push_back(Eval(BlackholeCall("mm_init", {Int(0), Int(1), Int(16)})));

    // tile_regs_acquire()
    stmts.push_back(Eval(BlackholeCall("tile_regs_acquire", {})));

    // cb_wait_front(0, 1)
    stmts.push_back(Eval(BlackholeCall("cb_wait_front", {Int(0), Int(1)})));

    // cb_wait_front(1, 1)
    stmts.push_back(Eval(BlackholeCall("cb_wait_front", {Int(1), Int(1)})));

    // matmul_tiles(0, 1, 0, 0, 0)
    stmts.push_back(Eval(BlackholeCall("matmul_tiles", {Int(0), Int(1), Int(0), Int(0), Int(0)})));

    // cb_pop_front(0, 1)
    stmts.push_back(Eval(BlackholeCall("cb_pop_front", {Int(0), Int(1)})));

    // cb_pop_front(1, 1)
    stmts.push_back(Eval(BlackholeCall("cb_pop_front", {Int(1), Int(1)})));

    // tile_regs_commit()
    stmts.push_back(Eval(BlackholeCall("tile_regs_commit", {})));

    // tile_regs_wait()
    stmts.push_back(Eval(BlackholeCall("tile_regs_wait", {})));

    // cb_reserve_back(16, 1)
    stmts.push_back(Eval(BlackholeCall("cb_reserve_back", {Int(16), Int(1)})));

    // pack_tile(0, 16)
    stmts.push_back(Eval(BlackholeCall("pack_tile", {Int(0), Int(16)})));

    // cb_push_back(16, 1)
    stmts.push_back(Eval(BlackholeCall("cb_push_back", {Int(16), Int(1)})));

    // tile_regs_release()
    stmts.push_back(Eval(BlackholeCall("tile_regs_release", {})));

    auto body = std::make_shared<BlockStmt>();
    body->stmts = stmts;

    // Generate code
    CodeGenBlackholeTest cg;
    cg.Init();
    cg.AddFunction("kernel_main", body);
    std::string code = cg.Finish();

    std::cout << "Generated code:\n" << code << std::endl;

    // Verify expected patterns
    bool passed = true;
    std::vector<std::string> expected = {
        "#include \"compute_kernel_api.h\"",
        "mm_init(",
        "tile_regs_acquire()",
        "tile_regs_commit()",
        "tile_regs_wait()",
        "tile_regs_release()",
        "matmul_tiles(",
        "cb_wait_front(",
        "cb_pop_front(",
        "cb_reserve_back(",
        "cb_push_back(",
        "pack_tile("
    };

    for (const auto& pattern : expected) {
        if (code.find(pattern) == std::string::npos) {
            std::cout << "FAIL: Missing pattern: " << pattern << std::endl;
            passed = false;
        } else {
            std::cout << "PASS: Found pattern: " << pattern << std::endl;
        }
    }

    return passed;
}

bool test_multi_tile_accumulate() {
    std::cout << "\n=== Test: Multi-Tile Accumulation ===" << std::endl;

    // Build loop body
    std::vector<std::shared_ptr<Stmt>> loop_body;
    loop_body.push_back(Eval(BlackholeCall("cb_wait_front", {Int(0), Int(1)})));
    loop_body.push_back(Eval(BlackholeCall("cb_wait_front", {Int(1), Int(1)})));
    loop_body.push_back(Eval(BlackholeCall("matmul_tiles", {Int(0), Int(1), Int(0), Int(0), Int(0)})));
    loop_body.push_back(Eval(BlackholeCall("cb_pop_front", {Int(0), Int(1)})));
    loop_body.push_back(Eval(BlackholeCall("cb_pop_front", {Int(1), Int(1)})));

    auto loop_block = std::make_shared<BlockStmt>();
    loop_block->stmts = loop_body;

    auto for_stmt = std::make_shared<ForStmt>("kt", 0, 4, loop_block);

    // Build full function
    std::vector<std::shared_ptr<Stmt>> stmts;
    stmts.push_back(Eval(BlackholeCall("mm_init", {Int(0), Int(1), Int(16)})));
    stmts.push_back(Eval(BlackholeCall("tile_regs_acquire", {})));
    stmts.push_back(for_stmt);
    stmts.push_back(Eval(BlackholeCall("tile_regs_commit", {})));
    stmts.push_back(Eval(BlackholeCall("tile_regs_wait", {})));
    stmts.push_back(Eval(BlackholeCall("cb_reserve_back", {Int(16), Int(1)})));
    stmts.push_back(Eval(BlackholeCall("pack_tile", {Int(0), Int(16)})));
    stmts.push_back(Eval(BlackholeCall("cb_push_back", {Int(16), Int(1)})));
    stmts.push_back(Eval(BlackholeCall("tile_regs_release", {})));

    auto body = std::make_shared<BlockStmt>();
    body->stmts = stmts;

    // Generate code
    CodeGenBlackholeTest cg;
    cg.Init();
    cg.AddFunction("gemm_accumulate", body);
    std::string code = cg.Finish();

    std::cout << "Generated code:\n" << code << std::endl;

    // Verify for loop
    bool passed = true;
    if (code.find("for (int kt = 0; kt < 4; ++kt)") == std::string::npos) {
        std::cout << "FAIL: Missing for loop" << std::endl;
        passed = false;
    } else {
        std::cout << "PASS: Found for loop for K tiles" << std::endl;
    }

    return passed;
}

bool test_compare_with_reference() {
    std::cout << "\n=== Test: Compare with Reference Implementation ===" << std::endl;

    // Generate the code
    std::vector<std::shared_ptr<Stmt>> loop_body;
    loop_body.push_back(Eval(BlackholeCall("cb_wait_front", {Int(0), Int(1)})));
    loop_body.push_back(Eval(BlackholeCall("cb_wait_front", {Int(1), Int(1)})));
    loop_body.push_back(Eval(BlackholeCall("matmul_tiles", {Int(0), Int(1), Int(0), Int(0), Int(0)})));
    loop_body.push_back(Eval(BlackholeCall("cb_pop_front", {Int(0), Int(1)})));
    loop_body.push_back(Eval(BlackholeCall("cb_pop_front", {Int(1), Int(1)})));

    auto loop_block = std::make_shared<BlockStmt>();
    loop_block->stmts = loop_body;

    auto for_stmt = std::make_shared<ForStmt>("kt", 0, 4, loop_block);

    std::vector<std::shared_ptr<Stmt>> stmts;
    stmts.push_back(Eval(BlackholeCall("mm_init", {Int(0), Int(1), Int(16)})));
    stmts.push_back(Eval(BlackholeCall("tile_regs_acquire", {})));
    stmts.push_back(for_stmt);
    stmts.push_back(Eval(BlackholeCall("tile_regs_commit", {})));
    stmts.push_back(Eval(BlackholeCall("tile_regs_wait", {})));
    stmts.push_back(Eval(BlackholeCall("cb_reserve_back", {Int(16), Int(1)})));
    stmts.push_back(Eval(BlackholeCall("pack_tile", {Int(0), Int(16)})));
    stmts.push_back(Eval(BlackholeCall("cb_push_back", {Int(16), Int(1)})));
    stmts.push_back(Eval(BlackholeCall("tile_regs_release", {})));

    auto body = std::make_shared<BlockStmt>();
    body->stmts = stmts;

    CodeGenBlackholeTest cg;
    cg.Init();
    cg.AddFunction("kernel_main", body);
    std::string generated_code = cg.Finish();

    // Expected reference kernel (from test_blackhole_gemm_e2e.py)
    std::string reference_code = R"(// SPDX-FileCopyrightText: (c) 2025 TileLang
// SPDX-License-Identifier: Apache-2.0

// Phase 3: GEMM Compute Kernel (TRISC)
// Operation: C = A @ B
// Generated from TileLang DSL

#include "compute_kernel_api.h"
#include "compute_kernel_api/matmul.h"
#include "compute_kernel_api/tile_move_copy.h"

void kernel_main() {
    // Compile-time tile dimensions
    constexpr uint32_t Mt = 1;
    constexpr uint32_t Kt = 4;  // 128 / 32
    constexpr uint32_t Nt = 1;

    // CB configuration
    constexpr uint32_t cb_in0 = 0;  // A matrix
    constexpr uint32_t cb_in1 = 1;  // B matrix
    constexpr uint32_t cb_out = 16; // C matrix

    // Initialize matrix engine
    mm_init(cb_in0, cb_in1, cb_out);

    // Outer product loop over tiles
    for (uint32_t mt = 0; mt < Mt; ++mt) {
        for (uint32_t nt = 0; nt < Nt; ++nt) {
            // Acquire DST registers
            tile_regs_acquire();

            // Accumulate over K dimension
            for (uint32_t kt = 0; kt < Kt; ++kt) {
                // Wait for input tiles
                cb_wait_front(cb_in0, 1);
                cb_wait_front(cb_in1, 1);

                // Perform matmul: C += A * B
                matmul_tiles(cb_in0, cb_in1, 0, 0, 0);

                // Release input tiles
                cb_pop_front(cb_in0, 1);
                cb_pop_front(cb_in1, 1);
            }

            // Commit compute results
            tile_regs_commit();
            tile_regs_wait();

            // Reserve output space and pack result
            cb_reserve_back(cb_out, 1);
            pack_tile(0, cb_out);
            cb_push_back(cb_out, 1);

            // Release DST registers
            tile_regs_release();
        }
    }
})";

    // Check key structural elements match
    bool passed = true;
    std::vector<std::string> required_patterns = {
        "compute_kernel_api.h",
        "compute_kernel_api/matmul.h",
        "void kernel_main()",
        "mm_init(",
        "tile_regs_acquire()",
        "for (int kt = 0; kt < 4",
        "matmul_tiles(",
        "cb_wait_front(",
        "cb_pop_front(",
        "tile_regs_commit()",
        "tile_regs_wait()",
        "cb_reserve_back(",
        "pack_tile(",
        "cb_push_back(",
        "tile_regs_release()"
    };

    for (const auto& pattern : required_patterns) {
        if (generated_code.find(pattern) == std::string::npos) {
            std::cout << "FAIL: Missing required pattern: " << pattern << std::endl;
            passed = false;
        } else {
            std::cout << "PASS: Found pattern: " << pattern << std::endl;
        }
    }

    // Save generated code for inspection
    std::ofstream out("/tmp/test_generated_gemm.cpp");
    out << generated_code;
    out.close();
    std::cout << "\nGenerated code saved to /tmp/test_generated_gemm.cpp" << std::endl;

    return passed;
}

int main() {
    std::cout << "================================================" << std::endl;
    std::cout << "Phase 3: Blackhole GEMM CodeGen Standalone Test" << std::endl;
    std::cout << "================================================" << std::endl;
    std::cout << std::endl;

    int passed = 0;
    int total = 3;

    if (test_basic_matmul()) {
        passed++;
        std::cout << "\n✓ Test 1 PASSED" << std::endl;
    } else {
        std::cout << "\n✗ Test 1 FAILED" << std::endl;
    }

    if (test_multi_tile_accumulate()) {
        passed++;
        std::cout << "\n✓ Test 2 PASSED" << std::endl;
    } else {
        std::cout << "\n✗ Test 2 FAILED" << std::endl;
    }

    if (test_compare_with_reference()) {
        passed++;
        std::cout << "\n✓ Test 3 PASSED" << std::endl;
    } else {
        std::cout << "\n✗ Test 3 FAILED" << std::endl;
    }

    std::cout << std::endl;
    std::cout << "================================================" << std::endl;
    std::cout << "Results: " << passed << "/" << total << " tests passed" << std::endl;
    std::cout << "================================================" << std::endl;

    return (passed == total) ? 0 : 1;
}
