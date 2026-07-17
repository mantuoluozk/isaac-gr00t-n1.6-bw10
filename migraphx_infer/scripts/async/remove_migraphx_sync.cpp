#include <migraphx/iterator_for.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/load_save.hpp>
#include <migraphx/program.hpp>

#include <iostream>
#include <string>
#include <vector>

int main(int argc, char** argv)
{
    if(argc != 3)
    {
        std::cerr << "usage: remove_migraphx_sync INPUT.mxr OUTPUT.mxr\n";
        return 2;
    }

    auto program = migraphx::load(argv[1]);
    auto* module = program.get_main_module();
    std::vector<migraphx::instruction_ref> syncs;
    for(auto ins : migraphx::iterator_for(*module))
    {
        if(ins->name() == "hip::sync_stream")
            syncs.push_back(ins);
    }
    if(syncs.size() != 1)
    {
        std::cerr << "expected exactly one hip::sync_stream, found " << syncs.size() << "\n";
        return 3;
    }

    auto sync = syncs.front();
    if(sync->inputs().size() != 1)
    {
        std::cerr << "hip::sync_stream has unexpected input count\n";
        return 4;
    }
    auto async_output = sync->inputs().front();
    module->replace_return({async_output});
    module->remove_instruction(sync);
    migraphx::save(program, argv[2]);
    std::cout << "removed hip::sync_stream and saved: " << argv[2] << "\n";
    return 0;
}
