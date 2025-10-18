// Minimal OpenVDB writer: reads a float32 RAW volume and writes a .vdb FloatGrid
// Build (via Homebrew + pkg-config):
//   brew install openvdb pkg-config
//   ./scripts/build_vdb_writer.sh

#include <openvdb/openvdb.h>
#include <openvdb/tools/Interpolation.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <iostream>

struct Args {
    std::string rawPath;
    std::string outPath;
    int dimX = 0, dimY = 0, dimZ = 0;
    double spacingX = 1.0, spacingY = 1.0, spacingZ = 1.0;
};

static void usage(const char* argv0) {
    std::cerr << "Usage: " << argv0 << " --raw density.raw --dim X Y Z --spacing sx sy sz --out volume.vdb\n";
}

static bool parseArgs(int argc, char** argv, Args& a) {
    for (int i = 1; i < argc; ++i) {
        std::string key(argv[i]);
        if ((key == "--raw" || key == "-r") && i + 1 < argc) {
            a.rawPath = argv[++i];
        } else if (key == "--out" && i + 1 < argc) {
            a.outPath = argv[++i];
        } else if (key == "--dim" && i + 3 < argc) {
            a.dimX = std::atoi(argv[++i]);
            a.dimY = std::atoi(argv[++i]);
            a.dimZ = std::atoi(argv[++i]);
        } else if (key == "--spacing" && i + 3 < argc) {
            a.spacingX = std::atof(argv[++i]);
            a.spacingY = std::atof(argv[++i]);
            a.spacingZ = std::atof(argv[++i]);
        } else {
            std::cerr << "Unknown or incomplete option: " << key << "\n";
            return false;
        }
    }
    if (a.rawPath.empty() || a.outPath.empty() || a.dimX <= 0 || a.dimY <= 0 || a.dimZ <= 0) {
        return false;
    }
    return true;
}

int main(int argc, char** argv) {
    Args args;
    if (!parseArgs(argc, argv, args)) {
        usage(argv[0]);
        return 2;
    }

    const size_t voxelCount = static_cast<size_t>(args.dimX) * args.dimY * args.dimZ;
    std::vector<float> voxels(voxelCount);
    {
        FILE* f = std::fopen(args.rawPath.c_str(), "rb");
        if (!f) {
            std::perror("fopen raw");
            return 1;
        }
        const size_t read = std::fread(voxels.data(), sizeof(float), voxelCount, f);
        std::fclose(f);
        if (read != voxelCount) {
            std::cerr << "Error: RAW size mismatch, expected " << voxelCount << " floats, got " << read << "\n";
            return 1;
        }
    }

    openvdb::initialize();

    // Create a FloatGrid with background 0
    openvdb::FloatGrid::Ptr grid = openvdb::FloatGrid::create(/*background=*/0.0f);
    grid->setName("density");

    // Set a linear transform with voxel size (uniform spacing)
    // If spacing is anisotropic, this still stores uniform transform using X spacing.
    // Here we choose the average spacing to keep scale roughly correct.
    const double s = (args.spacingX + args.spacingY + args.spacingZ) / 3.0;
    grid->setTransform(openvdb::math::Transform::createLinearTransform(/*voxelSize=*/s));

    // Populate active voxels
    openvdb::FloatGrid::Accessor accessor = grid->getAccessor();

    // Memory order of RAW buffer is z-major slices, row-major within each slice (y,x)
    size_t idx = 0;
    for (int z = 0; z < args.dimZ; ++z) {
        for (int y = 0; y < args.dimY; ++y) {
            for (int x = 0; x < args.dimX; ++x, ++idx) {
                const float v = voxels[idx];
                if (v != 0.0f) {
                    accessor.setValueOn(openvdb::Coord(x, y, z), v);
                }
            }
        }
    }

    // Write to file
    try {
        openvdb::io::File file(args.outPath);
        openvdb::GridPtrVec grids;
        grids.push_back(grid);
        file.write(grids);
        file.close();
    } catch (const std::exception& e) {
        std::cerr << "OpenVDB write error: " << e.what() << "\n";
        return 1;
    }

    std::cout << "Wrote VDB: " << args.outPath << "\n";
    return 0;
}

